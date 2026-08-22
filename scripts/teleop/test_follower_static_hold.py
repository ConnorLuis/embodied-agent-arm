#!/usr/bin/env python3

import time
from pprint import pprint

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower


FOLLOWER_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5C82110810-if00"
)

ROBOT_ID = "follower_white"

MONITOR_DURATION_SECONDS = 10.0
MONITOR_INTERVAL_SECONDS = 0.5

# 归一化位置误差超过该值时，立即关闭扭矩并终止测试。
MAX_ALLOWED_DEVIATION = 10.0


def main() -> None:
    follower = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=ROBOT_ID,
        )
    )

    torque_enabled = False

    try:
        print("连接 Follower 总线……")
        follower.bus.connect()

        print("验证标定文件与舵机寄存器……")
        if not follower.is_calibrated:
            raise RuntimeError(
                "Follower 标定文件与舵机寄存器不一致，禁止开启扭矩。"
            )

        print("标定验证通过。")

        # 确保准备阶段扭矩关闭。
        print("确保 Follower 扭矩关闭……")
        follower.bus.disable_torque(num_retry=5)

        motors = list(follower.bus.motors)

        print("\n检查工作模式：")
        operating_modes = {
            motor: follower.bus.read(
                "Operating_Mode",
                motor,
                normalize=False,
                num_retry=3,
            )
            for motor in motors
        }
        pprint(operating_modes, sort_dicts=False)

        expected_mode = OperatingMode.POSITION.value
        invalid_modes = {
            motor: mode
            for motor, mode in operating_modes.items()
            if mode != expected_mode
        }

        if invalid_modes:
            raise RuntimeError(
                "存在非位置控制模式，禁止开启扭矩："
                f"{invalid_modes}"
            )

        print("六个舵机均处于 POSITION 模式。")

        print("\n读取 Follower 当前原始位置：")
        start_raw = follower.bus.sync_read(
            "Present_Position",
            normalize=False,
            num_retry=3,
        )
        pprint(start_raw, sort_dicts=False)

        print("\n读取 Follower 当前归一化位置：")
        start_normalized = follower.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )
        pprint(start_normalized, sort_dicts=False)

        print("\n在扭矩关闭状态下，将当前位置写入 Goal_Position……")

        # 使用逐电机可靠写入，每次写入都要求舵机返回状态包。
        for motor in motors:
            follower.bus.write(
                "Goal_Position",
                motor,
                start_raw[motor],
                normalize=False,
                num_retry=3,
            )

        print("读取 Goal_Position 进行回读验证：")
        goal_raw = follower.bus.sync_read(
            "Goal_Position",
            normalize=False,
            num_retry=3,
        )
        pprint(goal_raw, sort_dicts=False)

        goal_mismatches = {
            motor: {
                "expected": start_raw[motor],
                "actual": goal_raw[motor],
            }
            for motor in motors
            if goal_raw[motor] != start_raw[motor]
        }

        if goal_mismatches:
            raise RuntimeError(
                "Goal_Position 回读与当前位置不一致，"
                f"禁止开启扭矩：{goal_mismatches}"
            )

        print("\n目标位置回读验证通过。")
        print("Follower 开启扭矩后应保持当前姿态，不应主动大幅运动。")
        print()
        print("请做到：")
        print("  1. 一只手轻托 Follower 前臂")
        print("  2. 另一只手准备切断 Follower 12V 电源")
        print("  3. 手指离开夹爪和所有关节夹缝")
        print()

        confirmation = input(
            "准备完成后输入 ENABLE 并按回车；其他输入将取消："
        )

        if confirmation.strip() != "ENABLE":
            print("用户取消测试。")
            return

        print("\n开启 Follower 扭矩……")
        follower.bus.enable_torque(num_retry=5)
        torque_enabled = True

        print("扭矩已开启，开始监控 10 秒。")
        print("出现剧烈跳动、持续抖动或异响时，立即切断 12V。\n")

        start_time = time.monotonic()
        sample_index = 0
        worst_deviation = 0.0

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= MONITOR_DURATION_SECONDS:
                break

            current = follower.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=3,
            )

            deviations = {
                motor: current[motor] - start_normalized[motor]
                for motor in motors
            }

            max_deviation = max(
                abs(value) for value in deviations.values()
            )
            worst_deviation = max(
                worst_deviation,
                max_deviation,
            )

            sample_index += 1

            print(
                f"[{sample_index:02d}] "
                f"时间={elapsed:4.1f}s "
                f"最大位置变化={max_deviation:5.2f}"
            )

            if max_deviation > MAX_ALLOWED_DEVIATION:
                raise RuntimeError(
                    "位置变化超过安全阈值："
                    f"{max_deviation:.2f} > "
                    f"{MAX_ALLOWED_DEVIATION:.2f}"
                )

            time.sleep(MONITOR_INTERVAL_SECONDS)

        print("\n静态保持测试完成。")
        print(f"测试期间最大位置变化：{worst_deviation:.2f}")

        if worst_deviation <= 5.0:
            print("结果：通过，静态保持稳定。")
        else:
            print(
                "结果：未触发安全中止，但位置变化偏大，"
                "暂不进入遥操作。"
            )

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在关闭 Follower 扭矩……")
        raise

    finally:
        if follower.bus.is_connected:
            if torque_enabled:
                try:
                    print("\n关闭 Follower 扭矩……")
                    follower.bus.disable_torque(num_retry=5)
                    print("Follower 扭矩已关闭。")
                except Exception as exc:
                    print(
                        "警告：软件关闭扭矩失败："
                        f"{type(exc).__name__}: {exc}"
                    )
                    print("请立即切断 Follower 的 12V 电源。")

            follower.bus.disconnect(disable_torque=False)
            print("Follower 串口已关闭。")


if __name__ == "__main__":
    main()
