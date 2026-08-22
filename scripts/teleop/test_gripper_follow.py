#!/usr/bin/env python3

import time

from lerobot.teleoperators.so101_leader.config_so101_leader import (
    SO101LeaderConfig,
)
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower


LEADER_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5C82110797-if00"
)

FOLLOWER_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5C82110810-if00"
)

BODY_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

ALL_MOTORS = BODY_MOTORS + ["gripper"]

TEST_DURATION = 25.0
LOOP_INTERVAL = 0.2

INITIAL_ALIGNMENT_TOLERANCE = 5.0
BODY_SAFE_LIMIT = 20.0
BODY_DRIFT_LIMIT = 5.0

# Follower 夹爪硬限制，避免首次测试靠近端点。
GRIPPER_COMMAND_MIN = 25.0
GRIPPER_COMMAND_MAX = 65.0

# 每个 0.2 秒周期最多变化 2 个归一化单位。
MAX_GRIPPER_STEP = 2.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def main() -> None:
    leader = SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id="leader_black",
        )
    )

    follower = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id="follower_white",
        )
    )

    follower_torque_enabled = False

    try:
        print("连接 Leader 总线……")
        leader.bus.connect()

        print("连接 Follower 总线……")
        follower.bus.connect()

        print("检查双臂标定……")
        if not leader.is_calibrated:
            raise RuntimeError(
                "Leader 标定文件与舵机寄存器不一致。"
            )

        if not follower.is_calibrated:
            raise RuntimeError(
                "Follower 标定文件与舵机寄存器不一致。"
            )

        print("关闭 Leader 和 Follower 扭矩……")
        leader.bus.disable_torque(num_retry=5)
        follower.bus.disable_torque(num_retry=5)

        leader_start = leader.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )

        follower_start = follower.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )

        print("\n初始归一化位置：")
        print(
            f"{'关节':<16}"
            f"{'Leader':>11}"
            f"{'Follower':>12}"
            f"{'差值':>11}"
        )
        print("-" * 50)

        initial_errors = []

        for motor in ALL_MOTORS:
            difference = (
                leader_start[motor] - follower_start[motor]
            )

            print(
                f"{motor:<16}"
                f"{leader_start[motor]:>11.2f}"
                f"{follower_start[motor]:>12.2f}"
                f"{difference:>11.2f}"
            )

            if abs(difference) > INITIAL_ALIGNMENT_TOLERANCE:
                initial_errors.append(
                    f"{motor} 初始差值 {difference:.2f}"
                )

        for motor in BODY_MOTORS:
            if abs(leader_start[motor]) > BODY_SAFE_LIMIT:
                initial_errors.append(
                    f"Leader {motor} 不在安全中间区域："
                    f"{leader_start[motor]:.2f}"
                )

            if abs(follower_start[motor]) > BODY_SAFE_LIMIT:
                initial_errors.append(
                    f"Follower {motor} 不在安全中间区域："
                    f"{follower_start[motor]:.2f}"
                )

        if not 30.0 <= leader_start["gripper"] <= 70.0:
            initial_errors.append(
                "Leader gripper 不在 30～70 区域："
                f"{leader_start['gripper']:.2f}"
            )

        if not 30.0 <= follower_start["gripper"] <= 70.0:
            initial_errors.append(
                "Follower gripper 不在 30～70 区域："
                f"{follower_start['gripper']:.2f}"
            )

        if initial_errors:
            print("\n初始姿态不满足测试条件：")
            for error in initial_errors:
                print(f"  - {error}")

            raise RuntimeError(
                "请重新运行 align_so101_pair.py 完成对齐。"
            )

        print("\n初始姿态检查通过。")

        # 在扭矩关闭时，将 Follower 当前原始位置写成所有关节目标。
        follower_start_raw = follower.bus.sync_read(
            "Present_Position",
            normalize=False,
            num_retry=3,
        )

        print("写入 Follower 当前姿态作为保持目标……")

        for motor in ALL_MOTORS:
            follower.bus.write(
                "Goal_Position",
                motor,
                follower_start_raw[motor],
                normalize=False,
                num_retry=3,
            )

        goal_raw = follower.bus.sync_read(
            "Goal_Position",
            normalize=False,
            num_retry=3,
        )

        mismatches = {
            motor: (
                follower_start_raw[motor],
                goal_raw[motor],
            )
            for motor in ALL_MOTORS
            if follower_start_raw[motor] != goal_raw[motor]
        }

        if mismatches:
            raise RuntimeError(
                f"Follower 目标位置回读不一致：{mismatches}"
            )

        print("Follower 保持目标回读验证通过。")
        print()
        print("本次只测试夹爪：")
        print("  - 不要移动 Leader 的肩、肘和腕部")
        print("  - 只缓慢开合 Leader 夹爪")
        print("  - 建议 Leader 夹爪保持在 30～60")
        print("  - Follower 夹爪命令会被限制在 25～65")
        print("  - 出现异常时立即切断 Follower 12V")
        print()

        confirmation = input(
            "准备完成后输入 START 并按回车；"
            "其他输入将取消："
        )

        if confirmation.strip() != "START":
            print("用户取消测试。")
            return

        print("\n开启 Follower 扭矩……")
        follower.bus.enable_torque(num_retry=5)
        follower_torque_enabled = True

        print("开始 15 秒夹爪微跟随测试。\n")

        start_time = time.monotonic()
        sample_index = 0
        worst_body_drift = 0.0
        worst_gripper_error = 0.0

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= TEST_DURATION:
                break

            leader_position = leader.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=3,
            )

            follower_position = follower.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=3,
            )

            body_drifts = {
                motor: (
                    follower_position[motor]
                    - follower_start[motor]
                )
                for motor in BODY_MOTORS
            }

            max_body_drift = max(
                abs(value)
                for value in body_drifts.values()
            )

            worst_body_drift = max(
                worst_body_drift,
                max_body_drift,
            )

            if max_body_drift > BODY_DRIFT_LIMIT:
                raise RuntimeError(
                    "Follower 主体关节漂移超过阈值："
                    f"{max_body_drift:.2f} > "
                    f"{BODY_DRIFT_LIMIT:.2f}"
                )

            requested_gripper = clamp(
                leader_position["gripper"],
                GRIPPER_COMMAND_MIN,
                GRIPPER_COMMAND_MAX,
            )

            present_gripper = follower_position["gripper"]

            step = clamp(
                requested_gripper - present_gripper,
                -MAX_GRIPPER_STEP,
                MAX_GRIPPER_STEP,
            )

            commanded_gripper = clamp(
                present_gripper + step,
                GRIPPER_COMMAND_MIN,
                GRIPPER_COMMAND_MAX,
            )

            # 单关节、低频、带状态包确认的可靠写入。
            follower.bus.write(
                "Goal_Position",
                "gripper",
                commanded_gripper,
                normalize=True,
                num_retry=3,
            )

            tracking_error = (
                requested_gripper - present_gripper
            )

            worst_gripper_error = max(
                worst_gripper_error,
                abs(tracking_error),
            )

            sample_index += 1

            print(
                f"[{sample_index:02d}] "
                f"时间={elapsed:4.1f}s  "
                f"Leader={leader_position['gripper']:6.2f}  "
                f"Follower={present_gripper:6.2f}  "
                f"Command={commanded_gripper:6.2f}  "
                f"主体最大漂移={max_body_drift:5.2f}"
            )

            time.sleep(LOOP_INTERVAL)

        print("\n夹爪微跟随测试完成。")
        print(
            "测试期间主体关节最大漂移："
            f"{worst_body_drift:.2f}"
        )
        print(
            "测试期间夹爪最大瞬时跟踪误差："
            f"{worst_gripper_error:.2f}"
        )
        print("结果：测试流程正常完成。")

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止测试……")

    finally:
        if follower.bus.is_connected:
            if follower_torque_enabled:
                try:
                    print("\n关闭 Follower 扭矩……")
                    follower.bus.disable_torque(num_retry=5)
                    print("Follower 扭矩已关闭。")
                except Exception as exc:
                    print(
                        "软件关闭 Follower 扭矩失败："
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        "请立即切断 Follower 12V 电源。"
                    )

            follower.bus.disconnect(disable_torque=False)
            print("Follower 串口已关闭。")

        if leader.bus.is_connected:
            try:
                leader.bus.disable_torque(num_retry=5)
            except Exception:
                pass

            leader.bus.disconnect(disable_torque=False)
            print("Leader 串口已关闭，扭矩保持关闭。")


if __name__ == "__main__":
    main()
