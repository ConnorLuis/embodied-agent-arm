#!/usr/bin/env python3

import argparse
import time
from pprint import pprint

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so101_follower.config_so101_follower import (
    SO101FollowerConfig,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import (
    SO101LeaderConfig,
)
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader


LEADER_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5C82110797-if00"
)

FOLLOWER_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5C82110810-if00"
)

ALL_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Follower 的临时工程软限位，不是舵机标定范围。
# 后续根据实际工作空间和碰撞测试再逐关节调整。
SOFT_LIMITS = {
    "shoulder_pan": (-70.0, 70.0),
    "shoulder_lift": (-60.0, 65.0),
    "elbow_flex": (-65.0, 65.0),
    "wrist_flex": (-55.0, 55.0),
    "wrist_roll": (-75.0, 75.0),
    "gripper": (15.0, 85.0),
}

# Leader 只用于检查是否已经非常接近标定端点。
LEADER_START_LIMITS = {
    "shoulder_pan": (-85.0, 85.0),
    "shoulder_lift": (-85.0, 85.0),
    "elbow_flex": (-85.0, 85.0),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-85.0, 85.0),
    "gripper": (5.0, 95.0),
}

# gain：Leader 移动 1 单位时，Follower 相对移动多少。
# max_step：每个 0.2 秒周期允许的最大命令变化。
PROFILES = {
    "gripper": {
        "gain": 0.8,
        "max_step": 2.0,
        "max_leader_delta": 50.0,
    },
    "wrist_roll": {
        "gain": 0.5,
        "max_step": 1.0,
        "max_leader_delta": 60.0,
    },
    "wrist_flex": {
        "gain": 0.4,
        "max_step": 0.8,
        "max_leader_delta": 60.0,
    },
    "shoulder_pan": {
        "gain": 0.3,
        "max_step": 0.7,
        "max_leader_delta": 45.0,
    },
    "shoulder_lift": {
        "gain": 0.10,
        "max_step": 0.20,
        "max_leader_delta": 20.0,
    },
    "elbow_flex": {
        "gain": 0.10,
        "max_step": 0.20,
        "max_leader_delta": 20.0,
    },
}

# 运行过程中持续检查 Leader 是否接近标定端点。
LEADER_RUNTIME_LIMITS = {
    "shoulder_pan": (-85.0, 85.0),
    "shoulder_lift": (-85.0, 85.0),
    "elbow_flex": (-85.0, 85.0),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-85.0, 85.0),
    "gripper": (5.0, 95.0),
}

LOOP_INTERVAL = 0.2
DEFAULT_DURATION = 30.0

LEADER_DEADBAND = 0.4
INACTIVE_DRIFT_LIMIT = 5.0
ACTIVE_TRACKING_ERROR_LIMIT = 12.0
ACTIVE_ERROR_TIMEOUT = 2.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def validate_start_positions(
    leader_start: dict[str, float],
    follower_start: dict[str, float],
) -> None:
    errors = []

    for motor in ALL_MOTORS:
        leader_min, leader_max = LEADER_START_LIMITS[motor]
        follower_min, follower_max = SOFT_LIMITS[motor]

        if not leader_min <= leader_start[motor] <= leader_max:
            errors.append(
                f"Leader {motor} 接近标定端点："
                f"{leader_start[motor]:.2f}，"
                f"允许范围 {leader_min:.0f}～{leader_max:.0f}"
            )

        if not follower_min <= follower_start[motor] <= follower_max:
            errors.append(
                f"Follower {motor} 超出当前软限位："
                f"{follower_start[motor]:.2f}，"
                f"允许范围 {follower_min:.0f}～{follower_max:.0f}"
            )

    if errors:
        print("\n启动姿态检查未通过：")
        for error in errors:
            print(f"  - {error}")

        raise RuntimeError(
            "请把超限关节移离端点。"
            "不需要让两台机械臂互相对齐。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SO-101 自然姿态单关节相对跟随测试"
    )

    parser.add_argument(
        "--joint",
        required=True,
        choices=list(PROFILES),
        help="本次允许跟随的单个关节",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="测试持续时间，默认 30 秒",
    )

    args = parser.parse_args()

    active_motor = args.joint
    duration = args.duration
    profile = PROFILES[active_motor]

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
    normal_completion = False

    leader_start = {}
    follower_start = {}
    last_command = 0.0

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

        print("确保 Leader 和 Follower 扭矩关闭……")
        leader.bus.disable_torque(num_retry=5)
        follower.bus.disable_torque(num_retry=5)

        operating_modes = {
            motor: follower.bus.read(
                "Operating_Mode",
                motor,
                normalize=False,
                num_retry=3,
            )
            for motor in ALL_MOTORS
        }

        invalid_modes = {
            motor: mode
            for motor, mode in operating_modes.items()
            if mode != OperatingMode.POSITION.value
        }

        if invalid_modes:
            raise RuntimeError(
                "Follower 存在非 POSITION 模式舵机："
                f"{invalid_modes}"
            )

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

        print("\nLeader 自然启动姿态：")
        pprint(leader_start, sort_dicts=False)

        print("\nFollower 自然启动姿态：")
        pprint(follower_start, sort_dicts=False)

        validate_start_positions(
            leader_start,
            follower_start,
        )

        print("\n启动姿态检查通过。")
        print("两台机械臂不需要互相对齐。")

        # 在扭矩关闭时锁定 Follower 当前姿态。
        follower_start_raw = follower.bus.sync_read(
            "Present_Position",
            normalize=False,
            num_retry=3,
        )

        print("\n写入 Follower 当前姿态作为保持目标……")

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
            motor: {
                "present": follower_start_raw[motor],
                "goal": goal_raw[motor],
            }
            for motor in ALL_MOTORS
            if follower_start_raw[motor] != goal_raw[motor]
        }

        if mismatches:
            raise RuntimeError(
                f"Follower 保持目标回读失败：{mismatches}"
            )

        print("Follower 启动姿态锁定目标验证通过。")

        print("\n本次测试配置：")
        print(f"  激活关节：{active_motor}")
        print(f"  相对增益：{profile['gain']}")
        print(f"  每周期最大变化：{profile['max_step']}")
        print(f"  持续时间：{duration:.0f} 秒")
        print()
        print("只有该关节会跟随 Leader。")
        print("Follower 其他五个关节保持启动姿态。")
        print("Leader 可以保持自然下垂或放在软支撑上。")
        print("出现异常时立即切断 Follower 12V。")

        confirmation = input(
            "\n准备完成后输入 START 并按回车；"
            "其他输入将取消："
        )

        if confirmation.strip() != "START":
            print("用户取消测试。")
            return

        print("\n开启 Follower 扭矩……")
        follower.bus.enable_torque(num_retry=5)
        follower_torque_enabled = True

        # 先静态观察一小段时间，确认启动无冲击。
        print("先进行 2 秒静态保持检查……")

        for _ in range(10):
            current = follower.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=3,
            )

            start_drift = max(
                abs(current[motor] - follower_start[motor])
                for motor in ALL_MOTORS
            )

            if start_drift > 3.0:
                raise RuntimeError(
                    "Follower 启动后位置变化过大："
                    f"{start_drift:.2f}"
                )

            time.sleep(LOOP_INTERVAL)

        print("静态保持正常。")
        print()
        print("请把 Leader 放在舒适、可控的起始姿态。")
        print("保持激活关节不动，1 秒后记录相对零点。")
        time.sleep(1.0)

        # 必须在用户准备完成、Follower 已稳定后记录零点，
        # 避免准备过程中移动 Leader 形成启动阶跃。
        leader_zero = leader.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )

        follower_zero = follower.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )

        last_command = follower_zero[active_motor]

        follower.bus.write(
            "Goal_Position",
            active_motor,
            last_command,
            normalize=True,
            num_retry=3,
        )

        print(
            f"零点已记录：Leader {active_motor}="
            f"{leader_zero[active_motor]:.2f}，"
            f"Follower {active_motor}="
            f"{follower_zero[active_motor]:.2f}"
        )
        print("现在可以开始缓慢移动 Leader。")
        print(
            f"开始 {duration:.0f} 秒 "
            f"{active_motor} 相对跟随测试。\n"
        )

        start_time = time.monotonic()

        worst_inactive_drift = 0.0
        worst_active_error = 0.0
        large_error_since = None
        sample_index = 0

        while True:
            elapsed = time.monotonic() - start_time

            if elapsed >= duration:
                normal_completion = True
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

            inactive_motors = [
                motor
                for motor in ALL_MOTORS
                if motor != active_motor
            ]

            max_inactive_drift = max(
                abs(
                    follower_position[motor]
                    - follower_start[motor]
                )
                for motor in inactive_motors
            )

            worst_inactive_drift = max(
                worst_inactive_drift,
                max_inactive_drift,
            )

            if max_inactive_drift > INACTIVE_DRIFT_LIMIT:
                raise RuntimeError(
                    "Follower 非活动关节漂移超过阈值："
                    f"{max_inactive_drift:.2f}"
                )

            # 所有关节都必须远离 Leader 标定端点。
            leader_limit_errors = {}

            for motor in ALL_MOTORS:
                minimum, maximum = LEADER_RUNTIME_LIMITS[motor]
                value = leader_position[motor]

                if not minimum <= value <= maximum:
                    leader_limit_errors[motor] = {
                        "value": round(value, 2),
                        "allowed": (minimum, maximum),
                    }

            if leader_limit_errors:
                raise RuntimeError(
                    "Leader 关节接近标定端点，停止测试："
                    f"{leader_limit_errors}"
                )

            leader_delta = (
                leader_position[active_motor]
                - leader_zero[active_motor]
            )

            if (
                abs(leader_delta)
                > profile["max_leader_delta"]
            ):
                raise RuntimeError(
                    f"Leader {active_motor} 相对移动过大："
                    f"{leader_delta:.2f}，允许最大值为 "
                    f"{profile['max_leader_delta']:.2f}"
                )

            if abs(leader_delta) < LEADER_DEADBAND:
                leader_delta = 0.0

            desired = (
                follower_zero[active_motor]
                + profile["gain"] * leader_delta
            )

            soft_min, soft_max = SOFT_LIMITS[active_motor]

            desired = clamp(
                desired,
                soft_min,
                soft_max,
            )

            command_delta = (
                desired - last_command
            )

            command_delta = clamp(
                command_delta,
                -profile["max_step"],
                profile["max_step"],
            )

            last_command = clamp(
                last_command + command_delta,
                soft_min,
                soft_max,
            )

            follower.bus.write(
                "Goal_Position",
                active_motor,
                last_command,
                normalize=True,
                num_retry=3,
            )

            active_error = abs(
                last_command
                - follower_position[active_motor]
            )

            worst_active_error = max(
                worst_active_error,
                active_error,
            )

            if active_error > ACTIVE_TRACKING_ERROR_LIMIT:
                if large_error_since is None:
                    large_error_since = time.monotonic()
                elif (
                    time.monotonic() - large_error_since
                    > ACTIVE_ERROR_TIMEOUT
                ):
                    raise RuntimeError(
                        "活动关节持续无法跟踪目标："
                        f"误差 {active_error:.2f}"
                    )
            else:
                large_error_since = None

            sample_index += 1

            print(
                f"[{sample_index:03d}] "
                f"时间={elapsed:5.1f}s  "
                f"LeaderΔ={leader_delta:7.2f}  "
                f"Follower={follower_position[active_motor]:7.2f}  "
                f"Command={last_command:7.2f}  "
                f"非活动最大漂移={max_inactive_drift:5.2f}"
            )

            time.sleep(LOOP_INTERVAL)

        print("\n测试时间结束。")
        print("正在缓慢返回 Follower 启动姿态……")

        target_start = follower_zero[active_motor]

        for _ in range(300):
            difference = target_start - last_command

            if abs(difference) <= 0.2:
                last_command = target_start
                follower.bus.write(
                    "Goal_Position",
                    active_motor,
                    last_command,
                    normalize=True,
                    num_retry=3,
                )
                break

            return_step = clamp(
                difference,
                -profile["max_step"],
                profile["max_step"],
            )

            last_command += return_step

            follower.bus.write(
                "Goal_Position",
                active_motor,
                last_command,
                normalize=True,
                num_retry=3,
            )

            time.sleep(LOOP_INTERVAL)

        time.sleep(1.0)

        final_position = follower.bus.sync_read(
            "Present_Position",
            normalize=True,
            num_retry=3,
        )

        return_error = abs(
            final_position[active_motor]
            - follower_zero[active_motor]
        )

        print(
            f"返回启动姿态误差：{return_error:.2f}"
        )
        print(
            "测试期间非活动关节最大漂移："
            f"{worst_inactive_drift:.2f}"
        )
        print(
            "测试期间活动关节最大跟踪误差："
            f"{worst_active_error:.2f}"
        )

        if return_error <= 3.0:
            print("结果：测试完成并已返回自然启动姿态。")
        else:
            print(
                "结果：测试完成，但返回误差偏大，"
                "关闭扭矩后机械臂可能继续自然下垂。"
            )

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，执行紧急停止。")

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
                    print("请立即切断 Follower 12V。")

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
