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

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


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

    try:
        print("连接 Leader 总线……")
        leader.bus.connect()

        print("连接 Follower 总线……")
        follower.bus.connect()

        if not leader.is_calibrated:
            raise RuntimeError(
                "Leader 标定文件与舵机寄存器不一致。"
            )

        if not follower.is_calibrated:
            raise RuntimeError(
                "Follower 标定文件与舵机寄存器不一致。"
            )

        print("关闭 Leader 扭矩……")
        leader.bus.disable_torque(num_retry=5)

        print("关闭 Follower 扭矩……")
        follower.bus.disable_torque(num_retry=5)

        print("\n两台机械臂扭矩均已关闭。")
        print("请持续支撑机械臂，缓慢调整姿态。")
        print("完成后按 Ctrl+C。\n")

        while True:
            leader_pos = leader.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=2,
            )

            follower_pos = follower.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=2,
            )

            deltas = {
                motor: leader_pos[motor] - follower_pos[motor]
                for motor in MOTORS
            }

            max_delta = max(
                abs(value)
                for value in deltas.values()
            )

            print("\033[2J\033[H", end="")
            print("Leader / Follower 归一化姿态对齐")
            print("=" * 54)
            print(
                f"{'关节':<16}"
                f"{'Leader':>11}"
                f"{'Follower':>12}"
                f"{'差值':>11}"
            )
            print("-" * 54)

            for motor in MOTORS:
                print(
                    f"{motor:<16}"
                    f"{leader_pos[motor]:>11.1f}"
                    f"{follower_pos[motor]:>12.1f}"
                    f"{deltas[motor]:>11.1f}"
                )

            print("-" * 54)
            print(f"最大绝对差值：{max_delta:.1f}")
            print()
            print("目标：")
            print("  主体五关节尽量在 -20～+20")
            print("  gripper 尽量在 30～70")
            print("  两台机械臂每个关节差值绝对值 <= 5")
            print()
            print("完成后保持姿态不动，并按 Ctrl+C。")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n姿态对齐程序结束。")

    finally:
        try:
            if follower.bus.is_connected:
                follower.bus.disconnect(
                    disable_torque=False
                )
                print(
                    "Follower 串口已关闭，扭矩保持关闭。"
                )
        finally:
            if leader.bus.is_connected:
                leader.bus.disconnect(
                    disable_torque=False
                )
                print(
                    "Leader 串口已关闭，扭矩保持关闭。"
                )


if __name__ == "__main__":
    main()
