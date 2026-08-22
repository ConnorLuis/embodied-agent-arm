#!/usr/bin/env python3
"""SO-101 Leader -> Follower 六关节安全相对遥操作。

启动流程：
1. 连接双臂并验证标定；
2. Follower 用当前位置锁定；
3. 自动倒计时 5 秒；
4. 倒计时结束时重新记录双臂零点；
5. 六关节同时相对跟随；
6. 正常结束时在本次会话工作区内缓慢返回启动折叠姿态；
7. Ctrl+C 或异常时立即关闭 Follower 扭矩。

该版本支持从正常软工作区之外的完全折叠姿态安全启动。
该脚本同时保存控制诊断 CSV。CSV 中的 action_* 是实际发送给
Follower 的最终绝对 Goal_Position，可用于后续检查动作语义；它还不是
包含摄像头和 Episode 元数据的正式 LeRobot 训练数据集。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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

ALL_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# 一比一相对映射。它不限制最终活动范围。
GAINS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 1.0,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.0,
    "gripper": 1.0,
}

# 每个控制周期的最大目标变化，只限制速度，不限制总行程。
MAX_STEP = {
    "shoulder_pan": 1.5,
    "shoulder_lift": 1.2,
    "elbow_flex": 1.2,
    "wrist_flex": 1.5,
    "wrist_roll": 2.0,
    "gripper": 2.5,
}

# 正常结束时返回折叠启动姿态使用更保守的速度。
RETURN_STEP = {
    "shoulder_pan": 1.0,
    "shoulder_lift": 0.8,
    "elbow_flex": 0.8,
    "wrist_flex": 1.0,
    "wrist_roll": 1.2,
    "gripper": 1.5,
}

# Follower 的绝对工作区。这里没有单关节诊断阶段的相对 ±20 限制。
SOFT_LIMITS = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-60.0, 65.0),
    "elbow_flex": (-65.0, 65.0),
    "wrist_flex": (-60.0, 60.0),
    "wrist_roll": (-90.0, 90.0),
    "gripper": (5.0, 95.0),
}

# Follower 的标定归一化范围。完整折叠姿态可能位于正常软工作区之外，
# 但仍应位于标定范围内。脚本会根据启动姿态构造“本次会话工作区”：
# 允许从折叠端点向正常工作区展开，也允许正常结束时返回该启动姿态，
# 但不允许继续向启动端点外侧运动。
FOLLOWER_CALIBRATED_RANGES = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}
FOLLOWER_SENSOR_TOLERANCE = 5.0

# Leader 的标定归一化范围。完整折叠姿态可能正好位于端点，
# 因此不能像单关节诊断那样禁止靠近 ±100/0；这里只检查传感器
# 是否明显越出标定范围，并对端点外侧的小抖动做方向约束。
LEADER_CALIBRATED_RANGES = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

LEADER_SENSOR_TOLERANCE = 5.0
LEADER_ENDPOINT_GUARD = 5.0
LEADER_OUTWARD_JITTER_TOLERANCE = 2.0

DEADBAND = {
    "shoulder_pan": 0.5,
    "shoulder_lift": 0.5,
    "elbow_flex": 0.5,
    "wrist_flex": 0.5,
    "wrist_roll": 0.4,
    "gripper": 0.4,
}

TRACKING_ERROR_LIMIT = {
    "shoulder_pan": 12.0,
    "shoulder_lift": 12.0,
    "elbow_flex": 12.0,
    "wrist_flex": 12.0,
    "wrist_roll": 12.0,
    "gripper": 15.0,
}

TRACKING_ERROR_TIMEOUT = 2.0
COUNTDOWN_DRIFT_LIMIT = 3.0
START_GOAL_TOLERANCE = 1.5
RETURN_TOLERANCE = 0.5
RETURN_TIMEOUT_SECONDS = 15.0
PRINT_INTERVAL_SECONDS = 0.5


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def assert_finite_positions(label: str, positions: dict[str, float]) -> None:
    missing = [motor for motor in ALL_MOTORS if motor not in positions]
    if missing:
        raise RuntimeError(f"{label} 缺少关节数据：{missing}")

    invalid = {
        motor: value
        for motor, value in positions.items()
        if motor in ALL_MOTORS and not math.isfinite(float(value))
    }
    if invalid:
        raise RuntimeError(f"{label} 出现非有限数值：{invalid}")


def check_follower_sensor_range(positions: dict[str, float]) -> None:
    """只拦截明显越出 Follower 标定范围的异常读数。"""
    violations: dict[str, dict[str, object]] = {}
    for motor in ALL_MOTORS:
        value = float(positions[motor])
        minimum, maximum = FOLLOWER_CALIBRATED_RANGES[motor]
        allowed_minimum = minimum - FOLLOWER_SENSOR_TOLERANCE
        allowed_maximum = maximum + FOLLOWER_SENSOR_TOLERANCE

        if not allowed_minimum <= value <= allowed_maximum:
            violations[motor] = {
                "value": round(value, 2),
                "allowed": (allowed_minimum, allowed_maximum),
            }

    if violations:
        raise RuntimeError(
            "Follower 位置读数明显越出标定范围，已停止遥操作："
            f"{violations}"
        )


def build_follower_session_limits(
    follower_zero: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """根据启动姿态构造本次会话的安全绝对工作区。

    - 启动姿态在正常软限位内：使用正常软限位；
    - 启动姿态低于软下限：允许 [启动值, 正常软上限]；
    - 启动姿态高于软上限：允许 [正常软下限, 启动值]。

    因此可以从完全折叠姿态向内展开，也能正常返回启动姿态，
    但不会继续向机械端点外侧发送目标。
    """
    limits: dict[str, tuple[float, float]] = {}
    for motor in ALL_MOTORS:
        zero = float(follower_zero[motor])
        soft_min, soft_max = SOFT_LIMITS[motor]

        if zero < soft_min:
            limits[motor] = (zero, soft_max)
        elif zero > soft_max:
            limits[motor] = (soft_min, zero)
        else:
            limits[motor] = (soft_min, soft_max)

    return limits


def get_extended_follower_joints(
    follower_zero: dict[str, float],
) -> dict[str, dict[str, object]]:
    extended: dict[str, dict[str, object]] = {}
    for motor in ALL_MOTORS:
        value = float(follower_zero[motor])
        soft_min, soft_max = SOFT_LIMITS[motor]
        if value < soft_min or value > soft_max:
            extended[motor] = {
                "start": round(value, 2),
                "normal_soft_limits": (soft_min, soft_max),
            }
    return extended

def check_leader_sensor_range(positions: dict[str, float]) -> None:
    """仅拦截明显越出标定范围的异常读数。"""
    violations: dict[str, dict[str, object]] = {}
    for motor in ALL_MOTORS:
        value = float(positions[motor])
        minimum, maximum = LEADER_CALIBRATED_RANGES[motor]
        allowed_minimum = minimum - LEADER_SENSOR_TOLERANCE
        allowed_maximum = maximum + LEADER_SENSOR_TOLERANCE

        if not allowed_minimum <= value <= allowed_maximum:
            violations[motor] = {
                "value": round(value, 2),
                "allowed": (
                    allowed_minimum,
                    allowed_maximum,
                ),
            }

    if violations:
        raise RuntimeError(
            "Leader 位置读数明显越出标定范围，已停止遥操作："
            f"{violations}"
        )


def get_endpoint_aware_delta(
    motor: str,
    zero: float,
    current: float,
) -> float:
    """先裁剪 Leader 读数，再计算相对于启动零点的位移。

    允许从接近端点的启动位置移动到合法标定端点；
    对标定范围之外的传感器抖动进行裁剪，避免其放大为动作命令。
    """
    minimum, maximum = LEADER_CALIBRATED_RANGES[motor]
    bounded_current = clamp(current, minimum, maximum)
    return bounded_current - zero


def max_abs_difference(
    left: dict[str, float],
    right: dict[str, float],
) -> tuple[str, float]:
    motor = max(
        ALL_MOTORS,
        key=lambda name: abs(float(left[name]) - float(right[name])),
    )
    difference = abs(float(left[motor]) - float(right[motor]))
    return motor, difference


def make_log_writer(
    log_dir: Path,
) -> tuple[object, csv.DictWriter, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"relative_teleop_{stamp}.csv"
    handle = log_path.open("w", newline="", encoding="utf-8")

    fieldnames = ["time_s"]
    for motor in ALL_MOTORS:
        fieldnames.extend(
            (
                f"leader_{motor}",
                f"leader_delta_{motor}",
                f"follower_{motor}",
                f"action_{motor}",
            )
        )

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer, log_path


def write_log_row(
    writer: csv.DictWriter,
    elapsed: float,
    leader_position: dict[str, float],
    leader_delta: dict[str, float],
    follower_position: dict[str, float],
    final_action: dict[str, float],
) -> None:
    row: dict[str, float] = {"time_s": round(elapsed, 6)}
    for motor in ALL_MOTORS:
        row[f"leader_{motor}"] = float(leader_position[motor])
        row[f"leader_delta_{motor}"] = float(leader_delta[motor])
        row[f"follower_{motor}"] = float(follower_position[motor])
        row[f"action_{motor}"] = float(final_action[motor])
    writer.writerow(row)


def smooth_return_to_start(
    follower: SO101Follower,
    current_command: dict[str, float],
    target: dict[str, float],
    session_limits: dict[str, tuple[float, float]],
    hz: float,
) -> dict[str, float]:
    print("\n正常测试结束，正在缓慢返回启动折叠姿态……")
    period = 1.0 / hz
    deadline = time.monotonic() + RETURN_TIMEOUT_SECONDS
    command = dict(current_command)

    while time.monotonic() < deadline:
        finished = True
        for motor in ALL_MOTORS:
            difference = float(target[motor]) - float(command[motor])
            if abs(difference) > RETURN_TOLERANCE:
                finished = False
                step = clamp(
                    difference,
                    -RETURN_STEP[motor],
                    RETURN_STEP[motor],
                )
                command[motor] = clamp(
                    float(command[motor]) + step,
                    *session_limits[motor],
                )
            else:
                command[motor] = float(target[motor])

        follower.bus.sync_write(
            "Goal_Position",
            command,
            normalize=True,
            num_retry=1,
        )

        if finished:
            break
        time.sleep(period)

    time.sleep(0.5)
    final_position = follower.bus.sync_read(
        "Present_Position",
        normalize=True,
        num_retry=3,
    )
    motor, error = max_abs_difference(final_position, target)
    print(f"返回后最大姿态误差：{motor}={error:.2f}")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SO-101 六关节 Leader→Follower 安全相对遥操作",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="遥操作持续秒数；设为 0 表示运行到 Ctrl+C，默认 30。",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="自动开始前倒计时秒数，默认 5。",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=5.0,
        help="控制频率，首次测试默认 5 Hz。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/teleop"),
        help="CSV 日志目录。",
    )
    args = parser.parse_args()

    if args.duration < 0:
        parser.error("--duration 不能小于 0。")
    if args.countdown < 1:
        parser.error("--countdown 必须至少为 1。")
    if not 1.0 <= args.hz <= 20.0:
        parser.error("首次验证要求 --hz 在 1～20 之间。")
    return args


def main() -> int:
    args = parse_args()

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

    leader_connected = False
    follower_connected = False
    follower_torque_enabled = False
    normal_completion = False
    log_handle = None
    last_command: dict[str, float] = {}
    follower_zero: dict[str, float] = {}
    session_limits: dict[str, tuple[float, float]] = {}

    try:
        print("连接 Leader 总线……")
        leader.bus.connect()
        leader_connected = True

        print("连接 Follower 总线……")
        follower.bus.connect()
        follower_connected = True

        print("验证双臂标定……")
        if not leader.is_calibrated:
            raise RuntimeError(
                "Leader 标定不一致，停止。不要运行自动标定或 setup-motors。"
            )
        if not follower.is_calibrated:
            raise RuntimeError(
                "Follower 标定不一致，停止。不要运行自动标定或 setup-motors。"
            )
        print("双臂标定验证通过。")

        print("确保 Leader 和 Follower 扭矩关闭……")
        leader.bus.disable_torque(num_retry=5)
        follower.bus.disable_torque(num_retry=5)

        modes = follower.bus.sync_read(
            "Operating_Mode",
            normalize=False,
            num_retry=3,
        )
        invalid_modes = {
            motor: value
            for motor, value in modes.items()
            if int(value) != 0
        }
        if invalid_modes:
            raise RuntimeError(
                "Follower 存在非 POSITION 模式关节，停止："
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
        assert_finite_positions("Leader 启动姿态", leader_start)
        assert_finite_positions("Follower 启动姿态", follower_start)
        check_leader_sensor_range(leader_start)
        check_follower_sensor_range(follower_start)

        print("\nLeader 启动姿态：")
        print({motor: round(float(leader_start[motor]), 2) for motor in ALL_MOTORS})
        print("\nFollower 启动姿态：")
        print({motor: round(float(follower_start[motor]), 2) for motor in ALL_MOTORS})

        print("\n将 Follower 当前六关节位置写为保持目标……")
        follower.bus.sync_write(
            "Goal_Position",
            follower_start,
            normalize=True,
            num_retry=1,
        )
        goal_readback = follower.bus.sync_read(
            "Goal_Position",
            normalize=True,
            num_retry=3,
        )
        motor, error = max_abs_difference(goal_readback, follower_start)
        if error > START_GOAL_TOLERANCE:
            raise RuntimeError(
                "Follower 启动目标回读不一致："
                f"{motor} 误差={error:.2f}"
            )
        print("Follower 六关节启动目标回读通过。")

        print("开启 Follower 扭矩……")
        follower.bus.enable_torque(num_retry=5)
        follower_torque_enabled = True

        countdown_reference = dict(follower_start)
        print(
            "\n无需输入 START。请保持双臂处于折叠但未顶死的姿态，"
            "倒计时结束前不要移动 Leader。"
        )

        for remaining in range(args.countdown, 0, -1):
            print(f"{remaining}……", flush=True)
            time.sleep(1.0)
            current = follower.bus.sync_read(
                "Present_Position",
                normalize=True,
                num_retry=3,
            )
            motor, drift = max_abs_difference(
                current,
                countdown_reference,
            )
            if drift > COUNTDOWN_DRIFT_LIMIT:
                raise RuntimeError(
                    "倒计时静态保持期间 Follower 漂移过大："
                    f"{motor}={drift:.2f}"
                )

        # 必须在倒计时结束时重新置零，避免准备阶段姿态变化造成启动跳变。
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
        assert_finite_positions("Leader 零点", leader_zero)
        assert_finite_positions("Follower 零点", follower_zero)
        check_leader_sensor_range(leader_zero)
        check_follower_sensor_range(follower_zero)

        session_limits = build_follower_session_limits(follower_zero)
        extended_joints = get_extended_follower_joints(follower_zero)
        if extended_joints:
            print("\n检测到 Follower 从正常软工作区之外的折叠姿态启动：")
            print(extended_joints)
            print(
                "本次会话允许从该折叠姿态向正常工作区展开，并允许返回；"
                "不会向折叠端点外侧继续发送目标。"
            )

        last_command = {
            motor: float(follower_zero[motor])
            for motor in ALL_MOTORS
        }
        follower.bus.sync_write(
            "Goal_Position",
            last_command,
            normalize=True,
            num_retry=1,
        )

        log_handle, log_writer, log_path = make_log_writer(args.log_dir)

        print("\n零点已记录，开始六关节整体遥操作。")
        print("映射：六关节相对变化 1:1；没有相对 ±20 限制。")
        print("正常结束会缓慢返回启动折叠姿态；Ctrl+C/异常会立即关闭扭矩。")
        print(f"控制诊断日志：{log_path}")
        print("现在开始移动 Leader。\n")

        period = 1.0 / args.hz
        start_time = time.monotonic()
        next_tick = start_time
        last_print_time = -PRINT_INTERVAL_SECONDS
        error_since: dict[str, float | None] = {
            motor: None for motor in ALL_MOTORS
        }

        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if args.duration > 0 and elapsed >= args.duration:
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
            assert_finite_positions("Leader 实时位置", leader_position)
            assert_finite_positions("Follower 实时位置", follower_position)
            check_leader_sensor_range(leader_position)

            leader_delta: dict[str, float] = {}
            final_action: dict[str, float] = {}
            limited_motors: list[str] = []

            for motor in ALL_MOTORS:
                delta = get_endpoint_aware_delta(
                    motor,
                    float(leader_zero[motor]),
                    float(leader_position[motor]),
                )
                if abs(delta) < DEADBAND[motor]:
                    delta = 0.0
                leader_delta[motor] = delta

                desired_unclamped = (
                    float(follower_zero[motor])
                    + GAINS[motor] * delta
                )
                session_min, session_max = session_limits[motor]
                desired = clamp(
                    desired_unclamped,
                    session_min,
                    session_max,
                )
                if abs(desired - desired_unclamped) > 1e-6:
                    limited_motors.append(motor)

                command_delta = clamp(
                    desired - float(last_command[motor]),
                    -MAX_STEP[motor],
                    MAX_STEP[motor],
                )
                final_action[motor] = clamp(
                    float(last_command[motor]) + command_delta,
                    session_min,
                    session_max,
                )

            # 六关节在同一个 sync_write 包中发送。
            follower.bus.sync_write(
                "Goal_Position",
                final_action,
                normalize=True,
                num_retry=1,
            )
            last_command = dict(final_action)

            tracking_errors = {
                motor: abs(
                    float(final_action[motor])
                    - float(follower_position[motor])
                )
                for motor in ALL_MOTORS
            }

            for motor, tracking_error in tracking_errors.items():
                if tracking_error > TRACKING_ERROR_LIMIT[motor]:
                    if error_since[motor] is None:
                        error_since[motor] = now
                    elif now - float(error_since[motor]) > TRACKING_ERROR_TIMEOUT:
                        raise RuntimeError(
                            "Follower 关节持续无法跟踪目标："
                            f"{motor} 误差={tracking_error:.2f}"
                        )
                else:
                    error_since[motor] = None

            write_log_row(
                log_writer,
                elapsed,
                leader_position,
                leader_delta,
                follower_position,
                final_action,
            )
            log_handle.flush()

            if elapsed - last_print_time >= PRINT_INTERVAL_SECONDS:
                worst_motor = max(
                    ALL_MOTORS,
                    key=lambda motor: tracking_errors[motor],
                )
                delta_text = " ".join(
                    f"{motor}={leader_delta[motor]:+.1f}"
                    for motor in ALL_MOTORS
                )
                suffix = (
                    f"  会话限位={','.join(sorted(set(limited_motors)))}"
                    if limited_motors
                    else ""
                )
                print(
                    f"[{elapsed:5.1f}s] LΔ {delta_text}  "
                    f"最大跟踪误差={worst_motor}:"
                    f"{tracking_errors[worst_motor]:.2f}{suffix}"
                )
                last_print_time = elapsed

            next_tick += period
            sleep_seconds = next_tick - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                next_tick = time.monotonic()

        if normal_completion:
            last_command = smooth_return_to_start(
                follower,
                last_command,
                follower_zero,
                session_limits,
                args.hz,
            )
            print("六关节整体遥操作测试正常完成。")

        return 0

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C：不执行自动回程，立即进入扭矩关闭流程。")
        return 130
    except Exception as exc:
        print(f"\n测试异常：{exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

        if follower_connected:
            if follower_torque_enabled:
                try:
                    print("关闭 Follower 扭矩……")
                    follower.bus.disable_torque(num_retry=5)
                    print("Follower 扭矩已关闭。")
                except Exception as exc:
                    print(
                        f"警告：软件关闭 Follower 扭矩失败：{exc}。"
                        "请立即切断 Follower 12V。",
                        file=sys.stderr,
                    )
            try:
                follower.bus.disconnect(disable_torque=False)
                print("Follower 串口已关闭。")
            except Exception as exc:
                print(f"Follower 串口关闭失败：{exc}", file=sys.stderr)

        if leader_connected:
            try:
                leader.bus.disable_torque(num_retry=3)
            except Exception:
                pass
            try:
                leader.bus.disconnect(disable_torque=False)
                print("Leader 串口已关闭，扭矩保持关闭。")
            except Exception as exc:
                print(f"Leader 串口关闭失败：{exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
