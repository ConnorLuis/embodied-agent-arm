# embodied-agent-arm

基于低成本主从机械臂、LeRobot、双视角感知与 LLM Agent 的具身智能操作项目。

## Project Goal

构建一个完整的具身智能闭环：

Natural Language Task
→ Visual Perception
→ High-level Planning
→ Robot Policy / Skill
→ Physical Execution
→ Visual Verification

## Hardware

- Hiwonder SO-ARM101-compatible Leader arm
- Hiwonder SO-ARM101-compatible Follower arm
- Wrist-view USB camera
- Third-person USB camera
- HX series magnetic encoder bus servos
- USB servo bus adapters

## Planned Milestones

- M0: Repository and environment initialization
- M1: Robot connection, calibration and joint-state reading
- M2: Leader-Follower teleoperation
- M3: Dual-camera data collection and replay
- M4: ACT behavior-cloning training and deployment
- M5: LLM Agent / MCP high-level task integration
- M6: Evaluation, failure analysis and demonstration video

## Status

Work in progress.
