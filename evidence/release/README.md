# 公开证据包

此目录是从本地完整证据中抽取的最小公开包，用于让读者核验 Stage 1 的核心结论，同时避免提交原始数据、checkpoint、标定、设备路径和完整运行日志。

## 文件

| 文件 | 内容 |
|---|---|
| `stage1_summary.json` | 数据、系统、安全门禁、单回合执行、任务失败和冻结决策的机器可读汇总 |
| `v3_checkpoint_metrics.csv` | V3 16K–40K 夹爪转移专项指标 |
| `v4_checkpoint_metrics.csv` | V4 4K–20K 转移均衡实验指标 |
| `front_task_outcome.png` | 单受控策略回合结束、返回 Home 前的 Front 帧 |
| `wrist_task_outcome.png` | 同一时刻附近的 Wrist 帧 |
| `fixtures/follower_white_calibration_semantics.json` | 仅供 CPU guard 自审使用的公开语义 fixture；不含端口、用户名或设备序列号 |

## 证据解释

- `execution_pass=true` 只表示受控运动、命令确认、守卫与清理满足合同。
- `task_success=false` 表示方块没有被成功抓取；二者不矛盾。
- V4 的最佳 close recall 0.500 来自 16K，最佳 open recall 0.324 来自 20K；它们是跨 checkpoint 的最佳单项值。
- 所有 V3/V4 checkpoint 的 `passes_joint_gate` 均为 false，因此没有可继续部署的候选。

## 来源完整性

公开汇总来源于本地保留的受控回合报告、命令/轨迹 CSV、相机帧和 V3/V4 离线审计。原始受控回合报告 SHA-256：

```text
140186b5d144452692af7d9c5144f48a67e378eb525bd8d9ec81e98fc99f9850
```

任务结束帧 SHA-256：

```text
95313646aff5590855072c8048170554ba5eb5256658d94f8741981e9b179f11  front_task_outcome.png
6ef68da1e4b7682073805fed5043b4f30ae5626bbb8f74bb759af6885c41eafc  wrist_task_outcome.png
```

完整原始证据仍保留在本地项目归档中，但不是公开仓库的一部分。
