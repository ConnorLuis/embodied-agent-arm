# 厂商源码记录

## 1. 来源

- 设备厂商：幻尔
- 源码包名称：`lerobot.zip`
- 获取日期：2026-07-12
- 使用目的：SO-ARM101 兼容机械臂的标定、遥操作、双摄像头数据采集、策略训练与真实推理
- WSL 原始压缩包：`vendor_packages/hiwonder/lerobot.zip`
- WSL 解压目录：`vendor/hiwonder/lerobot`

## 2. 文件校验

```text
lerobot.zip
SHA256：b404af16c4a5a58d137704cb9b3dadc2589d841f26f734ab221285628a518cbf
```

该校验值用于确认后续使用的源码包与厂商最初提供的文件一致。

## 3. 源码版本

- Git 远程仓库：`https://github.com/huggingface/lerobot.git`
- 分支：`main`
- Commit：`882c80d446a63a44868c67ae535467af32ce0e80`
- Commit 日期：2025-08-29
- LeRobot 版本：`0.3.4`
- Python 要求：`>=3.10`
- Ruff 目标版本：`py310`
- 当前项目 Python：`3.10.18`

当前 Conda 环境 `embodiedarm` 与该源码的 Python 版本要求兼容。

## 4. 硬件与依赖

实际硬件：

- Leader 主臂：HX-10HM
- Follower 从臂：HX-30HM
- 控制器：USB 总线舵机驱动模块
- LeRobot 类型：`so101_leader`、`so101_follower`

`pyproject.toml` 中的可选依赖：

```text
feetech = ["feetech-servo-sdk>=1.0.0"]
```

厂商文档要求：

```bash
pip install -e ".[feetech]"
```

在 `src/` 和 `pyproject.toml` 中没有检索到 `HX-10HM`、`HX-30HM` 或 `hiwonder` 字符串。这只能说明源码没有通过显式型号名称建立独立驱动，不能据此认定硬件不兼容。兼容性可能来自 Feetech 协议、标准 SO-101 抽象层或通用 MotorBus 修改。

## 5. 厂商源码修改

压缩包解压后，Git 工作区不是完全干净。

确认存在真实修改的核心文件：

```text
src/lerobot/cameras/utils.py
src/lerobot/configs/policies.py
src/lerobot/motors/motors_bus.py
```

修改规模：

```text
1 行新增，0 行删除：src/lerobot/cameras/utils.py
29 行新增，22 行删除：src/lerobot/configs/policies.py
6 行新增，6 行删除：src/lerobot/motors/motors_bus.py
```

使用 `git diff --ignore-space-at-eol` 后仍有差异，因此不是单纯的行尾格式变化。

### 5.1 摄像头修改

Windows 分支新增：

```python
return cv2.CAP_DSHOW
```

原有的 `return cv2.CAP_MSMF` 位于其后，因此不会执行。

实际效果：

- Windows 使用 DirectShow；
- 可能用于提高双 USB 摄像头兼容性；
- WSL/Linux 分支不直接受该条件分支影响。

### 5.2 电机总线修改

`motors_bus.py` 的变化：

- 同步读取默认重试次数：`0 → 1`
- 同步写入默认重试次数：`0 → 1`
- 通信失败日志级别：`debug → info`

该修改会在首次通信失败后额外重试一次，并让通信异常更容易在终端中看到。它属于通信稳定性增强，不是显式的 HX 型号驱动。

### 5.3 策略配置修改

`policies.py` 存在 29 行新增、22 行删除，但当前日志只显示了差异开头，尚未完整审查。

```text
状态：待完整检查
```

在查看完整差异前，不对其修改目的作确定判断。

## 6. 厂商补丁备份

已导出核心源码差异：

```text
文件：vendor_packages/hiwonder/vendor_source_changes.patch
行数：187
SHA256：1e578fe57ef0f4f2565818825852bf9f324cc16e9900921bd10d74f051aa57f0
```

该补丁用于保存厂商修改，防止误执行 Git 恢复命令后丢失。

## 7. Git LFS 状态

已成功安装：

```text
git-lfs 3.4.1
```

初次 `git status` 中，大量 `.png`、`.bag` 和 `.safetensors` 测试资源被标记为修改。这可能与 Git LFS、压缩包解压或文件规范化有关。

当前日志没有明确显示以下命令的结果：

```bash
git lfs install
git lfs status
git status --short
```

因此目前只能确认 Git LFS 软件包已安装，不能确认 LFS 初始化后的状态。

需要重新执行：

```bash
cd ~/projects/embodied-agent-arm/vendor/hiwonder/lerobot

git lfs install
git lfs status
git status --short
git lfs ls-files | head -n 30
```

在确认前，不对测试资源执行 `git restore`、`git reset` 或 `git clean`。

## 8. 主项目忽略规则

主项目 `.gitignore` 已增加：

```gitignore
# 厂商原始安装包和源码快照
vendor_packages/
```

验证结果：

```text
vendor_packages/hiwonder/lerobot.zip
```

已被正确忽略。

对应提交：

```text
62b3bba chore: ignore vendor source archives
```

该提交已推送到 `origin/main`。

主项目原有规则也已经忽略：

```gitignore
vendor/
third_party/
```

## 9. 使用原则

禁止在 `vendor/hiwonder/lerobot` 中执行：

```bash
git pull
git reset --hard
git restore .
git checkout .
git clean -fd
```

同时禁止：

- 在 Conda `base` 环境中安装依赖；
- 直接执行 `pip install lerobot`；
- 直接升级到最新上游 LeRobot；
- 在机械臂正常工作时主动烧录舵机固件；
- 对成品机械臂重新设置舵机 ID；
- 未确认端口和接线前运行标定或遥操作。

## 10. 安装前检查清单

- [x] 保存厂商原始源码包
- [x] 计算源码包 SHA256
- [x] 解压厂商源码
- [x] 记录 Git Commit
- [x] 记录 LeRobot 版本
- [x] 确认 Python 版本要求
- [x] 检查 `feetech` 可选依赖
- [x] 检索 HX 与 Hiwonder 字符串
- [x] 导出核心源码差异补丁
- [x] 计算补丁 SHA256
- [x] 忽略 `vendor_packages/`
- [x] 安装 Git LFS
- [ ] 完成 `git lfs install`
- [ ] 检查 `git lfs status`
- [ ] 检查 LFS 文件列表
- [ ] 完整检查 `policies.py` 差异
- [ ] 执行 Python 语法检查
- [ ] 安装厂商源码依赖
- [ ] 验证 LeRobot 命令行入口

## 11. 下一步命令

### 11.1 Git LFS 验证

```bash
cd ~/projects/embodied-agent-arm/vendor/hiwonder/lerobot

git lfs install
git lfs status
git status --short
git lfs ls-files | head -n 30
```

### 11.2 保存完整策略差异

```bash
git diff -- src/lerobot/configs/policies.py   > ../../../vendor_packages/hiwonder/policies_diff.patch

wc -l ../../../vendor_packages/hiwonder/policies_diff.patch
sha256sum ../../../vendor_packages/hiwonder/policies_diff.patch
```

### 11.3 检查补丁与语法

```bash
git diff --check

python -m py_compile   src/lerobot/cameras/utils.py   src/lerobot/configs/policies.py   src/lerobot/motors/motors_bus.py
```

完成以上检查后，再进入厂商依赖安装阶段。
