# 环境配置说明

## 1. 开发平台

- 宿主系统：Windows 11
- Linux 环境：WSL2
- Linux 发行版：Ubuntu 24.04 LTS
- 项目路径：`~/projects/embodied-agent-arm`
- Conda 环境名称：`embodiedarm`

## 2. 环境策略

本项目当前优先保证幻尔 SO-ARM101 兼容版机械臂及其厂商 LeRobot 源码能够稳定运行。

因此第一阶段采用厂商兼容环境：

- Python 3.10.18
- FFmpeg 7.1.1
- `evdev`
- `pytest`
- `ruff`
- `mypy`

在机械臂标定、主从遥操作、数据录制、ACT 训练和真实推理全部跑通之后，再单独评估迁移到上游最新 LeRobot 的可行性。

## 3. 创建 Conda 环境

项目根目录中包含 `environment.yml`。

执行：

```bash
cd ~/projects/embodied-agent-arm
conda env create -f environment.yml
conda activate embodiedarm
python -m pip install --upgrade pip setuptools wheel
```

## 4. 验证环境

执行：

```bash
echo "$CONDA_DEFAULT_ENV"
python --version
ffmpeg -version | head -n 1
python -c "import evdev; print(evdev.__file__)"
pytest --version
ruff --version
mypy --version
```

预期核心结果：

```text
Conda 环境：embodiedarm
Python：3.10.18
FFmpeg：7.1.1
evdev：能够正常导入
pytest、ruff、mypy：能够正常输出版本
```

## 5. 当前验证结果

截至 2026-07-12，已经确认：

- WSL2 正常运行
- Ubuntu 24.04 LTS 正常运行
- 项目位于 Linux 文件系统
- Conda 环境 `embodiedarm` 能够正常激活
- Python 3.10.18 正常
- FFmpeg 7.1.1 正常
- `evdev` 能够正常导入
- `pytest`、`ruff`、`mypy` 可用
- GitHub 仓库已创建并推送

## 6. 依赖安装原则

禁止在 Conda `base` 环境中安装 LeRobot 或厂商依赖。

安装厂商源码前，必须完成：

1. 备份厂商原始源码包
2. 检查 `pyproject.toml`
3. 检查 `setup.py`
4. 检查 `requirements.txt`
5. 检查安装脚本
6. 记录源码版本或 Git Commit
7. 确认硬件驱动目录
8. 确认厂商修改的 LeRobot 接口
9. 在 `embodiedarm` 环境内安装
10. 安装完成后导出依赖快照

## 7. 厂商源码目录策略

厂商源码不直接混入主项目业务代码。

建议目录：

```text
embodied-agent-arm/
├── src/
├── docs/
├── configs/
├── scripts/
├── tests/
├── vendor/
│   └── hiwonder_lerobot/
└── third_party/
```

其中：

- `vendor/` 用于本地保存厂商源码
- `third_party/` 用于保存外部依赖或适配代码
- 默认不将大体积厂商源码直接提交到主仓库
- 应记录源码来源、版本和安装方式

## 8. 依赖快照

厂商源码安装完成后执行：

```bash
python --version > docs/python_version.txt
python -m pip freeze > requirements.lock.txt
conda env export --from-history > environment.history.yml
```

建议同时记录：

```bash
git rev-parse HEAD
```

若厂商源码不是 Git 仓库，则记录：

- 源码包文件名
- 下载日期
- 文件校验值
- 厂商文档版本
- 安装步骤
- 修改过的文件

## 9. WSL2 USB 设备策略

机械臂通过 USB 总线驱动模块与电脑通信。

Windows 端需要使用 `usbipd-win` 将 USB 设备挂载到 WSL2。

Windows PowerShell 中：

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

WSL2 中检查：

```bash
lsusb
ls /dev/ttyUSB* 2>/dev/null
ls /dev/ttyACM* 2>/dev/null
```

预期能够识别 Leader 与 Follower 对应的两个串口设备。

## 10. 摄像头策略

项目包含两个 USB 摄像头：

- 第一视角腕部摄像头
- 第三视角固定摄像头

WSL2 下需要检查：

```bash
ls /dev/video* 2>/dev/null
v4l2-ctl --list-devices
```

若 WSL2 下摄像头不稳定，则采用分层方案：

```text
Windows：机械臂控制与数据采集
WSL2：代码开发、模型训练、Agent 服务
```

不优先使用虚拟机，因为虚拟机同样需要处理 USB、摄像头和 GPU 透传，并会增加系统复杂度。

## 11. 上游 LeRobot 与厂商版本隔离

本项目暂时不直接执行：

```bash
pip install lerobot
```

也不在厂商版本尚未确认时直接安装最新 LeRobot。

建议后续建立独立上游环境，例如：

```text
Conda 环境：lerobot-upstream
用途：阅读、兼容性测试、上游迁移验证
```

厂商环境和上游环境必须隔离，避免依赖冲突。

## 12. Git 提交要求

环境相关改动提交前执行：

```bash
git status
git diff
```

提交示例：

```bash
git add environment.yml docs/01_environment_setup.md
git commit -m "chore: update Chinese environment setup documentation"
git push
```

## 13. 常见问题

### 13.1 Conda 环境未激活

检查：

```bash
echo "$CONDA_DEFAULT_ENV"
```

应输出：

```text
embodiedarm
```

### 13.2 Python 版本不正确

检查：

```bash
which python
python --version
```

Python 路径应位于：

```text
~/miniforge3/envs/embodiedarm/
```

### 13.3 FFmpeg 不可用

检查：

```bash
which ffmpeg
ffmpeg -version | head -n 1
```

### 13.4 串口没有权限

后续根据实际设备情况处理：

```bash
sudo usermod -aG dialout "$USER"
```

执行后需要重新登录 WSL。

### 13.5 USB 设备无法进入 WSL

优先检查：

- WSL 是否为 WSL2
- `usbipd-win` 是否安装
- BUSID 是否正确
- 设备是否已经绑定
- 设备是否已经附加到 WSL
- Windows 是否仍占用设备
