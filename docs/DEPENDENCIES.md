# 依赖与复现边界

本项目跨 Windows 与 WSL2，不能用一个 `environment.yml` 准确描述全部运行时。公开仓库采用三层记录：WSL 基础环境、冻结的厂商 LeRobot 源码、Windows 相机桥。

## 已验证版本

| 层 | 组件 | 最终验证版本或身份 |
|---|---|---|
| WSL 基础 | Python | 3.10.18 |
| WSL 基础 | FFmpeg | 7.1.1 |
| WSL 图像 | NumPy | 2.2.6 |
| WSL 图像 | OpenCV (`cv2`) | 5.0.0 |
| 机器人栈 | LeRobot | 0.3.4 |
| 机器人栈 | 厂商源码基线 | commit `882c80d446a63a44868c67ae535467af32ce0e80` |
| 机器人栈 | 厂商源码包 SHA-256 | `b404af16c4a5a58d137704cb9b3dadc2589d841f26f734ab221285628a518cbf` |
| 机器人栈 | 厂商补丁 SHA-256 | `1e578fe57ef0f4f2565818825852bf9f324cc16e9900921bd10d74f051aa57f0` |
| Windows 相机桥 | Python | 3.12.3 |
| Windows 相机桥 | NumPy | 2.5.2 |
| Windows 相机桥 | opencv-python | 5.0.0.93 |
| Windows 相机桥 | cv2-enumerate-cameras | 1.3.3 |
| USB/WSL | usbipd-win | 5.3.0 |
| WSL | kernel | 6.6.87.2-microsoft-standard-WSL2 |

厂商源码在上述 commit 上包含本地补丁。来源、补丁 SHA-256 和修改范围记录在 [`docs/03_vendor_source_record.md`](03_vendor_source_record.md)。厂商源码、原始压缩包和补丁备份因许可证与体积边界不随本仓库发布。

冻结源码声明的关键依赖范围也写入了 `environment.yml`：PyTorch `>=2.2.1,<2.8.0`、torchvision `>=0.21.0,<0.23.0`、torchcodec `>=0.2.1,<0.6.0`、datasets `>=2.19.0,<=3.6.0`、transformers `>=4.50.3,<4.52.0`、safetensors `>=0.4.3` 和 feetech-servo-sdk `>=1.0.0`。

## 没有猜测的版本

保留的公开证据能够证明 CUDA 推理已经运行，但没有保存最终 PyTorch/CUDA wheel 的完整版本字符串。这里不根据日期或 GPU 型号倒推出一个看似精确的版本。

在原 WSL 环境中执行以下命令即可生成新的只读版本清单；脚本不会连接机器人、串口或相机：

```bash
conda activate embodiedarm
python scripts/release/capture_runtime_manifest.py \
  --output evidence/release/runtime_manifest.local.json
```

该清单可能包含主机、GPU 和本地路径信息，公开提交前必须人工检查。Stage 1 的结论不依赖重新生成它。

## WSL 基础环境

```bash
conda env create -f environment.yml
conda activate embodiedarm
python -m pip install --upgrade pip setuptools wheel
```

`environment.yml` 是基础环境，不包含厂商源码和 CUDA wheel。若只想阅读代码或运行公开 CI 的静态核验，不需要机器人、CUDA、数据或 checkpoint。

## 厂商 LeRobot

将已校验的厂商源码恢复到本地忽略目录后：

```bash
git -C vendor/hiwonder/lerobot rev-parse HEAD
# 预期：882c80d446a63a44868c67ae535467af32ce0e80

python -m pip install -e "vendor/hiwonder/lerobot[feetech]"
python scripts/train/inspect_act_environment.py \
  --vendor-root vendor/hiwonder/lerobot
```

不要用最新版 `pip install lerobot` 替换冻结的厂商版本；本项目依赖其 DirectShow、MotorBus 重试和策略配置修改。

## Windows 相机桥

在 Windows PowerShell 中创建独立环境：

```powershell
$bridgePython = "$env:USERPROFILE\venvs\act-v3-camera-bridge\Scripts\python.exe"
py -3.12 -m venv "$env:USERPROFILE\venvs\act-v3-camera-bridge"
& $bridgePython -m pip install --upgrade pip
& $bridgePython -m pip install -r requirements-camera-bridge.txt
```

双相机最终通过 DirectShow 按 VID:PID 绑定：Front `32E6:9221`，Wrist `32E6:9005`。这些是硬件身份，不应替换为不稳定的相机索引。

## 公开 CI 的边界

`.github/workflows/public-closeout-ci.yml` 只执行：

- Python 语法编译；
- shell 语法检查；
- 使用公开语义校准 fixture 运行 11/11 的纯 guard 算法自审；
- 公开证据 JSON/CSV/PNG 校验；
- README 与收尾口径检查。

CI 不模拟串口、相机、CUDA 或真实运动，也不把“静态检查通过”解释为硬件安全认证。
