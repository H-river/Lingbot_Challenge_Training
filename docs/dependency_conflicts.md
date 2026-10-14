# 官方配方的声明冲突

固定源码：LingBot `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`。

官方 `tools/create_train_env.sh` 明确使用 `--no-deps` 安装 LeRobot v0.4.2、numpydantic 1.9.0 和深度子模块，并在 `pip check` 失败时只警告。以下冲突是源码/PyPI 元数据实际核实结果，不能同时满足：

| 组件声明 | LingBot 官方固定值 |
|---|---|
| LeRobot Torch >=2.2.1,<2.8.0 | Torch 2.8.0 |
| LeRobot torchvision >=0.21,<0.23 | torchvision 0.23.0 |
| LeRobot torchcodec <0.6 | torchcodec 0.6.0 |
| LeRobot datasets >=4.0,<4.2 | datasets 3.6.0 |
| LeRobot accelerate >=1.10 | accelerate 1.7.0 |
| numpydantic 1.9.0 NumPy >=2 | NumPy 1.26.4 |
| vendored mdm Torch ==2.6.0 | Torch 2.8.0 |
| vendored mdm xformers ==0.0.29.post2 | 不在 LingBot 官方核心安装列表中 |

不修改包的依赖元数据来制造 `pip check` 全绿。用户已明确选择保留官方 --no-deps 运行版本，记录冲突并实测。用 `configs/accept_upstream_metadata_conflicts` 记录决定。此决定不把缺失的必要模块视为成功：实际导入和数据转换仍须运行，未知错误须继续排障。

另发现官方 utils3d `.pth` 指向当前仓库不存在的 `morgbd_clean/3rd/utils3d`；最小修复采用 vendored MoGe pyproject 声明的官方 utils3d commit `3fab839f0be9931dac7c8488eb0e1600c236e183`，不修改模型代码。

## 安装后的实际 pip check

原始逐行输出保存在 `results/lingbot-pip-check-raw.txt`，已审阅的精确条目保存在 `configs/known_lingbot_pip_check.txt`。验收只将完全匹配这些条目的结果标为 `KNOWN_METADATA_CONFLICT`，不是 PASS，且返回待完成状态；新增未知条目仍为 FAIL。

还实测发现官方固定 `plyfile 1.1.4` 要求 NumPy>=2、MLflow 2.22.5 要求 PyArrow<20、mlflow-skinny 要求 packaging<25，与官方核心 NumPy1.26.4/PyArrow21/packaging25 冲突。保留核心固定值。

MoGe/MDM 声明 opencv-python，而环境使用官方 opencv-python-headless，cv2 已实测导入。MoGe 独立 Gradio 演示界面、MoGe 自身训练/测试 pipeline、LeRobot 的其他策略/遥操作/可视化依赖（diffusers、pynput、pyserial、rerun-sdk）未装；当前比赛的 LingBot 策略和离线数据转换不使用这些入口。系统 CMake 已安装，pip 元数据不识别系统包。xformers 0.0.29.post2 与官方 Torch2.8 栈不匹配，未强装；vendored 深度模型保留官方自身 fallback，未改模型代码。

MLflow 的未固定传递依赖 databricks-sdk 最新版引入了与核心 protobuf3.20.3 冲突的声明，已在允许范围内固定为0.57.0，其官方 PyPI 元数据没有该 protobuf 限制；这是普通依赖兼容修复，不改训练逻辑。

单条 clean episode 已用 LeRobot0.4.2 + datasets3.6 完成转换，并回读第0及138帧：三路视频均为3×480×640，state/action均14维且数值有限。flash-attn、TorchCodec、LingBot数据模块、MoGe与MDM模型模块导入已通过；LingBot主模型模块因导入时检查CUDA而报“No CUDA GPUs are available”，保留GPU_PENDING。
