# 蚂蚁灵波具身大模型挑战赛 · 初赛规则与工作守则

> 来源：天池赛题页 532514、官方帖《初赛仿真必做任务赛题规则解读》（2026-10-04）、官方 FAQ。
> 最后更新：2026-10-12。算力方案改为**智星云 AI Galaxy 云主机**：AMD 算力要排队到 11 月，赶不上初赛。
> 规则有更新时，以天池页面 / 主办方书面答复为准，并同步修改本文件。

---

## 0. 给 Claude Code 的工作守则（每次动手前先过一遍）

**合规**
1. 任何改动数据管线、训练、模型结构、推理、评测的方案，**先过第 3 节「合规三问」和第 4 节禁止清单**。不合规的不写；有疑问就停下来问用户，不要自己判定"应该没问题"。
2. 训练 dataloader **只能读 clean 数据**。改 dataset 或 config 时，主动检查有没有 randomized 路径、unseen_instructions、外部数据混进训练。官方 `robotwin.yaml` 配方可能混有 randomized 数据，复现时必须改掉。
3. 训练过程**不能调用仿真环境**，包括 reset、step、render、`check_success()`。只有评测脚本可以调用。
4. 推理**只能加载一份权重**，不能按 task id 或 instruction 做路由。
5. 遇到规则没覆盖的灰色地带，在回复里明确标出「⚠️ 规则灰区，建议向主办方确认」。

**复现**
6. 每加一个技巧，**同时更新 README / config 里的披露说明**（第 6 节）。评委会用我们的 `train.sh`、`eval.sh` 从训练到推理完整复现。
7. `train.sh` 必须能从基座权重一路复现到提交的 checkpoint。不能依赖手动步骤、本地临时文件或没记录的超参。随机种子、数据版本、依赖版本全部固定。
8. **全队一共只有 3 次正式提交机会。** 不要建议"先交一版看看"。提交前必须先跑完全量评测。

**算力（智星云，第 8 节）**
9. **GPU 按小时计费。** 任务跑完、或者需要长时间等待时，提醒用户释放实例。写长任务脚本时，在最后一步自动保存产物并提示释放。
10. 所有机器使用同一套环境：**Ubuntu 24.04 + CUDA 13.0 镜像，torch 2.8.0+cu128**。编译 CUDA 扩展时**禁止使用系统自带的 nvcc 13.0**，要用 conda 装的 cuda-toolkit 12.8。
11. 机房在国内：
    - 下载 HuggingFace 资源先设置 `HF_ENDPOINT=https://hf-mirror.com`。
    - 模型权重优先从 ModelScope 下载。
    - pip 和 conda 使用国内镜像源。
12. 训练和全量评测前，先做小规模冒烟测试：1 个任务、少量 episode，并记录单个 episode 的耗时，再决定开几台机器。
13. 初赛提交用的 checkpoint 和 `results.json` **必须出自同一套环境**，不能训练在一个平台、评测在另一个平台。

---

## 1. 赛制

| 项 | 内容 |
|---|---|
| 主办 | 蚂蚁灵波 × 魔搭社区 × 阿里云天池（AMD 为算力合作方） |
| 阶段 | 线上初赛 → 线下决赛（11/13–15，上海，真机黑客松，真机由主办方提供）。没有复赛 |
| 初赛截止 | **2026-10-26 24:00**。官方建议提前 2–3 天提交 |
| 提交次数 | **初赛累计 3 次**（不是每天 3 次） |
| 组队 | 每队 2–5 人；单人报名后必须组队 |
| 初赛内容 | 仿真任务必做；真机 Demo 选做，不设固定分值，作为筛选参考 |
| 选拔逻辑 | 按仿真成功率从高到低排序，依次复现核验，再结合真机 Demo 选出入围队伍。**入围队伍会从训练到推理完整复现** |
| 成绩认定 | **以官方复测为准** |
| 未公开 | clean 和 random 两个 setting、以及各任务之间怎么加权；晋级名额；奖金 |

## 2. 仿真任务定义

- **训练起点**：`lingbot-VLA-v2-6b`
  - [HF](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main) / [ModelScope](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b)
- **训练数据**：RoboTwin 2.0 Aloha-AgileX 50 个任务的 clean 数据，每任务 50 条，共 2500 条。[数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)。LeRobot 格式或原始格式都可以。**只下载 clean 子集。**
- **训练方式**：cotrain，50 个任务混合训练同一份权重。
- **评测设定**：每任务、每 setting 各测 100 次，合计 10,000 个 episode。
  - `clean2clean`：clean 环境 + **seen** 指令
  - `clean2random`：randomized 环境 + **unseen** 指令
- **成功判定**：只看 RoboTwin 的 `check_success()` 返回值，没有人工复核。
- **官方复测**：使用我们 `eval.sh` 里的推理参数。初赛不限制单次推理时间和显存。
- **测试次数**：没测满 100 次也可以提交，但最终以官方复测为准。
- **专家数据采集配额**：每个场景最多发起 5 次，换 seed 或同 seed 重试都计入。用完后该场景不能再采集，但 policy 测试照常完成，如实记录结果。

## 3. 合规判定：三问

1. 训练全程是否**零环境调用**？
2. reward、标签、样本权重是否**全部能从 release 数据内部获得或推导出来**？
3. 教师模型或训练起点是否**涉及官方 robotwin 后训练权重**？必须不涉及。

三问都通过才合规。判定看监督信号的来源，不看方法叫什么名字。

**release 资产**指 clean 演示数据、任务定义和 `seen_instructions` 指令池。这是唯一合规的数据来源。

## 4. 允许 / 禁止清单

### ✅ 允许

**数据处理**
- 训练期视觉增广（亮度、颜色噪声等）、机器人 state 扰动
- 训练机制内部的噪声（如 diffusion 加噪），前提是不额外生成 demonstration
- 用 jerk / Z-score 过滤低质量轨迹；按数据自身的统计量（时长、jerk、平滑度、能量）给样本加权或设过滤阈值
- seen 池内指令替换：同一条轨迹的指令，换成本任务 `seen_instructions` 里的其它指令（不新增 (指令, 轨迹) 配对）

**模型结构**
- 可以大幅改结构，唯一要求是主干从 `lingbot-VLA-v2-6b` 初始化
- 可以引入公开的预训练模块（vision / depth / seg encoder 等），需在 README 注明来源
- LoRA 训练后合并回全量权重再上传，需在 config / README 中说明

**训练策略**
- 优化器、学习率、归一化统计、动作表示、损失函数、训练期 EMA
- 基座自蒸馏；以我们自己训练的权重做教师
- 官方配方自带的双教师蒸馏（LingBot-Depth / DINO-Video，即 `align_params`）
- offline RL（IQL / CQL / AWAC）、advantage-weighted BC，前提是零环境调用、监督信号全部来自 release 数据

**推理**（前提：单一权重、不按任务路由、不与环境交互）
- 调整 action chunk 的执行长度；在 rollout 中途重新推理（replanning）
- action smoothing、temporal ensemble、基于模型自身输出的置信度门控
- 切换 bf16 / fp32、开关编译
- 以上都要在 config / README 中如实披露
- ⚠️ 官方 FAQ 第 9 条只写了"只能用一份权重、不能按题目信息做确定性路由"，上面逐项允许的说法来自规则解读帖。如果要靠其中某项拿分，建议在钉钉群再确认一次

### ❌ 禁止

**数据**
- randomized 数据以任何比例、任何形式参与训练
- 外部数据（真机自采、第三方数据集）
- 自建 randomization（改光照、桌面高度后重新渲染 observation）
- 用 simulator 额外 rollout 的数据训练，即使标成 clean 也不行
- 需要跑环境才能得到的 reward / 价值标注；用环境打分筛选或加权数据
- unseen_instructions 以任何形式进入训练（直接使用、改写、用于筛选或加权）
- 用 LLM 生成 release 里没有的新指令并绑定到轨迹
- HER 伪目标重标注

**起点 / 教师**
- 以 `lingbot-vla-v2-6b-robotwin`（含 randomized 数据）作为训练起点、蒸馏教师、reward 模型或判据模型。这个权重**只允许用来验证评测管线**，不能进入训练，也不能出现在提交内容里

**学习方法**
- 在线 RL，或任何训练过程中与仿真环境交互采样的方法

**推理**
- 按 task id 路由的 LoRA / adapter / head / checkpoint；基于题目信息的确定性路由
- 多 checkpoint / 多模型 ensemble；多 seed 投票；test-time adaptation
- 推理时加载多于一份权重
- 检测到失败后重置 episode 重新执行（主办方没有书面放行前，一律视为禁止）

## 5. 提交材料（打包成 Zip，通过天池入口上传）

```text
团队名称_初赛提交材料/
├── 01_评测结果/results.json        # 两个 setting × 50 任务的测试次数和成功次数，按官方模板填写
├── 02_代码材料/
│   ├── README.md
│   ├── train.sh
│   ├── eval.sh                     # 官方用这里的推理参数复测
│   ├── configs/
│   └── src/
├── 03_模型权重/checkpoint/          # 必须和 results.json 对应
├── 04_复现与调优说明/reproduction_report.md
├── 05_真机创新可选任务/              # 可选：demo_video.mp4 + demo_report.md
└── 06_小红书创作活动/xiaohongshu_info.md
```

模板见天池赛题页「数据」部分的【必读_模板和目录要求】。真机 Demo 禁止人工遥控、剪辑伪造和硬编码。

## 6. 必须披露的内容（写进 README / config / reproduction_report）

- 数据增广 V1/V2：颜色/几何、教师仅几何、样本 RNG、seen 逐轨迹校验、可选 state 噪声。
- V2 seen 背景纹理：用户明确授权本次实现的规则灰区，待主办方确认；不是对第 3 节的普遍放宽。
- 严禁读取 unseen 纹理；提供不读素材的 procedural 备用；V1/V2/V2b 干扰物只来自 release clean 首帧。
- V4 / V5 干扰物为程序生成的几何体（`distractor.source: procedural`，不读任何素材），用途是打破"面前有东西就抓"的 shortcut；
  V4 为纯 2D 形状 + 颜色（无明暗、无阴影），V5 带明暗、高光与接触阴影。贴合成物体未在允许清单中逐项列出，⚠️ 规则灰区，待主办方确认。
- 披露 mask/机械臂过滤的启发式局限、实际吞吐量、8k provenance 缺口及未完成验证（AUG_REPORT.md）。

- 训练方式、参数配置、模型改动、调优思路、完整的复现步骤
- LoRA 等方法合并回基座后的权重形态
- 引入的公开预训练模块：来源和使用方式
- 所有推理策略的实现细节（chunk 长度、replanning、smoothing、ensemble、门控等）
- **评测环境**：
  - 平台：智星云云主机
  - GPU 型号和数量
  - 系统：Ubuntu 24.04
  - 驱动：580（CUDA 13.0）
  - PyTorch：2.8.0+cu128
  - 运动规划器：cuRobo
  - 有 Dockerfile 的话一并提交

## 7. 资源

- 大赛页：https://tianchi.aliyun.com/competition/entrance/532514
- 规则解读帖：https://tianchi.aliyun.com/forum/post/1081761
- LingBot-VLA 代码：https://github.com/Robbyant/lingbot-vla
  - 主页写的要求是 Python 3.12 + PyTorch 2.8.0 + CUDA 12.8
  - 主页展示的可能是 1.0 版本，2.0 的要求以对应分支 / README 为准
- 答疑渠道：大赛钉钉群、小红书、灵波开发者微信群
- AMD Radeon Cloud：排队到 11 月，初赛不使用。参考：https://github.com/AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09

---

## 8. 算力：智星云 AI Galaxy

平台：https://gpu.ai-galaxy.com
- 类型选**「云主机」（KVM 完整虚拟机）**，可以自己装驱动、Vulkan 和 Docker
- 平台提供开放 API，可以用脚本开机、跑任务、释放
- 公司在上海，机房在国内

### 8.1 机型选择

以下是 2026-10-12 的价格，单卡、按小时计，库存经常变动：

| 用途 | 机型 | 约 ¥/时 | 说明 |
|---|---|---|---|
| 全参训练 | **8×A800-SXM4-40G** | ~18.5（2.31/卡） | 用 FSDP 分摊参数、梯度和 Adam 状态（约 100GB）。性价比最高 |
| 全参训练（备选） | 4×A100-80G | ~32（8.00/卡） | A800 没货时再考虑 |
| LoRA / 小实验 | 1–2×A800-40G 或 4090PLUS-48G | 2.3–5 | |
| 评测 | **RTX 4090PLUS-48G**，多台并行 | 2.53/台 | 有图形渲染能力，SAPIEN 跑得稳；一张卡能同时跑 2–3 个评测进程 |
| 评测（备选） | A100 / A800 | | 前提是 `vulkaninfo` 能识别到这张卡 |
| 下载数据 / 预处理 | CPU 云主机 | 0.18 | 前提是数据盘能挂到 GPU 机器上 |

**不要用：**
- V100：不支持 bf16
- RTX 5090：Blackwell 架构，cuRobo / SAPIEN 兼容性有风险
- 1080Ti / 2080Ti / 3080：显存不够
- 本地笔记本 4080（12GB）：放不下 6B 模型推理，只用来写代码和小规模调试

**CPU 和内存选最高档**（16 核 / 48G 或以上）。仿真和视频解码都吃 CPU。

**磁盘：**
- 训练机 ≥500G–1T。带优化器状态的完整 checkpoint 约 80–100GB
- 只保留必要的 checkpoint，中间版本只存权重

### 8.2 统一环境

所有机器使用镜像 **`ubuntu24.04_cuda13.0`**（580 驱动，A100 也支持）。

| conda 环境 | Python | 主要依赖 | 用途 |
|---|---|---|---|
| `lingbot` | 3.12 | torch 2.8.0+cu128，官方 `install.sh` | 训练 / 推理 / policy server |
| `robotwin` | 3.10 | torch（cu128）、`cuda-toolkit=12.8`（conda 安装） | 仿真、cuRobo、评测 client |

两个环境之间通过 RoboTwin 的 policy server / client 模式通信。

**编译 cuRobo 等 CUDA 扩展：**

```bash
conda activate robotwin
conda install -c nvidia cuda-toolkit=12.8
export CUDA_HOME=$CONDA_PREFIX
nvcc --version        # 必须显示 12.8，不能是系统的 13.0
```

### 8.3 开机自检（每台新机器都要做）

```bash
nvidia-smi                                   # 驱动 580，CUDA Version 13.0
sudo apt install -y libvulkan1 vulkan-tools
vulkaninfo --summary                         # 必须能看到 NVIDIA 卡，不能只有 llvmpipe
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  # 2.8.0 12.8 True
```

`vulkaninfo` 失败时，通常是驱动缺图形库，补装 `libnvidia-gl-<驱动版本>`。

### 8.4 国内网络

```bash
export HF_ENDPOINT=https://hf-mirror.com     # HuggingFace 数据集
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

- 模型权重优先从 ModelScope 下载：`Robbyant/lingbot-vla-v2-6b`
- GitHub 访问可能很慢。代码和 Docker 镜像要提前准备好，镜像推到阿里云 ACR，或者用 `docker save` 打包后拷过去

### 8.5 Docker（推荐，同时作为复现环境提交）

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9"      # A100/A800 = 8.0，4090 = 8.9（build 时机器上没有 GPU，要显式指定）
ENV NVIDIA_DRIVER_CAPABILITIES=all      # 必须包含 graphics，否则容器里 Vulkan 不可用
RUN apt-get update && apt-get install -y libvulkan1 vulkan-tools ...
```

运行时加 `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`。

### 8.6 计费与数据安全

- 按小时计费，任务跑完立即释放实例
- 释放前确认 checkpoint、日志、`results.json` 已经保存到持久化位置
- 先确认关机后磁盘是否继续收费、卡会不会被别人占走
- 长时间连续使用时，再考虑包天 / 包月的折扣

### 8.7 推荐工作流

1. **本地**：写 repo、Dockerfile、dataloader；用 1–2 个任务的数据配合 mock policy 把流程跑通。
2. **冒烟测试**（4090PLUS，约 1 小时）：
   - 配好环境，跑完开机自检
   - 用专家策略跑通 1 个任务
   - 测出单个 episode 的耗时
3. **验证评测管线**：拿官方 robotwin 权重评测几个任务，结果和公开成绩对得上，就说明管线没问题。这一步只做验证，不能进入训练或提交内容。
4. **训练 baseline**（8×A800）：从 `lingbot-VLA-v2-6b` 出发，只用 clean 数据。
5. **全量评测**：多台 4090PLUS 并行，跑完 10,000 个 episode，得到第一个真实分数。
6. **调优迭代**：先在子集上快速比较，最后用全量评测确认，再决定是否提交（总共只有 3 次）。