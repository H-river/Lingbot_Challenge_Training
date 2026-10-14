# LingBot-VLA v2 单卡全量训练冒烟验收

日期：2026-10-14，Asia/Singapore（UTC+8）。最终状态：**训练脚本冒烟通过**。

用户最后明确本轮只检查训练脚本，冒烟 checkpoint 不需要保留。因此没有执行恢复训练；本轮生成的大体积权重在验证后已清理，配置、日志、统计和权重更新证据保留。

## 实测机器与环境

- GPU 实际为 **1×NVIDIA A100-SXM4-80GB**，不是此前计划的 H100。Torch 可用容量 79.251 GiB。
- 16 CPU、47.039 GiB RAM，8 GiB swap；Ubuntu 24.04，驱动 595.84。
- 训练环境 `/root/lingbot_challenge/envs/lingbotvla`；Python 3.12、Torch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.57.3、LeRobot 0.4.2、TorchCodec 0.6.0、NumPy 1.26.4。
- bf16 GPU matmul、FlashAttention 实际运算、LingBot 主模型模块导入通过。
- 固定 LingBot commit `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`，RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`。

## 数据与训练范围

- 允许基座：`models/lingbot-vla-v2-6b`。未下载或加载 robotwin 后训练权重。
- 训练路径包含全部 50 个 clean 任务、2500 条演示，使用已完成转换的 LeRobot 数据。11 个单样本 step 不代表遍历了所有任务或所有演示。
- 采用正式全量 clean 统计 `configs/clean_norm/robotwin.json`。manifest SHA256 `4a3d8889720dbfff26743243831752c8cab7101efd96f5513bfb9ca3126ec2d4`；norm SHA256 `bccf8d00c0bae955802c9d9522383b8f38761e568af6a49852b64184f095a4bc`。
- `mb=1 / GA=1 / gbs=1`，seed=42，11 optimizer steps；VLM、vision 和 action expert 均训练，三路官方对齐和 future image 保留。
- FSDP2 full shard、bf16 mixed precision、gradient checkpointing、compile，官方 Muon 与其 AdamW 子组。
- 不调用仿真，不采集数据，不启动正式训练或比赛提交。没有修改模型、数据加载器、FSDP 包装、优化器源码、数据或 norm。
- 使用未改动的官方 `tasks/vla/train_lingbotvla.py`。外围 benchmark 包装仅记录显存和进程时间，未触碰此前冻结 benchmark 的参数白名单。

## 排障和结果

| 组 | 变化 | 结果 | 已完成 step | allocated / reserved 峰值 GiB |
|---|---|---|---:|---:|
| a10080_mb1_gbs1 | 首次配置 | 启动失败：显式 YAML null 被参数解析成字符串 `null`，误触发 checkpoint 加载 | 0 | 26.644 / 26.701 |
| a10080_mb1_gbs1_r2 | 省略可选 `load_checkpoint_path`，并断言解析后为空 | 第 1 步完成，第 2 步 backward OOM；申请 `(151936,2560)` FP32 张量，约 1.45 GiB | 1 | 76.907 / 77.453 |
| a10080_mb1_gbs1_expandable | 只增加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；复用编译缓存 | **11 步通过，DCP 保存与更新检查通过** | 11 | **77.845 / 78.207** |

可扩展分配段是 PyTorch 提供的实验性分配器选项，本次用它处理上一组 OOM 时的预留碎片，不改训练精度、损失或优化器。[PyTorch 2.8 文档](https://docs.pytorch.org/docs/2.8/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

第 1 步：33.019 s；第 2–11 步官方 StepTime：**5.468 ± 2.161 s**（样本标准差）。第 9–11 步：**2.359 ± 0.025 s**，只有 3 个点。

这些是官方 StepTime，并非独立同步测量的 compute_s。日志存在 `aten.nonzero` graph break；第 9 步前还记录 `self.steps` 导致 `recompile_limit=8`，之后部分区域退回 eager。因此不能把最后三步说成“全部编译完成的长期稳态”，也不能将 gbs=1 的吞吐直接当作正式 gbs=32 或双卡吞吐。

11 步各项 loss 和 GradNorm 都有限且梯度非零。L1_fm 对应 VLA_Loss 范围约 0.195–0.549；不同单样本 loss 有波动，不声称已收敛或性能达标。

成功组训练和 DCP 保存期间，采样到的主机可用内存最低仍约 **18.05 GiB**。说明实际 48G 主机可完成这次测试，不需要把先前的 64 GiB 建议当作硬性最低要求。GPU 显存余量很小，不能保证提高 micro batch 或长期训练仍不 OOM。

三次训练进程（含模型启动、编译和保存）的墙钟合计约 **5.813 分钟**，单 GPU；另有基础检查和 65 秒的 HF 导出。没有超出本轮 1 小时约束。实例实际计费还包括等待和准备时间，不能把该数字当作账单时长。

## 权重更新和保存验证

对 DCP 中的少量张量进行 CPU 读取，比较允许基座的对应片段，未加载完整模型进行 CPU 推理：

| 模块 | 抽样变化元素 / 16384 | 最大绝对变化 |
|---|---:|---:|
| VLM language，第 0 层 MLP down_proj | 16384 | 0.00073363 |
| VLM vision，第 0 层 attn.proj | 16383 | 0.00036945 |
| action expert，第 0 层 shared_expert.down_proj | 16384 | 0.00041940 |

这证明抽查模块真实更新，未沿用冻结 VLM 路径；不是对每个参数逐一验证。

DCP 模型/优化器 metadata 和 rank-0 extra state 完整，global_step=11。DCP 实占 **48.981 GiB**。随后用固定官方 `ckpt_to_state_dict` 与 `save_model_weights` 导出 BF16 HF 权重，3 个分片、1708 个参数键、6,375,907,511 个元素，约 **11.891 GiB**。导出需要 CPU 合并 state dict，属于格式转换，未进行 CPU 模型推理。

用户确认 checkpoint 无需保留后，仅删除本轮自建的 `checkpoints/full_clean50_smoke_20261014T113713Z/checkpoints/global_step_11`，共回收 **60.872 GiB**。删除前保留结构检查报告、DCP metadata、文件大小清单和三类权重变化记录；原始 release、转换中间目录、LeRobot 数据、基座、统计及旧结果均未删除。

恢复训练未执行：用户将本轮验收范围明确限定为训练脚本冒烟。历史报告中“需要恢复验证”的文字不应被当作本轮仍未完成的要求。

## 新服务器部署时必须保留的修复

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

新训练配置使用 `enable_resume: false` 时，**省略** `load_checkpoint_path`；不要显式写 `load_checkpoint_path: null`。运行前用固定官方参数类解析，并断言实际 `load_checkpoint_path` 为空。

当前完整可运行配置见 `a10080_mb1_gbs1_expandable/full_clean50_smoke.yaml`，差异和哈希见同目录 `config_provenance.json`。它是短测配置，学习率调度和 max_steps=11 也只服务于冒烟，不是正式训练预算。

本轮成功后没有继续运行 GPU 训练。模型、正式训练入口、norm 的哈希保持一致，Git tracked diff 为空。Docker 尚未构建；单卡训练通过不等于 Docker、新 H100 或双卡训练已通过。

服务器完整报告目录：`/root/lingbot_challenge/benchmarks/full_finetune_actual_20261014/`。
