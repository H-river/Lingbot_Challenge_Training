# LingBot-VLA v2 全量微调：挂卡前检查与冒烟测试

检查日期：2026-10-14。项目根目录：`/root/lingbot_challenge`。
服务器交付目录：`/root/lingbot_challenge/benchmarks/full_finetune_preflight_20261014/`。

当前已完成 CPU 验收，**尚未运行 H100 训练、checkpoint 保存或恢复**。Docker 封装在全量 GPU 验收后进行，本次未安装或修改任何系统服务、驱动。

## 数据和环境实况

- 50 个 Aloha-AgileX clean 任务、2500 条演示已全部转换，原转换流水线于北京时间 2026-10-14 01:49 完成。重新审计通过，LeRobot 元数据合计 549787 帧。
- 来源仅 `TianxingChen/RoboTwin2.0`，revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`；manifest SHA256 `4a3d8889720dbfff26743243831752c8cab7101efd96f5513bfb9ca3126ec2d4`。
- 正式全量 clean 统计：`configs/clean_norm/robotwin.json`，SHA256 `bccf8d00c0bae955802c9d9522383b8f38761e568af6a49852b64184f095a4bc`。训练使用此文件，旧的 `baseline_clean_norm` 保留但不作为这次训练统计。
- 实际训练环境：`envs/lingbotvla`，Python 3.12、Torch 2.8.0+cu128、Transformers 4.57.3、LeRobot 0.4.2、FlashAttention 2.8.3、TorchCodec 0.6.0、NumPy 1.26.4。
- 源码固定 LingBot `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`、RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`。本次未修改模型、正式训练入口、加载器、FSDP 包装、原数据或统计；没有新增 tracked diff。
- 单条转换后的 clean 数据实际读取：state/action 各 14 维，三路图像各 `3×480×640`，数值有限。50 份原转换验证报告已验证全部 state/action 与输入对应关系，视频范围是每任务首尾帧三路解码，**不是所有视频帧完整解码**。
- Torch CPU 运算、相关训练数据/教师模块/FlashAttention 导入、本地 tokenizer 和资产清单检查通过，没有将 6B 模型加载至 CPU。
- `pip check` 仍有官方固定版本已记录的声明冲突，状态是 `KNOWN_METADATA_CONFLICT`，不声称全绿；本次没有新增未知条目。详细条目见 `preflight_cpu.json` 及项目 `docs/dependency_conflicts.md`。

## 挂卡和存储要求

用户已确认计划一起升级：**2×H100 80GB、主机内存至少 64 GiB、checkpoint 盘至少 180 GiB 可用空间**。计划升级不代表已升级；运行前重新检查。

180 GiB 是两份完整 DCP（模型、优化器、额外状态）及 HF 导出、临时空间的保守容量保护，实际占用须在 GPU 验收后记录。当前系统盘约 67 GiB 可用、数据盘约 52.8 GiB 可用，不满足保护条件。建议数据盘再扩容约 200 GiB。现有 HDF5/JPEG 转换中间文件约 55 GiB，本次未删除。

默认 checkpoint 根目录：`/mnt/lingbot-data/lingbot_challenge/checkpoints`，位于已确认使用的数据盘。若新盘挂载在其他位置，先确认持久化与容量，再在新交付目录重新执行 `prepare_config.py --checkpoint-root /确认的目录`；不要对未知盘自动格式化，也不要覆盖已有 config 或运行结果。

挂卡后先检查平台驱动、GPU 可见性、NVLink 拓扑和 NCCL 实测带宽。SXM 型号不等于 NVLink 已可用；之前 A100 实例曾实际走 SHM。现有仿真扩展曾按 A100 架构构建，**不能据此声称 H100 仿真可用**；当前训练不依赖 cuRobo/PyTorch3D 的仿真执行。

## 已准备的全量配置

`full_clean50_smoke.yaml` 从固定官方 `configs/vla/robotwin/robotwin.yaml` 生成，完整差异与哈希在 `config_provenance.json`。参数类直接从固定源码原样提取到内存进行 CPU 解析，未导入 GPU 训练入口。

- 只加载允许的 `models/lingbot-vla-v2-6b`，不加载 robotwin 后训练权重。
- `train_expert_only: false`、`freeze_vision_encoder: false`、`freeze_vit: false`，保留 current-depth、future-depth、future-video 对齐和 future image。
- FSDP2 full shard、官方可配置 gradient checkpointing、mixed precision/bf16、compile；不修改模型结构或新增训练方法。
- 双卡 `mb=1 / GA=16 / gbs=32`，seed=42；11 optimizer steps，在 step 11 保存 DCP 和 HF 权重。
- 只使用 `configs/train_clean.txt` 的 50 个 clean 任务及全量 clean 统计；评测 disabled、command=null，不调用任何仿真环境。
- 这是硬件冒烟参数，**不是已调优的正式训练 batch、学习率或训练预算建议**。

## 可复制命令

先检查 CPU 条件；可以现在运行：

```bash
export LINGBOT_ROOT=/root/lingbot_challenge
PREP="$LINGBOT_ROOT/benchmarks/full_finetune_preflight_20261014"
bash "$PREP/check_cpu.sh"
```

退出码 1 表示实际失败；退出码 3 表示 CPU 输入已检查，但存在 GPU/STORAGE_PENDING 或已记录 metadata 冲突，不能当作 GPU 验收通过。日志保存在项目 `logs/`，明细在 `preflight_cpu.json`。

**以下命令本次未执行，须等 GPU、RAM 和磁盘升级到位，并通知用户开始冒烟测试后才运行：**

```bash
export LINGBOT_ROOT=/root/lingbot_challenge
PREP="$LINGBOT_ROOT/benchmarks/full_finetune_preflight_20261014"
bash "$PREP/run_smoke.sh" initial
bash "$PREP/run_smoke.sh" resume
```

第一次运行包含 GPU bf16/FlashAttention 运算检查、两 rank NCCL all-gather/reduce-scatter 带宽测试和 11 步全量训练。第二次从明确的 step-11 checkpoint 恢复，仅新增 step 12 并保存，不能通过隐式搜索随便恢复旧结果。

两段训练分别设 20 分钟和 8 分钟上限，带宽测试 90 秒上限。双卡进程时间约束意图是不超过约 60 GPU-min；CPU 导出检查不占 GPU。超时即失败，不自动反复重试或继续正式训练。到时仍需人工记录总计费时长。

正式入口保持原样：`tasks/vla/train_lingbotvla.py`。`train_with_memory_record.py` 仅通过 runpy 执行该入口，并记录各 rank 的峰值 allocated/reserved 和包含启动/保存的进程总时间，不改冻结逻辑或计算语义。日志 StepTime 按官方口径报告，不能冒称单独同步测量的 compute_s。

## 必须通过的 GPU 验收

1. 两张 H100 的 bf16 运算、FlashAttention 和实际训练成功；记录拓扑、NCCL transport、实测带宽与每 rank 显存。
2. step 1–11 的各项 loss 和 GradNorm 有限，梯度范数非零；单列 step 2–11 的官方 StepTime 均值/标准差，编译抖动另外解释。
3. DCP 模型/优化器 `.metadata`、每 rank extra state、HF 全部分片齐全，参数元素数大于 6B。
4. 从允许基座与保存后的 HF 权重各读取少量张量切片，比较 VLM language、VLM vision、action expert 的更新。先按导出 dtype 转换基座样本以排除精度转换假象；不是加载整个 6B 模型，也不声称所有参数逐个验证。
5. 恢复日志明确包含指定 checkpoint 加载成功，不能出现从头训练/加载失败；训练日志仅新增 step 12。原入口会捕获加载异常并可能从头开始，因此退出码零不足以证明恢复通过。不宣称 CUDA RNG/bitwise 完全一致。

真实 GPU 结果只在实际执行后生成。当前目录的 CPU 报告或 helper 静态检查均不能替代它们。

通过后确认报告、日志、检查点和版本清单已保存，再提醒用户释放 GPU 实例。Docker 封装固定依赖和源码，模型、clean 数据、统计、checkpoint 外部挂载；宿主机驱动/容器运行时由平台提供。新服务器要单独做 GPU/保存恢复验收。

## 新机器安装资料

现有可复制安装说明和固定版本包：本地 `LingBot_VLA_v2_Server_Setup.md`、`lingbot_setup_handoff_20261013.tar.gz`，服务器 `docs/setup_handoff_20261013/`。该包反映 10 月 13 日交付时的历史状态；**“只转换了 2 个任务”的历史快照已被本次 50 任务审计更新**。

迁移应同步已验收的 `data/clean_processed/lerobot/`、clean manifest、`train_clean.txt`、`configs/robot_configs/` 和 `configs/clean_norm/`；原 train list 和 YAML 有绝对路径，必须在新机按项目路径重新生成并审计，不能直接带着旧路径启动。原始 ZIP/HDF5 与 55 GiB 转换中间文件并非训练入口必读资产，可按存储计划另外保留；本次没有删除任何旧文件。
