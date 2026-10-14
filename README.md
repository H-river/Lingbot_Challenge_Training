# LingBot Challenge Training

固定源码、clean 数据准备、训练入口及真实冒烟记录。遵守 [COMPETITION_RULES.md](COMPETITION_RULES.md) 第 3、4 节：仅 release clean/seen 数据、允许的原始 `lingbot-vla-v2-6b` 基座，训练不调用仿真。这份规则来自用户，不代表已独立核验全部外部来源。

## 当前状态（2026-10-14）

- 远端 50 个 clean 任务、2500 条演示、549787 帧已转换；本次现场只读审计通过。仓库包含清单、统计及验证报告，**不含数据或权重**。视频验证仅为每任务数据集首尾帧的三路解码，不是所有帧完整解码。
- 固定上游 `Robbyant/lingbot-vla-v2` commit `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`，完整 tracked 源码快照在 `repos/lingbot-vla-v2/`，原许可证保留。源码哈希见 `docs/upstream_source_sha256.json`；不是上游 Git 历史的镜像。
- 今天的 A100 80GB 全参冒烟完成 11 optimizer steps；DCP 保存、BF16 HF 导出和三类模块抽样权重更新通过。峰值 allocated 77.845 GiB / reserved 78.207 GiB。冒烟 checkpoint 已按用户要求删除，恢复训练未执行。
- 本次现场检查时 GPU 数量为 0；系统盘剩余约 67 GB，数据盘约 53 GB。这些是检查时快照，启动前必须重新核实。
- 已验收的是 batch=1/max_steps=11 短测，不代表正式训练预算、长期稳定性、双卡/H100、Docker 或比赛成绩通过。依赖仍有已记录的 upstream metadata conflicts。

## 在新机器上部署（Docker）

交给队友部署时，看 [DEPLOY.md](DEPLOY.md)：一个镜像 + 权重包 + 数据包，用 `run.sh check / smoke / train` 三条命令完成。镜像构建：`bash docker/build.sh --save <输出目录>`；8 卡配置：`configs/train_8xA100.yaml`。

## 目录

- `repos/lingbot-vla-v2/`：固定官方训练源码及其许可证。**其中官方默认 RoboTwin 配方不是本项目 clean-only 入口，不要直接用来训练。**
- `scripts/`：安装、下载、clean 转换、统计、审计和启动工具。安装工具兼容本仓库的带哈希源码快照。
- `configs/`：clean manifest、训练列表、机器人配置、正式 `clean_norm`；`baseline_clean_norm` 仅是历史 baseline 统计。
- `benchmarks/full_finetune_actual_20261014/`：成功单卡短测原始配置、脚本、日志和证据；保留原服务器绝对路径及历史时间。
- `docs/full_finetune_smoke_actual_20261014.md`：最新已执行结果；早期 preflight 和部署交付文档属于历史记录，冲突时以最新实测报告为准。
- `evidence/`：数据准备和转换验证记录。

## 安装与数据

建议在新机器克隆到 `/root/lingbot_challenge`，以匹配已验证配置。复制 `configs/paths.env.example` 为 `configs/paths.env`，填好 Conda、环境和真实存储位置。不要复制旧磁盘 UUID。若换目录，必须重新生成训练列表/config 并审计；历史 benchmark 脚本不保证任意目录可运行。

安装过程参考 [服务器安装说明](docs/LingBot_VLA_v2_Server_Setup.md) 与 `scripts/setup_training_from_handoff.sh`；安装不会启动训练。源码快照哈希检查仅适用于随仓库提供的未修改上游源码。外部安装下载仍需网络，干净新机安装未在本次上传中重跑。

模型、clean 原始/转换后数据、环境、checkpoint 均外部准备或挂载。资源来源/版本见 `configs/assets.manifest.json`、`configs/clean_archives.manifest.json` 和 `docs/versions/`。原数据文件与转换验证报告须配套保留，数据审计会核实它们。缺少资源时不能仅凭本仓库宣称训练可运行。

## 训练入口

`train.sh` 转发到现有 clean-only launcher，并默认保留单卡短测所需的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。入口要求显式 config，不自动开始任何工作：

```bash
bash train.sh /absolute/path/to/reviewed-clean-config.yaml
```

正式训练前确定训练步数、batch、学习率计划、GPU 拓扑和 checkpoint 空间，并生成新的 output_dir。可参考成功短测的 `full_clean50_smoke.yaml`，但其 11-step 调度不是正式训练预算。`enable_resume: false` 时省略 `load_checkpoint_path`，不要写显式 YAML null。禁止覆盖旧 run 或自动恢复停止的 baseline。

`run_initial.sh` 是保存的原服务器短测入口，含单 A100/显存检查及 20 分钟上限；不是正式长训入口。恢复、HF 导出和新硬件验证需按各自范围单独完成。不要把早期 frozen-VLM 工具误当成本次全参冒烟路径。

本次 GitHub 交付只完成代码整理、语法/哈希/敏感内容检查和上传，不新增训练、仿真或提交。训练完成后保存产物，及时检查云资源计费并按需释放。
