# LingBot Challenge Training & Step2000 Replay

训练部署、checkpoint 转换与历史 500 局 BF16 本地评测的代码。**不包含 checkpoint、训练数据、场景/RNG 数据包、密码或机器凭据。**

## 同步范围（2026-10-18）

- 同步服务器的 `run.sh`、`docker/`、`configs/train_8xA100.yaml`：检查、真实 batch 冒烟、正式训练及显式续训；每次运行保存实际配置与 provenance。
- 同步服务器 500 局评测的 policy、client、转换器、调度及恢复脚本。历史脚本保留用于审计；推荐使用新的 `eval.sh` 严格回放入口。
- 固定 RoboTwin `13c3c47ff4312dd62484bcd51be034af55c062d1`、cuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，LingBot 上游 revision 保持不变。
- 已完成的历史结果：clean **127/250**，randomized **89/250**，总计 **216/500 (43.2%)**。详见 [原结果报告](reproduction/step2000/evidence/REPORT.md)。这些是历史结果，不是本次同步重新运行的结果。

**复现边界：** 原 step2000 训练运行的 `run_config.yaml/run_provenance.json` 没有随 checkpoint 保存到本服务器，不能将 8×A100 模板冒充实际训练参数，也不能承诺从零训练生成相同 checkpoint。推理使用的 normalization 已记录并校验，但无法独立确证训练时使用的 normalization。500 局回放需要原权重和外部历史场景包；环境、场景哈希不匹配时直接失败，不静默换 seed，不保证跨硬件逐位相同成绩。

## 训练

完整资产、环境与 Docker 交付步骤见 [DEPLOY.md](DEPLOY.md)。本镜像只包含训练环境，**不包含 RoboTwin 渲染环境**。

```bash
# 构建需要 Docker BuildKit、网络和外部数据/模型；不启动训练。
bash docker/build.sh
bash run.sh check --full
# 明确启动训练冒烟，并保留 checkpoint 供检查：
KEEP_SMOKE=1 bash run.sh smoke
# 根据实际确认的预算设置 MAX_STEPS；下例仅演示，不能声称是历史训练命令。
MAX_STEPS=2000 MBS=4 GBS=1024 PRECISION=bf16 bash run.sh train
```

每次生成 `checkpoints/<run>.config/run_config.yaml` 和 `run_provenance.json`，与 checkpoint、日志一起保存在外部存储。另保留原生入口 `bash train.sh /absolute/path/to/reviewed-config.yaml`。禁止覆盖旧 run；不要用新的短调度代替原训练调度后声称复现了旧训练。

## 历史 500 局评测

见 [详细复现步骤](docs/step2000_reproduction.md)，包含外部依赖、场景包导出、DCP 转换、严格回放、恢复语义和已知异常。

```bash
# 在拥有原 results/step2000_bf16 的服务器上导出，不上传 GitHub：
python3 reproduction/step2000/replay500.py export-bundle \
  --history /root/lingbot_challenge/results/step2000_bf16 \
  --destination /root/deliverables/step2000_replay_bundle
python3 reproduction/step2000/replay500.py check-bundle \
  --bundle /root/deliverables/step2000_replay_bundle
# 完成文档中的环境与资产准备后：
nohup bash eval.sh --execute \
  --bundle /root/deliverables/step2000_replay_bundle \
  --weights /root/deliverables/trained_ckpt/global_step_2000/hf_ckpt \
  --output /root/lingbot_challenge/results/step2000_replay_01 \
  > /root/lingbot_challenge/logs/step2000_replay_01.log 2>&1 < /dev/null &
```

回放参数固定：50 tasks × clean/seen 与 randomized/unseen × 5 局，BF16、chunk50、compile=False、不录像、每局 policy seed42。先执行一局完整 smoke（计入500），再启动任务队列；没有新增专家调用。与 base 比较只使用双方完成且场景、指令和 seed 匹配的案例。

本次同步验证为：代码语法、CPU 输入/校验测试、服务器历史500场景包完整性核验；未启动新训练或 GPU eval，也未重新构建 Docker 镜像。原始代码快照哈希见 `reproduction/step2000/server_source_sha256.json`；新增入口的行为变更见复现文档。

---

# 历史训练准备记录

固定源码、clean 数据准备、训练入口及真实冒烟记录。遵守 [COMPETITION_RULES.md](COMPETITION_RULES.md) 第 3、4 节：仅 release clean/seen 数据、允许的原始 `lingbot-vla-v2-6b` 基座，训练不调用仿真。这份规则来自用户，不代表已独立核验全部外部来源。

## 当前状态（2026-10-14）

- 远端 50 个 clean 任务、2500 条演示、549787 帧已转换；本次现场只读审计通过。仓库包含清单、统计及验证报告，**不含数据或权重**。视频验证仅为每任务数据集首尾帧的三路解码，不是所有帧完整解码。
- 固定上游 `Robbyant/lingbot-vla-v2` commit `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`，完整 tracked 源码快照在 `repos/lingbot-vla-v2/`，原许可证保留。源码哈希见 `docs/upstream_source_sha256.json`；不是上游 Git 历史的镜像。
- 今天的 A100 80GB 全参冒烟完成 11 optimizer steps；DCP 保存、BF16 HF 导出和三类模块抽样权重更新通过。峰值 allocated 77.845 GiB / reserved 78.207 GiB。冒烟 checkpoint 已按用户要求删除，恢复训练未执行。
- 本次现场检查时 GPU 数量为 0；系统盘剩余约 67 GB，数据盘约 53 GB。这些是检查时快照，启动前必须重新核实。
- 已验收的是 batch=1/max_steps=11 短测，不代表正式训练预算、长期稳定性、双卡/H100、Docker 或比赛成绩通过。依赖仍有已记录的 upstream metadata conflicts。

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

## 数据增广（V1 / V2）

增广只作用于 dataloader；默认 `data.aug: {profile: none}`，原有 `image_augment`
默认关闭。V1 开启颜色、几何和 seen 指令替换；V2 增加纹理和外任务干扰物。
推理 `policy_eval=True` 无条件禁用全部新增广和指令替换。

| 配置 | 默认启用 profile 时的值 | 行为 |
|---|---|---|
| `color.p` | 0.5 | 亮度 0.7–1.3、对比度 0.6–1.4、饱和度 0.5–1.5；HSV hue ±4/256，噪声 std 0–2/255，blur radius 0–0.5 |
| `geometric` | p=0.3, min_area=0.90, max_shift=0.04 | 同宽高比裁剪再 resize，小幅反射边界平移 |
| `texture` | p=0.3, regions=[table,wall] | 颜色距离、开闭运算、边缘保护、大连通区域及充分画面边界接触；保留原亮度，约 2–3 px 羽化 |
| `distractor` | p=0.3, max_count=3, max_retries=20 | 只选其他任务 cutout；默认 camera_top；当前位置与未来帧的可贴区域取交集 |
| `instruction_swap.enabled` | true | 只从本条轨迹经 SHA 验证的 seen 列表中抽取 |
| `state_noise` | enabled=false, std=0 | 动作减 state 和归一化完成后，仅扰动模型输入 state |

教师 `pil_images/future_pil_images` 仅作与学生相同的几何变换，不接收颜色、纹理、
干扰物。`teacher_uses_clean=false` 会报错。三相机共享归一化几何参数，当前/未来帧
共享全部参数；这保证每一视角学生/教师 patch 对齐，但不声称像素平移等价于真实
多视角三维相机运动。每个样本的 RNG 从 `(train.seed, epoch, global sample index)`
派生，seed/epoch 在数据加载器中传递，epoch 使用共享内存使 worker 可见。新 profile
下数据错误直接报错，不使用上游的随机换样本重试。

初始化时验证全部 50 个 manifest 与 2500 条原始 seen JSON、转换 provenance、
LeRobot episode metadata。转换适配器按数字 episode 排序；每条现有 prompt 必须
属于所映射原始 episode 的 seen 列表，任何错误终止。不会静默回退到固定指令。

V2 使用 RoboTwin `assets/background_texture/seen` 素材存在规则灰区，**待主办方确认**；
本任务由用户明确授权实现。解析后路径包含 `unseen` 段（包括 symlink）会硬性拒绝。
`texture.source=procedural` 提供木纹、大理石、织物、瓷砖、条纹、棋盘格、Perlin 噪声和
色块备用，不读取 seen 图片。Perlin 使用随机二维单位梯度与 quintic fade；程序色彩范围
为保守手设值，未声称匹配 seen 库统计。两份待确认配置默认使用
procedural；改成 `robotwin_seen` 即可启用素材。

干扰物库只读经 SHA 校验的 release clean 首帧。提取桌面 mask 内封闭组件，过滤
边界、下方区域、细长、暗色或缺少颜色信息的组件（因此也舍弃无彩色物体）。**这是保守启发式，不能保证与物体同色的机械臂部件
均被排除**；必须审阅可视化和 mask 报告后再用于正式训练。无法区分桌面/墙面颜色
时跳过墙面替换；也不宣称桌面 mask 对所有任务完美。

### 构建和 CPU 验证

仓库保留原始 vendored 源码。Dockerfile 在最后一次原始源码复制及
`docker/verify_source.py` 验证后应用 `patches/dataloader_aug.patch`，再验证独立的
`docs/patched_source_sha256.json`。本地应在隔离副本应用补丁：

```bash
python docker/verify_source.py
git apply --check patches/dataloader_aug.patch
git apply patches/dataloader_aug.patch
python scripts/verify_aug_patch.py
export LINGBOT_ROOT=/path/to/verified-clean-root
export AUG_SOURCE="$PWD/repos/lingbot-vla-v2"
export AUG_ORIGINAL_SOURCE=/path/to/pristine/lingbot-vla-v2
export PYTHONPATH="$AUG_SOURCE"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tests/test_dataloader_aug.py
python scripts/build_distractor_library.py --root "$LINGBOT_ROOT" \
  --source "$AUG_SOURCE" --output /persistent/augmentation/distractors
python scripts/validate_aug_visuals.py --root "$LINGBOT_ROOT" \
  --source "$AUG_SOURCE" --output /persistent/augmentation/visuals \
  --library /persistent/augmentation/distractors \
  --seen-dir "$LINGBOT_ROOT/repos/RoboTwin/assets/background_texture/seen" \
  --texture-source robotwin_seen
```

所有输出目录要求不存在，避免覆盖历史证据。原始数据、纹理库、干扰物库和 checkpoint
不入库；只提交用户明确要求的 10 张审查预览。测试依赖沿用固定训练环境（torch、
torchvision、numpy、Pillow、OpenCV、h5py、pandas/pyarrow）。

### 8k 增广续训尚不可直接启动

完整复现链应为「基座 → 8k（实际原配置及 provenance）→ 增广继续训练」。
`configs/train_8xA100_aug_v1.yaml` 和 V2 是 **BLOCKED 草稿**：3000 步、LR 5e-5 cosine
至 1e-5、每 1000 步保存，均待用户确认；其余参数目前取仓库模板，不能当作实际 8k
训练参数。8k 及其 provenance 的查找结果见 `AUG_REPORT.md`。

现有 `run.sh resume` 会恢复优化器、调度器和步数，**不支持 weights-only**。
原训练代码的 `model.model_path` 可作为新模型初始化入口，但 launcher 固定要求基座路径，
因此仍需用户决定是否扩展 launcher，验证 8k 导出和来源后再启动新优化器训练。
本任务没有修改训练循环、模型或 eval，也不启动正式训练。
