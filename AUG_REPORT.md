# V1 / V2 数据增广报告

日期：2026-10-18。开发基线：`codex/sync-server-train-eval` 的 `e946171e3db5aab820164c260621f30b41c08a73`。
目标分支：`feat/dataloader-aug`。本次只做离线数据处理和 CPU 验证，没有正式训练、GPU eval、仿真调用或 checkpoint 修改。

## 结果与适用范围

已实现 V1/V2 数据管线补丁和验证工具，13 项 CPU 测试通过、全部 2500 条 seen 映射校验通过。
这是待真实 GPU 冒烟的实验性增广实现，不能据此宣称 8k 续训已 ready：服务器未找到 8k/provenance，且现有 launcher 不支持所需 weights-only 路径。

## 文件与补丁证明

- `patches/dataloader_aug.patch`：新增 `augment.py`，最小修改 `data_loader.py`、`base_dataset.py`、`multi_vla_dataset.py`、`transform.py`、`utils.py`，以及 `arguments.py` 仅新增 DataArguments.aug 字段。
- `docs/patched_source_sha256.json`、`scripts/verify_aug_patch.py`、`docker/Dockerfile`：原始源码验证后再应用补丁，并校验修改后哈希。
- 两份 `configs/train_8xA100_aug_v*.yaml`：明确 BLOCKED / 待用户确认的配置草稿。
- `scripts/audit_aug_inputs.py`、`build_distractor_library.py`、`validate_aug_visuals.py`、`benchmark_aug_loader.py`：离线验证及素材生成。
- `tests/test_dataloader_aug.py`、README、COMPETITION_RULES 第 6 节、train.sh 的复现链注释、本报告。
- `evidence/augmentation_validation/`：测试文本、输入审计、patch 证明、逐任务/相机覆盖 CSV、吞吐量和可视化统计。
- `evidence/augmentation_preview/`：按用户明确授权，仅 10 张检查用 JPG 拼图；无源纹理、原始训练数据、cutout 库或权重。

原始 vendored 源码已恢复。`python3 docker/verify_source.py`：**358 files PASS**；原始清单和验证器未改动。
`git apply --check`：PASS；隔离副本实际应用后 **7 files hash PASS**，与远端测试副本逐文件一致。
训练循环、模型、deploy、eval 文件均无变更。没有执行完整 Docker 镜像重建。

## 数据管线设计

seed/epoch/global sample index 独立 RNG；数据加载器原有 set_epoch 接口将 epoch 传入共享内存，不改训练循环。
三相机共用归一化几何参数，当前/未来帧共享颜色、几何、纹理和贴图位置。选择共享参数是为了固定扰动强度和时序一致性；不声称模拟真实三维相机移动。
教师仅几何；学生额外颜色/纹理/干扰物。state 扰动发生在相对动作计算和归一化之后，不改变 labels。
policy_eval 无条件禁用新逻辑。profile none 保留旧路径。

seen 校验包括 50 个任务各 50 条轨迹、JSON 仅 seen 字段、原文件 SHA、转换 provenance 的合并/单任务 manifest 一致性及原 prompt 成员关系。
转换脚本明确采用数字 episode 排序，避免字典序错位。不是根据 prompt 猜测 episode。

V2 纹理同时支持 robotwin_seen 和 procedural（8 种，包括梯度 Perlin）；程序色彩范围手设，未使用 seen 库统计。
robotwin_seen 对解析后的 unseen 路径（含 symlink）硬性拒绝；procedural 运行时不读取 seen 图片。
seen 素材是用户授权的规则灰区，仍待主办方确认。

## 测试

13 项 CPU unittest 全部 PASS，日志见 `evidence/augmentation_validation/cpu_tests.txt`：

- 原始/修改后的 prepare_images 及 FeatureTransform 关闭增广路径逐位比较。
- 实际 FeatureTransform 的 policy_eval 防护：图像、state 和 prompt 不被增广上下文修改。
- 教师只接受几何；当前/未来帧一致；V2 含纹理/贴图时也一致。
- worker 0/2/4 同样本逐位一致；独立于全局 torch/numpy RNG。
- unseen 路径与 symlink 拒绝；procedural 不调用素材目录读取。
- 全部 2500 条映射 PASS；动作 labels 不受 state 噪声影响。
- 干扰物生成的相同输入产出逐字节相同 manifest（小型固定输入回归）。
- 封闭物体反光区域不能被认作背景的回归。

CPU 单测使用部分受控输入/processor stub，不等价于真实模型前后向验证。

## 可视化与 mask 审查

持久化磁盘 `/root/deliverables` 为 `/dev/vdc` ext4。
最终 100 组（50 tasks × 2 随机样本，seed 42）none/v1/v2 × 三视角 × 学生/教师拼图：
`/root/deliverables/dataloader_aug_validation/visuals_v3/`。
10 张代表性预览已逐张查看；全量统计见 CSV。没有逐帧人工分割 ground truth，不能保证未抽查帧零误分。

首轮发现 `place_container_plate` 金属碗内部反光被当成背景；已增加边缘保护、至少两条画面边界接触及最小接触长度，复查同一案例已不再替换碗内部，回归测试通过。
代价是偏保守，阴影/近色区域会留下未替换的块或白边，不能声称写实完整的背景替换。
最终样本中桌面/墙面主色混淆任务数为 0；头部桌面 mask 覆盖率至少一次 <20% 的任务共 27 个：

beat_block_hammer, blocks_ranking_size, click_alarmclock, click_bell, handover_mic, hanging_mug, lift_pot, move_stapler_pad, open_microwave, pick_diverse_bottles, pick_dual_bottles, place_bread_basket, place_bread_skillet, place_burger_fries, place_can_basket, place_dual_shoes, place_empty_cup, place_fan, place_object_basket, place_object_stand, place_shoe, put_bottles_dustbin, put_object_cabinet, rotate_qrcode, scan_object, stack_blocks_three, stamp_seal

低覆盖不等于任务失败，可能是遮挡、主体占屏或保守过滤；这些任务不应未经 GPU 小试直接启用激进纹理概率。

干扰物库：`/root/deliverables/dataloader_aug_validation/distractors_v2/`，**500 个**。
从 2500 条 SHA 验证 clean 轨迹首帧提取，过滤白色/阴影碎片、暗色/细长/边缘候选，故不覆盖所有物体类型。
贴图与当前/未来帧保守桌面 mask 的交集严格不重叠保护区域；但颜色分割本身仍是启发式。
初版库/可视化保留在同一验证目录供审计，不用于最终配置。

## CPU 吞吐量

真实 clean HDF5/JPEG，100 样本，batch=1，worker=0，每 profile 3 遍取中位数；包含当前/未来三视角解码、增广、学生归一化。

| profile | 样本/秒 |
|---|---:|
| none | 109.16 |
| v1 | 20.11 |
| v2 | 11.94 |

这是 raw-image DataLoader 测试，**不包含真实 LeRobot H264 解码、tokenizer、state/action collation 或模型计算**；不能外推训练 step/s。
V2 比 none 低约 89.1%，仍超过 20% 目标。
已做有界 mask 缓存、episode 首帧主色缓存、seen 图像/目录缓存、cutout 缓存、按样本一次生成共享纹理，纹理画布由 1024 缩至 256。
开销仍显著，不能宣称性能达标；实际训练瓶颈须在 GPU 冒烟后测量。完整重复计时和纯图像阶段计时均保留。

## 8k checkpoint / weights-only

检查 `/root/deliverables/trained_ckpt`、`/root/lingbot_challenge/checkpoints`、`/root/lingbot_build`、`/mnt/lingbot-data` 的相关目录：只找到 step2000 和历史 smoke，没有找到 8k 及配套 run_config/run_provenance。
这不是断言其它机器上不存在 8k。新配置中余下超参来自仓库 train_8xA100 模板，**不能冒充 8k 实际超参**。
LR 5e-5 -> 1e-5 cosine、3000 steps、每 1000 保存均标为待用户确认，模型路径故意为不可用占位。

现有 run.sh resume 会恢复优化器、scheduler、global_step 和 RNG；launcher 还硬性要求 model_path 为基座。因此当前完整 launch 流程**不支持**指定 8k weights-only。
需要用户决定是否允许扩展 launcher，并提供 8k/provenance/经校验的模型导出；模型初始化接口本身已有 model.model_path，可在新训练输出目录初始化权重并新建优化器，无需为此修改训练循环。
本任务没有绕过 launcher，也没有实际运行 8k 续训。

## GPU 冒烟与关机

`nvidia-smi` 返回无法与 NVIDIA 驱动通信，`lspci` 未列出 NVIDIA，未发现 `/dev/nvidia*`。
没有可用 CUDA GPU，**V1/V2 各 20 步冒烟未运行**，不是通过；未尝试擅自安装驱动或重启。
没有启动正式 train/eval，历史结果保持不变。外部产物位于持久化磁盘。

**不执行关机**：无法完成用户要求的 nvidia-smi GPU 进程确认；且真实 GPU 前后向仍未验证。
发布前会检查本任务的 CPU 验证进程已退出，保留当前实例供后续处理。最终发布状态和 commit 由交付回复及远端分支确认。

## 尚需处理

1. 找到 8k checkpoint 与原训练 provenance，决定是否扩展 weights-only launcher。
2. 在有可用 GPU 的实例执行两版各 20 步真实模型前后向，之后才能评价训练可用性。
3. 如需正式使用 V2，审查剩余近色/遮挡 mask 风险及完整训练吞吐量；当前 CPU 开销未达 20% 目标。
4. 主办方确认 seen 纹理例外；未确认可用 procedural 配置，不能把用户授权当作比赛规则批准。

## 2026-10-18 修订（V2 纹理质量 + 亮度）

由用户确认后修改，改动只在 `augment.py`（重新生成 patch 和哈希清单）、两份配置、测试与文档。

1. **亮度收紧到 0.75–1.1**，对比度/饱和度范围不变，三者均改为可配置（`color.brightness/contrast/saturation`）。
   原因：RoboTwin clean 场景接近白色，上游 1.3 会大面积过曝（见 put_object_cabinet 原预览）。
2. **头部相机阴影感知 mask**（`texture.shadow_aware_cameras: [camera_top]`）。原保守 mask 把机械臂阴影留成亮白剪影，
   与 randomized 场景中阴影落在纹理上的样子相反。新 mask 以原保守 mask 为种子，向「同色度、亮度 0.5–1.08 倍」的
   候选区域生长，只在硬边缘（Canny 60/150）处切开；被硬边缘围住、与种子无重叠的灰色/金属物体不会被替换。
   腕部相机保持保守 mask：全相机开启时 click_bell 腕部视角的铃会被替换，已用于确定默认值。
   已知取舍：透明物体（place_container_plate 玻璃碗）在头部视角会透出纹理，边缘保留。
3. **墙面/桌面接缝填补**（`texture.fill_seam: true`）：位于两块 mask 之间（距离和 ≤ 28 px）且颜色接近任一表面的像素，
   分配给较近的表面；机械臂和物体因颜色差异不受影响。
4. **V2 纹理来源改为 `robotwin_seen`**（用户决定，贴近 randomized 场景；规则灰区，已披露；unseen 硬性拒绝不变）。

验证：
- 新增 4 项测试（亮度范围与配置覆盖、阴影生长且保护被硬边缘围住的物体、阴影感知只作用于配置的相机、接缝填补且机械臂不受影响）。
- 在无服务器数据的 CPU 环境中，按 Dockerfile 顺序（原始源码校验 → 打 patch → patched 哈希校验）后，以下 14 项通过：4 项新增 + workers、global_rng、unseen、procedural、teacher_temporal、
  enclosed_reflection、profile_gate、v2_temporal_and_distractors、state_labels_unchanged、library_reproducible。
  test_none_original_prepare_images、test_feature_transform_inference_and_labels、test_all_2500 需要完整训练环境与真实数据，
  本次未在该环境中运行（这三项覆盖的 utils.py/transform.py/seen 映射本次未改动），**训练前须在服务器上重跑全部测试**。
- `evidence/augmentation_preview/revision_20261018/v2_before_after.jpg`：原图 / 修订前 / 修订后。
  该图的输入是本仓库原有预览图中裁出的 clean 帧（192 px 放大、任务中途帧），纹理为 procedural，干扰物为演示用小库，
  仅用于对比 mask 行为，不代表正式训练的素材和分辨率。正式配置下须在服务器上用 `scripts/validate_aug_visuals.py`
  （`--texture-source robotwin_seen`）重新生成可视化并复查 mask 覆盖率。
- 单张 320×240 图 mask 计算（CPU，缓存清空）：保守 mask 约 4 ms，阴影感知 mask 约 11.6 ms（含保守 mask）。只作用于头部相机，按纹理概率 0.3 摊到每个样本约 +9 ms，相对 V2 约 +10%。完整训练吞吐量仍须在 GPU 冒烟时测量。

## 2026-10-18 V3：ORACLE 对照实验（禁止提交）

用户要求的控制变量实验：V3 = V2 + 纹理池扩大为 seen ∪ unseen（评测纹理）。V3 − V2 衡量见过测试纹理的收益。
- 实现：`texture.source: robotwin_seen_plus_unseen_ORACLE`，在两个池的并集上均匀抽一张，只消耗一次 RNG，
  因此与 V2 的逐样本随机决定完全一致（有测试）。仍使用 profile v2。
- 防护：数据加载与 launcher 双重要求 `LINGBOT_UNSEEN_TEXTURE_ORACLE=NOT_FOR_SUBMISSION`；output_dir 必须含
  `ORACLE`、非 oracle 运行禁止含 `ORACLE`；launcher 写旁路标记文件；unseen 目录内 symlink 逃逸被拒绝；
  `safe_path`、seen 纹理加载、指令与干扰物路径对 unseen 的拒绝保持不变（有测试）。
- 新增 2 项测试（未开启 opt-in 拒绝、其它 unseen 防护不变、仅 v2、并集抽样覆盖两个池、与 V2 逐样本对齐）。
- 训练/评测时 V3 的结果只能作为诊断依据，不得进入提交、README 成绩或复现链。

## 2026-10-19 V2b：打破 shortcut 的增广 + 离线 shortcut 测试

动机：评测观察到策略"面前有非白色东西就移动/抓取"（另见失败分析：双臂争抢、闲置手碰倒物体、抓把手旁的杂物）。
- 新增 `texture.mode: overlay`（无分割的随机分割线 + 半透明叠加）与 `distractor.placement: v2b`（三相机、物体旁 hard negative、
  闲置手聚焦、接触阴影）；新增 `configs/train_8xA100_aug_v2b.yaml`。
- `utils.py` 只增加一行：归一化后调用 `aug_context.observe_actions(item)`，用 release 动作块判断闲置手；默认配置下为空操作。
- 回归：默认路径（mask + v1）与上一 commit 的 augment.py 在 60 个样本上逐位一致。
- 放置成功率（预览帧 320×240、演示用小库，每相机 30 次）：至少放上一个的比例 62–90%，平均 1.2–2.3 个；头部画面闲置侧约束全部满足。
- 新增 8 项测试（默认路径不变、overlay 教师/时序/全图、overlay 公式与腕部无墙、闲置手检测、闲置手聚焦的相机与左右侧、
  干扰物先贴后叠加纹理且有阴影、hard negative 靠近已有物体、probe 选帧）。
- `scripts/shortcut_probe.py`：`--dry-run` 已在模拟 release 目录结构上端到端跑通；带模型推理部分未在 GPU 上运行。
- 待服务器确认：真实数据上"有闲置手"的样本占比（阈值 0.05 为归一化单位）、V2b 可视化与吞吐量、probe 在 8k 上的基线结果。
- 预览图：`evidence/augmentation_preview/revision_20261019_v2b/`（每张 3 行 = 头部/左腕/右腕；4 列 = 原图、V2b 普通、闲置左手、闲置右手）。
- 已知局限：叠加是全图的，物体和机械臂也会被轻微染色（无分割的代价）；演示库缺少白色/灰色物体；腕部与头部的干扰物位置互不一致；
  头部闲置侧放置偶有失败（该样本退化为只有纹理）。

## 2026-10-20 V4：程序生成的几何干扰物（打破 shortcut）

> 同日修订：本节描述的立体渲染版本已改名为 **V5**（`configs/train_8xA100_aug_v5.yaml`，与本节 commit `29c2b68` 逐位相同），
> V4 改为纯 2D 形状 + 颜色，见下一节。本节的证据文件已移到 `evidence/augmentation_validation/v4_v5/`、
> `evidence/augmentation_preview/revision_20261020_v4_v5/`，原始预览图保留在 `29c2b68` 的 git 历史里。

动机与 V2b 相同：clean 训练的策略"夹爪前面有非白色东西就移动/抓取"。V2b 的干扰物来自 release 首帧抠图，抠图规则丢弃了
无彩色物体，所以面前出现的东西几乎都是彩色的，"彩色 = 该去抓"仍然成立。V4 把干扰物换成**程序生成的几何干扰物**，
颜色按大类均匀抽样（含白、灰、金属色），用途是打破这个 shortcut，不是模拟真实物体或 randomized 场景。

实现（只改 `augment.py`，其余 6 个补丁文件逐字节不变；重新生成 patch 与 `docs/patched_source_sha256.json`）：
- 新增 `distractor.source`：`library`（默认，原路径）/ `procedural`。`procedural` 只在 `placement: v2b` 下可用，
  否则配置校验直接报错；校验不消耗随机数。
- 形状：长方体、圆柱、瓶身加瓶颈（瓶盖另取颜色）、球、碗状半椭圆；Lambert + Phong 高光，金属加环境反射带；
  光照方向随机、来自上方；超采样（小物体 3×、高度 ≥48 px 时 2×）后按预乘 alpha 下采样，边缘柔和无黑边；
  低频亮度扰动 ±2.5%。物体按最终尺寸直接渲染（`size` × 画面宽/320 × 腕部 `wrist_scale`）。
- 颜色：13 个大类均匀抽样，类内在手设 HSV 范围取值；金属 75% 银色、25% 黄铜色。范围手设，未用任何数据统计。
- 新增 `configs/train_8xA100_aug_v4.yaml`：与 V2b 相比只多 `source: procedural` 与 `procedural:` 参数、去掉 `library_dir`（有测试逐键核对）。
- `scripts/shortcut_probe.py` 新增 `--distractor-source`（默认 `library`，行为不变）。
- 新增 `scripts/compare_aug_versions.py`（新旧 augment.py 逐张量回归）与 `scripts/preview_v4_distractors.py`（对比图 + 放置统计；现为 `scripts/preview_procedural_distractors.py`）。

合规：零仿真调用；不读外部数据、unseen 资产或任何图片素材（procedural 路径不打开 `library_dir`，有测试）；
标签、指令、state 处理均不变；教师仍只做几何。按"训练期视觉增广"处理，但贴合成物体未在允许清单中逐项列出，
**⚠️ 规则灰区，建议向主办方确认**（与 V2 seen 纹理一并确认）。

验证（本次 CPU 环境：Python 3.12、torch 2.8.0、numpy 1.26.4、opencv 4.11.0.86、Pillow 11.3.0，版本取自仓库锁定清单）：
- 逐位回归：上一 commit 与本次的 augment.py，在 V1 / V2 / V2b / V3 四个配置上各 120 个样本（当前帧 + 未来帧 × 三相机，
  学生/教师共 1440 个张量），**全部 torch.equal**。回归确实走到了被比较的分支：V2 放置 33 个、V2b 放置 208 个库干扰物，
  V2b 中闲置手左/右/无三种情况轮换。输入是从上一版预览图裁出的 4 个任务 clean 帧，纹理和抠图库是临时夹具
  （见 `evidence/augmentation_validation/v4_v5/regression_v1_v2_v2b_v3.json`）。
- 单元测试：新增 8 项（显式 `library` 与默认逐位相同；procedural 不读库且保持教师干净/当前未来一致；5 形状 × 13 颜色类
  均有软边缘、有明暗、底边贴底；白色物体无彩色但有明暗；形状与颜色类均匀；确定性且不受全局 RNG 影响；腕部按最终尺寸渲染；
  配置校验；V4 配置 = V2b + procedural）。全部 35 项中 34 项在本环境通过，日志 `evidence/augmentation_validation/v4_v5/cpu_tests.txt`。
  `test_all_2500` 需要服务器上完整的 2500 条 release 数据和转换 provenance，本环境没有数据，**未运行**（本次未改动其覆盖的
  seen 映射代码），训练前须在服务器上重跑全部测试。
- 放置统计（同一批裁出帧，每任务每相机 30 次，`placement.json`）：至少放上一个的比例 头部 72.5% / 左腕 86.7% / 右腕 70.8%，
  平均 1.46–1.74 个；头部闲置侧约束 276/276 满足。规划 + 粘贴耗时中位数 头部 12 ms、腕部 17–18 ms（单线程 CPU），
  按 `distractor.p` 0.5 与三相机摊到每个样本约 +20–25 ms，真实训练吞吐量仍须在 GPU 冒烟时测量。
- 预览：当时为 `evidence/augmentation_preview/revision_20261020_v4/`（见 `29c2b68`），已逐张查看。

已知局限：几何体都直立，没有倒放、旋转、纹理图案或文字；光照是简化模型；头部画面桌面 mask 小时常放不下（与 V2b 相同）；
预览与统计用的是 JPEG 再压缩的中途帧，正式训练前须在服务器上用 `--root` 重新生成并复查；
V4 训练后的 shortcut probe 应以 `library` 抠图物体为主要判据（procedural 物体是训练分布内的）。

## 2026-10-20 V4 改为纯 2D，立体版改名 V5

用户决定写两版：V4 = 纯 2D 形状和颜色，原来的立体渲染版改名 V5，两版都作为可选训练配置。

- `augment.py` 新增 `distractor.procedural.style`：`shaded`（默认，即 V5，行为与 `29c2b68` 完全相同）/ `flat`（V4）。
  `flat` 使用与 `shaded` 相同的几何轮廓（alpha 逐像素相同），整块填一种物体颜色，不做明暗、高光、瓶盖异色和材质扰动。
- 两种风格消耗相同的随机数；procedural 路径下即使 `shadow: false` 也照常抽取阴影强度（library 路径的代码不变）。
  因此同一样本在 V4 / V5 下形状、颜色、大小、位置完全一致，V4 − V5 只反映立体感和接触阴影本身。
- 配置：`configs/train_8xA100_aug_v4.yaml` 改为 `style: flat` + `shadow: false`；新增 `configs/train_8xA100_aug_v5.yaml`
  （`style: shaded` + `shadow: true`）。两份配置与 V2b 的差别由测试逐键核对。
- `scripts/preview_v4_distractors.py` 改名为 `scripts/preview_procedural_distractors.py`，可同时传入多个配置并排出图，
  新增 `--texture-p`（仅预览用的纹理概率覆盖，默认保持训练值）；`scripts/shortcut_probe.py` 新增 `--procedural-style`。

合规：与上一节相同。去掉明暗和阴影不改变合规三问的任何一项，"贴合成物体"这一灰区对 V4、V5 相同，⚠️ 仍建议向主办方确认。

验证：
- 逐位回归（每个配置 120 个样本、1440 个张量）：相对 `e79508a`（V4 之前），V1 / V2 / V2b / V3 全部相同；相对 `29c2b68`
  （第一版 V4），V1 / V2 / V2b / V3 与 **V5** 全部相同（`regression_v1_v2_v2b_v3.json`、`regression_vs_29c2b68_incl_v5.json`）。
- 单元测试共 37 项：procedural 相关测试改为同时覆盖两种风格，并新增 2 项（V4 每个形状 × 颜色类都是单一颜色且轮廓与 V5 相同；
  V4/V5 放置完全一致、V4 阴影强度为 0 且物体以外像素不变）。本环境 36 项通过；`test_all_2500` 需要服务器上的 release 数据，未运行。
  日志 `evidence/augmentation_validation/v4_v5/cpu_tests.txt`。
- 放置统计 V4 与 V5 完全相同（同一套随机数）：至少放上一个的比例 头部 72.5% / 左腕 86.7% / 右腕 70.8%；
  V4 规划 + 粘贴耗时中位数约 11–12 ms，V5 约 15–21 ms（单线程 CPU，`placement_v4.json` / `placement_v5.json`）。
- 预览 `evidence/augmentation_preview/revision_20261020_v4_v5/`，已逐张查看；纹理按训练概率抽样，所以多数列没有纹理。

已知局限（V4 特有）：白色/浅灰物体在浅色桌面上几乎看不见；平面色块与真实物体差异明显，模型可能学会"平的不用理"，
即 shortcut 换成"有立体感才是目标"。建议 V4 与 V5 都训练，用 shortcut probe（`library` 抠图物体）和评测结果比较。

