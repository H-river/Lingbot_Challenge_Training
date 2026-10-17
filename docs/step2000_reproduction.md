# Step2000：资产约定、严格回放与历史差异

## 哪些内容能复现

服务器代码与固定依赖已经同步。历史训练实际的 batch/precision/MAX_STEPS 覆盖参数没有留在当前服务器，训练配置来源记录明确为 `actual_training_run_config_available: false`。因此只能复现已记录的训练流程，不能从现有证据还原整个 step2000 训练实验。要从零精确复现实验，须另取得训练机的 `<run>.config/run_config.yaml`、`run_provenance.json`、日志、代码版本和数据清单；不可据 step2000 名称推断原 `MAX_STEPS=2000`（学习率调度可能原计划更长）。

训练配置生成器保存全部有效参数、环境覆盖、base/run 配置哈希及 GPU 数；训练 Docker 与原生入口均保留。`configs/train_8xA100.yaml` 默认 FP32，评测使用 BF16；这是两件不同的事。

## 目录与外部资产

严格回放要求仓库路径 `/root/lingbot_challenge`，因为历史 setup hash 包含绝对路径。改路径会改变哈希，不能关闭断言掩盖差异。

- `envs/lingbotvla`：Python 3.12 / Torch 2.8.0+cu128 / FlashAttn 2.8.3，参考已有 `docs/versions/lingbotvla*`。
- `envs/robotwin`：Python 3.10 / Torch 2.8.0+cu128，含 cuRobo、PyTorch3D、Warp、SAPIEN；参考 `docs/versions/robotwin*` 和 `docs/LingBot_VLA_v2_Server_Setup.md`。版本文件是历史快照，不代表新机器已验收。
- RoboTwin revision `13c3c47ff4312dd62484bcd51be034af55c062d1`，cuRobo revision `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。源码不重复 vendoring，先按安装文档准备，再核对 revision。RoboTwin tracked 文件不得修改。
- RoboTwin 仿真资产、机器人、背景和对象素材必须单独准备，保持原相对路径/版本；缺失资产不能用空素材替代。此同步没有下载或发布这些数据。
- 原始模型资产放在 `models/lingbot-vla-v2-6b`、`models/Qwen3-VL-4B-Instruct-config`、`models/MoGe-2-vitb-normal`。路径与 `reproduction/step2000/inference.yaml` 保持一致。
- 填写本机 `configs/paths.env`（不入 Git），设置 `LINGBOT_ROOT`、`LINGBOT_ENV`、`ROBOTWIN_ENV`。保留已有正确存储挂载配置，不复制别的机器 UUID。
- CUDA/Vulkan 必须正常识别 GPU；不要只确认 Torch 可以使用 CUDA。历史硬件为 A100 80GB，两套模型服务共享 GPU。不同 GPU/驱动/仿真环境可能产生不同轨迹。

可按安装文档取得源码，然后固定 revision：

```bash
git -C repos/RoboTwin checkout 13c3c47ff4312dd62484bcd51be034af55c062d1
git -C repos/curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
```

只在新建 checkout 上操作；不要覆盖本地未提交改动。训练镜像不提供 SAPIEN 图形栈，eval 使用独立原生环境。

## 转换 checkpoint

优先直接使用历史已校验的 `hf_ckpt`。如果只有 DCP，先确认传输端成功退出、所有模型和 optimizer 分片及 metadata 到齐；文件大小稳定不足以证明传输完成。

```bash
source configs/paths.env
"$LINGBOT_ENV/bin/python" reproduction/step2000/export_checkpoint.py \
  --checkpoint /root/deliverables/trained_ckpt/global_step_2000 \
  --output /root/deliverables/trained_ckpt/global_step_2000/hf_ckpt_new \
  --state-dir /root/lingbot_challenge/results/step2000_export_new
```

转换器沿用服务器的 Torch 2.8 DCP 内部 API，按约1.5GB分组导出，检查 metadata 所引用字节范围、张量键/形状、有限值、逐张量读回数值及 SHA256。只读取可信自有 checkpoint。拒绝已存在的输出目录；中断时保留证据，用新目录重新导出。保留源 checkpoint，不修改原件。

严格回放要求全部权重文件的 SHA256 与 `evidence/export_verified.json` 一致；新转换的数值相同并不一定保证序列化字节相同（工具版本、键顺序可影响文件），若字节不同，不要删除检查冒充历史权重，应先做逐张量独立验证或使用原始导出文件。本入口不是任意新 checkpoint 的通用打分器。

## 外部历史场景包

`export-bundle` 从原 `results/step2000_bf16/cases` 导出500份 seed、instruction、初始哈希和 simulation RNG 元数据，要求500个历史策略结果均 complete。该包不包含模型或训练演示，但仍作为外部评测数据保留，不上传 Git。

运行 README 中的 export/check 命令。包在新机器上须配套搬运；不能从成功率报告反推出这500个场景。原服务器导出目标为 `/root/deliverables/step2000_replay_bundle`。

## 启动、恢复与输出

运行 README 的 `nohup bash eval.sh ...`。新的输出目录必须为空/不存在；不得指定原 `results/step2000_bf16`，不恢复旧 `STOPPED_BY_USER` base。

- 先校验500场景身份、checksum、原模型分片hash、normalization、RoboTwin版本。
- 安装必要的新增 client/policy 文件时，如果目标已有不同内容则拒绝覆盖。原 tracked 上游代码保持不变。
- 端口默认为9430/9431，可用 `--port` 更改第一端口；端口已被占用就报错，不杀别人的进程。
- 固定 BF16 / chunk50 / compile=False / no video / policy seed42。完整第一局 smoke 算在500局中。
- 每局结果在 `bf16/{setting}/{task}/{index}.json`；记录 status、seed、哈希、步数、耗时和成败。策略失败正常计入。
- `status.json`、`summary.json`、各任务日志和 server 日志保留；结束或异常退出时清理自己启动的模型服务。
- 恢复需显式加 `--resume`，输入fingerprint必须一致；已完成局跳过，`started` 但未完成的局拒绝自动重跑，防止重复计数。初始化哈希不匹配不做自动场景替换。

`--help` 与 `check-bundle` 不加载 Torch/SAPIEN、不运行 GPU。仅同步代码不能代替实际新环境验收。

## 历史500局的异常口径

总结果216/500包括10局专家预检查上限后的策略测试（其中1成功）以及2局同seed/config但历史初始化哈希不同的场景（均失败）。两个替代场景在执行策略前验证了当前重复初始化的稳定性，并保留旧清单。原始恢复脚本还对初始化不稳定做了有界检查；没有重跑已完成策略结果。

新入口只回放最终已记录的500场景，不重复执行上述恢复/采集流程；因此不会再次调用专家，也不会再次替换场景。12局特殊案例标记从场景清单继承，配对base比较必须排除不匹配项。历史报告的295局配对比较不是完整500局base比较。

`original/`、除 `eval_policy_client_replay500.py` 外的 `robotwin/eval_policy_client_*` 是原服务器脚本存档，含历史绝对路径及授权文件读取，不是推荐的新入口。没有复制授权文件，不会伪造主办方豁免。不要直接运行这些历史脚本覆盖旧结果。

## 与服务器原版相比的包装改动

- 新增参数化且隔离输出的 strict replay runner，保留原版策略、噪声、加载方式和 hash 检查。
- 回放 client 要求每局有保存场景，禁止现场新采样；不再依赖聊天生成的本地授权JSON，必须显式 `--execute`。
- 导出器参数化输入/输出并拒绝覆盖；模型计算逻辑未改。
- Docker 下载固定服务器 Miniforge版本与SHA256、utils3d SHA256；未上传vendor安装包。对配置生成器增加零GPU和无效smoke步数校验。
- 保留原始源文件哈希清单供审计；被包装修改的文件不宣称与服务器逐字节相同。
