# LingBot-VLA v2：新服务器部署到可训练的交接手册

更新时间：2026-10-13，Asia/Singapore。交接依据是原实例的真实文件、实际已安装版本及已通过的双 A100 smoke，不是只按官方 README 推测。

## 给新 agent 的工作指令（可以直接复制）

请实际执行本手册，检查现场并排障，完成训练环境、允许的模型组件、50 个 clean 任务的 LeRobot 转换、clean-only norm、数据验收和 GPU 验收。保留已有环境/文件，所有安装失败必须明确退出。部署本身不自动启动长训练。用户安排训练模式后，可执行文末 11 步 checkpoint smoke；通过保存/回读验收后才能报告“可正式训练”。不要把旧的两任务 timing smoke 当成 50 任务 ready。

先读取新项目的 AGENTS.md、CLAUDE.md 和比赛文件。常规安装与可逆修复直接处理；凭证、未确认持久化目录、格式化、宿主驱动/服务、未知依赖冲突和规则灰区向用户确认。新 GPU 型号未知时先检测，不写死 CUDA 架构。

比赛边界：

- 只用 robbyant/lingbot-vla-v2-6b 起点；不下载/使用 lingbot-vla-v2-6b-robotwin 作为起点、教师或判据。
- 只用 RoboTwin2.0 Aloha-AgileX 50 tasks × 50 clean episodes，共 2500。训练与统计不能混 randomized、unseen instructions、外部数据。
- 离线转换与训练不得调用 simulator reset、step、render、check_success，不采集额外演示。
- 最终 50 任务混合训练同一份权重，不按 task/instruction 路由不同模型、不集成、不 TTA。
- 保留官方预训练教师配方，来源与用途单独记录，不把教师权重当新增训练数据。
- 不安装/启动/修改宿主 GPU 驱动或 nvidia-fabricmanager，不重启、不购买/挂卡。平台负责提供 GPU/驱动。

## 1. 先纠正“50 个任务 train-ready”的状态

2026-10-13 17:04 实际核查全部 50 个 manifest 任务：

| 阶段 | 原机实际状态 |
|---|---|
| 允许 clean ZIP | 50/50，约 23.78 GB |
| raw HDF5 + seen instructions | 50 tasks / 2500 episodes 已存在 |
| LingBot 实际读取的 LeRobot | **2/50 tasks、100 episodes** |
| adjust_bottle | LeRobot v3.0，50 episodes，7188 frames，50 FPS |
| beat_block_hammer | LeRobot v3.0，50 episodes，5682 frames，50 FPS |
| 另外 48 tasks | 无该任务的 LeRobot meta/info.json，尚未完成转换 |
| baseline_clean_norm/robotwin.json | 已存在，从全部 clean raw 计算的基线统计 |
| clean_norm/robotwin.json | **不存在**；全量 LeRobot 官方统计尚未完成 |

两任务均通过全部 state/action 行与 raw 的精确比对、三路视频首/末帧 native loader 解码。原协调器状态为 PAUSED_FOR_BASE500，completed_tasks 只含上述两项，2026-10-12 21:38 UTC+8 明确暂停转换。下载/解压 COMPLETE 不等于 LeRobot 转换 COMPLETE。如果此前声称“50 个都 train-ready”，该说法不正确。

证据在小交接包的 data_readiness_audit.json、cpu_continuation_status.snapshot.json、results/conversion_validation/。核查没有重新转换、未重哈希全部 raw；未来转换脚本会逐任务重新核对 raw SHA256。

## 2. 要带走哪些东西

本 MD 配套文件：lingbot_setup_handoff_20261013.tar.gz 和同名 .sha256。小包包含无凭证的精确 Conda/pip 导出、固定 manifest、下载/转换/统计适配器、已知冲突清单、配置和部署脚本。**小包不含模型、raw data、两项已转换视频、Conda 二进制环境或仿真资产。** 旧的 diag_20261013_artifacts.tar.gz 也只是诊断日志，不含训练资产。

旧实例最新入口（不是新训练机）：

~~~bash
ssh root@223.109.239.11 -p 28548
# 备用同一实例：
ssh root@180.127.11.169 -p 28548
~~~

已核验 SSH host fingerprint：SHA256:13F3VG0EfBW/o5vgMprzah7zqdPSoCWnCkvCL6OdEXE。旧实例地址端口会变，变化时由用户提供；不要拿旧 key 信任未知机器。密码由用户认证，本文和小包不含密码，不用 sshpass，不把密码写脚本/env/log。

小包位于旧机：

/root/lingbot_challenge/docs/setup_handoff_20261013/lingbot_setup_handoff_20261013.tar.gz

也保存在本地 D:\LingBot_Workspace。新 agent 有本地文件时直接上传；只有本文时，可从旧机取：

~~~bash
scp -P 28548 root@223.109.239.11:/root/lingbot_challenge/docs/setup_handoff_20261013/lingbot_setup_handoff_20261013.tar.gz .
scp -P 28548 root@223.109.239.11:/root/lingbot_challenge/docs/setup_handoff_20261013/lingbot_setup_handoff_20261013.tar.gz.sha256 .
sha256sum -c lingbot_setup_handoff_20261013.tar.gz.sha256
~~~

在模型/raw/已转换数据迁移并验证完成前，保留旧机文件或另做资产备份。200 GB 数据盘当前挂载不等于平台销毁实例后保证保留；不要根据 df/findmnt 单独判断持久性。

## 3. 固定版本与模型来源

| 项 | 固定值 |
|---|---|
| LingBot source | be969b8fd117fb70550c5d4bf4bc328211b5b1b6 |
| RoboTwin source | 13c3c47ff4312dd62484bcd51be034af55c062d1 |
| Linux | 原机 Ubuntu 24.04、x86_64；Conda explicit 为 linux-64 |
| 训练 Python | 3.12 |
| Torch/CUDA | 2.8.0 / cu128，Torch runtime CUDA 12.8 |
| torchvision / torchaudio | 0.23.0 / 2.8.0 |
| TorchCodec / TorchData / Triton | 0.6.0 / 0.11.0 / 3.4.0 |
| transformers / tokenizers | 4.57.3 / 0.22.2 |
| LeRobot | v0.4.2，转换产物实际为 v3.0；不是旧 v2.1 文件布局 |
| NumPy / datasets | 1.26.4 / 3.6.0 |
| utils3d | 3fab839f0be9931dac7c8488eb0e1600c236e183 |
| flash-attn | 2.8.3、cp312、CUDA12、Torch2.8、CXX11 ABI TRUE |

全部传递依赖以包内 docs/versions/lingbotvla.resolved-versions.json 和 portable-pins.txt 为准；原始 freeze/inspect/Conda explicit 也保留。不要直接 pip install 原 freeze：其中有旧机 file://、editable/VCS 路径，须用交接安装器分类重建。

| 资产 | revision | 范围/用途 |
|---|---|---|
| robbyant/lingbot-vla-v2-6b | 11c703bf6a5c1f45b3b69168482da11fdbba53d7 | 23文件、28,239,981,618 bytes；含 VLM 权重、LingBot-Depth、DINO-Video |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec70b05090aa6165b016eac8ec08e71b17 | 9个 config/processor/tokenizer 文件，约11.5 MB；原基座已含 backbone 权重，不重复下载4B权重 |
| Ruicheng/moge-2-vitb-normal | ca5f0e07ff01d3e5a364c1d954ed12ee1814b368 | 3文件、419,111,765 bytes；depth guide |
| TianxingChen/RoboTwin2.0 | 3dc3b798668feb99ac61cc9086d84cbcc3d79186 | 精确50个 aloha-agilex_clean_50 ZIP |

下载器使用发布者的 ModelScope 镜像时，每个文件的 mirror_revision、URL、hash 都在 assets.manifest.json/receipts；以逐文件 manifest 为准，不擅自切 main 或未知第三方包。HF基座 revision 与镜像 cosmetic 文件 revision 可能不同，权重需匹配记录的 hash。

参考：[固定官方安装脚本](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/tools/create_train_env.sh)、[RoboTwin准备说明](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/experiment/robotwin/README.md)、[训练配置文档](https://github.com/Robbyant/lingbot-vla-v2/blob/be969b8fd117fb70550c5d4bf4bc328211b5b1b6/configs/vla/Training_Config.md)。比赛约束高于其中 mixed clean/randomized 示例。

## 4. 主机、存储与目录

先读现有规则文件，再只读体检：

~~~bash
uname -a
cat /etc/os-release
nproc
free -h
df -hT
findmnt
id
command -v conda || true
command -v python || true
command -v nvcc || true
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status
nvidia-smi topo -p2p n
~~~

检查 GitHub、HF、ModelScope、PyPI/Conda 网络。新机若没有 GPU/驱动可用，完成 CPU 部分并报告 GPU_PENDING，联系平台，不安装驱动。

建议至少16 CPU、32 GiB RAM，64 GiB RAM更适合转换/保存。存储不能只算23.78GB下载：原机 models约27GiB，clean_raw约55GiB；两任务全部转换工作产物约1.3GiB。按帧数比例估计50任务工作产物约55GiB，压缩比例会变。另留环境/缓存及**首次完整checkpoint至少60GiB空余**。若全部放一块盘，300GiB以上可用空间更稳妥，持续保留20GiB余量；200GiB需跨盘规划，不能填满。

以下示例假定用户确认的持久化盘已挂载 /mnt/lingbot-data。不要自动格式化未知盘。把整个新项目放该盘，同时保留原绝对路径，避免搬迁 .pth/config 时隐性失效：

~~~bash
set -Eeuo pipefail
export LINGBOT_PERSIST=/mnt/lingbot-data   # 按真实、已确认的持久化位置修改
mountpoint -q "$LINGBOT_PERSIST"
export LINGBOT_ROOT=/root/lingbot_challenge
# 若/root/lingbot_challenge已存在，先检查，不能覆盖、移动或删除。
if [[ ! -e "$LINGBOT_ROOT" && ! -L "$LINGBOT_ROOT" ]]; then
  mkdir -p "$LINGBOT_PERSIST/lingbot_challenge"
  ln -s "$LINGBOT_PERSIST/lingbot_challenge" "$LINGBOT_ROOT"
fi
mkdir -p "$LINGBOT_ROOT"/{repos,envs,models,data/clean_raw,data/clean_processed,assets,cache,configs,scripts,logs,checkpoints,results,docs}
tar -xzf lingbot_setup_handoff_20261013.tar.gz -C "$LINGBOT_ROOT"
cd "$LINGBOT_ROOT"
sha256sum -c HANDOFF_FILES.sha256
~~~

若存在同名文件，先 diff；在新空目录解包后选择性复制，不能用整包覆盖当前工作。全新部署可直接上面命令。

不要复制旧 configs/paths.env 的 UUID。用新机实际挂载 UUID，并使用真实 Conda 二进制：

~~~bash
# 有现成Conda时，例如智星云常见路径：
export CONDA_EXE=/home/vipuser/miniconda3/bin/conda
"$CONDA_EXE" --version
# 没有时按下节安装后改为/root/miniconda3-lingbot/bin/conda
export LINGBOT_STORAGE_UUID="$(findmnt -n -o UUID --target "$LINGBOT_PERSIST")"
[[ -n "$LINGBOT_STORAGE_UUID" ]] || { echo '存储没有UUID，先核实存储类型及平台持久化保证；不要绕过guard'; exit 2; }
python3 - <<'PY'
import os,pathlib,shlex
r=pathlib.Path(os.environ['LINGBOT_ROOT']);v={
 'LINGBOT_ROOT':str(r),'CONDA_EXE':os.environ['CONDA_EXE'],
 'LINGBOT_ENV':str(r/'envs/lingbotvla'),'ROBOTWIN_ENV':str(r/'envs/robotwin'),
 'HF_HOME':str(r/'cache/huggingface'),'XDG_CACHE_HOME':str(r/'cache/xdg'),
 'PIP_CACHE_DIR':str(r/'cache/pip'),'CONDA_PKGS_DIRS':str(r/'cache/conda'),
 'TORCH_HOME':str(r/'cache/torch'),'TORCH_EXTENSIONS_DIR':str(r/'cache/torch_extensions'),
 'TMPDIR':str(r/'cache/tmp'),'PYTHONNOUSERSITE':'1','PIP_NO_INPUT':'1',
 'MAX_JOBS':'2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
 'LINGBOT_STORAGE_MOUNT':os.environ['LINGBOT_PERSIST'],
 'LINGBOT_STORAGE_UUID':os.environ['LINGBOT_STORAGE_UUID']}
p=r/'configs/paths.env'
assert not p.exists(), '已有paths.env，先对比人工更新，不能覆盖'
p.write_text(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in v.items()))
PY
source "$LINGBOT_ROOT/configs/paths.env"
source "$LINGBOT_ROOT/scripts/common.sh"
~~~

本示例不改 HOME；所有下载/编译缓存均进项目持久化路径。若平台持久化不是独立有UUID的盘，先确认目录，再明确调整 guard 的存储适配方式，不能单纯禁用 guard。

### 没有 Conda 时的固定安装器

从 [Anaconda 官方索引](https://repo.anaconda.com/miniconda/) 核实的 x86_64 安装器，不覆盖现有 Conda：

~~~bash
installer=Miniconda3-py312_25.7.0-2-Linux-x86_64.sh
curl -fL --retry 3 -C - -o "$installer" "https://repo.anaconda.com/miniconda/$installer"
printf '%s  %s\n' 188b5d94ab3acefdeaebd7cb470d2fb74a3280563c77075de6e3e1d58d84ab0a "$installer" | sha256sum -c -
[[ ! -e /root/miniconda3-lingbot ]]
bash "$installer" -b -p /root/miniconda3-lingbot
export CONDA_EXE=/root/miniconda3-lingbot/bin/conda
~~~

## 5. 重建训练环境（训练不要求先完成仿真扩展）

先安装普通工具。Ubuntu24镜像下可执行；非root请用平台允许的权限方式：

~~~bash
apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates rsync ffmpeg cmake ninja-build pkg-config build-essential libglib2.0-0 libgl1
cd "$LINGBOT_ROOT"
bash scripts/setup_training_from_handoff.sh
~~~

这个新交接脚本已做语法检查，现场全新重建尚未执行。它重建固定source/Conda prefix、安装cu128 Torch、按当前机器精确普通包版本 --no-deps安装，然后单独安装官方LeRobot tag、LingBot/Depth/MoGe editable、固定utils3d与匹配flash wheel。所有安装失败直接退出，未知pip check冲突也退出。

Flash wheel SHA256：f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb。必须检查Python3.12、Torch2.8、CUDA12、CXX11 ABI TRUE，不在无GPU/未知架构上盲编译。

重要兼容处理：

- 原官方配方 metadata 互相冲突；保留授权的 --no-deps 运行版本，见 docs/dependency_conflicts.md 与 configs/known_lingbot_pip_check.txt，不伪造pip check全绿。新未知错误不得吞掉。
- 官方安装脚本的 utils3d .pth 指向这个commit里不存在的 morgbd_clean/3rd/utils3d；使用MoGe自身声明的固定utils3d来源，未改模型。
- 训练使用Py3.12的lingbotvla，仿真是Py3.10的robotwin，不混装。当前旧robotwin也已迁到Torch2.8/cu128，旧Torch2.4.1/cu121仅备份。
- 不直接执行旧 setup_gpu.sh：它仍断言robotwin2.4.1/cu121；旧setup_gpu_cu128_v4.sh有固定8.0/等待旧日志等现场逻辑，也不是通用新机安装器。
- 不要为 registry 的可选PI0策略提示 No module named lerobot.common 而降级LeRobot；这里实际训练的是LingBot-v2，转换器已适配v0.4.2。

激活方式：

~~~bash
source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate "$LINGBOT_ENV"
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
~~~

## 6. 模型/raw 迁移优先，缺少时限定下载

最快路径是在新机拉旧机现成文件；认证由用户交互完成。不用 --delete，不迁移旧env二进制或整个cache，不复制凭证。

~~~bash
rsync -a --partial --info=progress2 -e 'ssh -p 28548' root@223.109.239.11:/root/lingbot_challenge/models/ "$LINGBOT_ROOT/models/"
rsync -a --partial --info=progress2 -e 'ssh -p 28548' root@223.109.239.11:/mnt/lingbot-data/lingbot_challenge/data/clean_raw/ "$LINGBOT_ROOT/data/clean_raw/"
rsync -a --partial --info=progress2 -e 'ssh -p 28548' root@223.109.239.11:/mnt/lingbot-data/lingbot_challenge/data/clean_processed/ "$LINGBOT_ROOT/data/clean_processed/"
~~~

复制后用限定下载器逐文件验证模型，已有正确文件不重复下载：

~~~bash
bash "$LINGBOT_ROOT/scripts/download_assets.sh" base --workers 2 --attempts 12 --min-rate-kib 32
bash "$LINGBOT_ROOT/scripts/download_assets.sh" qwen --workers 2 --attempts 12 --min-rate-kib 32
bash "$LINGBOT_ROOT/scripts/download_assets.sh" moge --workers 2 --attempts 12 --min-rate-kib 32
~~~

如果无法取旧机raw，则下载只选manifest中的50个clean ZIP，再解压成seen-only manifest：

~~~bash
bash "$LINGBOT_ROOT/scripts/download_assets.sh" clean --workers 4 --attempts 12 --min-rate-kib 32
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/prepare_clean.py"
~~~

如果已经复制完整extracted raw和原manifest，不必再次解压。convert_clean会逐任务核对50个原文件hash、seen指令hash/来源。不要用 snapshot_download 整个 RoboTwin2.0，也不要单独下载unseen指令池。

## 7. 继续另外48个clean任务转换，然后计算正式全量norm

先生成指向当前根目录的50任务train list和robot mapping。这份官方模板保留32GPU大batch参照，**不是可直接launch的本机训练配置**。

~~~bash
source "$LINGBOT_ROOT/configs/paths.env"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/build_clean_config.py"
bash "$LINGBOT_ROOT/scripts/resume_clean_conversion.sh"
~~~

可用nohup运行同一脚本并看 results/clean_handoff_status.json，不能重复启动：

~~~bash
nohup bash "$LINGBOT_ROOT/scripts/resume_clean_conversion.sh" > "$LINGBOT_ROOT/logs/clean-handoff-dispatch.log" 2>&1 < /dev/null &
tail -f "$LINGBOT_ROOT/logs/clean-handoff-dispatch.log"
~~~

上面前台/后台命令二选一，先检查同类进程。还在运行时不要把长期旧状态 READY/COMPLETE 当当前完成。

流程行为：

1. 全部50任务依次处理；已有两个输出的provenance/hash匹配时跳过转换，但重新验收。
2. 使用固定RoboTwin离线process_data.py及生成的v0.4.2 adapter；保持next-frame action、3×640×480视频、50FPS、每任务50条、seen instructions。没有仿真API。
3. 输出至data/clean_processed/lerobot/TASK，HDF5中间产物在同卷，逐任务空间guard raw_bytes×6 +20GiB；不删原文件。
4. 全行state/action与raw精确一致、episode0–49、指令池来源正确；native loader读取任务首/末帧的三路视频，不虚称完整视频每帧都验过。
5. 全部完成后运行official compute_norm_stats的import-only CPU适配器，写configs/clean_norm/robotwin.json及provenance。只读允许的50任务train list，不跑torchrun/模型/仿真。
6. audit_training_ready.py通过后写results/data_train_ready_50.json，明确DATA_TRAIN_READY_50。

不要直接重跑旧run_cpu_continuation.sh/finish_cpu_continuation.py：它含旧PID27033等待逻辑，本交接的resume流程不依赖旧PID。

如果失败任务已产生目录且没有完成provenance，转换器会拒绝覆盖。先查看日志/空间，保留失败目录，确认后移至带时间戳的隔离目录再重试；不要直接rm -rf。不要同时运行多个原转换器，它们共享generated adapter路径。

baseline_clean_norm可保留做原两任务smoke的来源证据，不能冒充新完成的50任务LeRobot官方统计。默认正式配置用新计算的clean_norm；不能用仓库自带、来源可能含randomized的robotwin.json。

## 8. GPU验收与50任务离线验收

~~~bash
bash "$LINGBOT_ROOT/scripts/check_training_gpu.sh"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/audit_training_ready.py"
~~~

GPU检查只做每卡bf16矩阵运算、flash-attn小算子和主模型模块import，不加载完整6B到CPU，不运行simulator。

旧2×A100的NVLink inactive、topo PHB，NCCL via SHM，AG约12.48GB/s，RS约12.43GB/s。新机重新测拓扑；SXM型号不保证NVLink。不能自行安装Fabric Manager修复，找平台确认。

至此可以报告“50任务data ready、训练环境/GPU算子通过”，但仍不能说正式checkpoint训练/回读已验收。

## 9. 训练模式、完整配置与正式保存入口

先明确模式。下面默认复制已经测过的冻结VLM方案；若用户明确全量训练，使用full配置分支，不把冻结方案偷偷当成全量。

### 冻结VLM：保留原白名单，保留正式checkpoint逻辑

~~~bash
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/prepare_frozen_training_entry.py"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/make_training_configs.py" --mode frozen --world-size 2 --micro-batch 8 --global-batch 32 --steps 10000 --save-steps 1000 | tee "$LINGBOT_ROOT/results/selected_training_configs.json"
~~~

2卡时mb8/GA2/gbs32，compile=true、bf16、FSDP2 full shard、align/future image均保留。按实际卡数与显存修改mb；gbs = mb × data_parallel_size × GA必须整除，OOM时可退mb4/GA4，不改模型或静默删依赖。

prepare_frozen_training_entry只在scripts生成固定官方正式训练入口的副本，并插入**字节一致的原benchmark冻结白名单**，不改repo/model/loader/FSDP。whitelist SHA256为f3fcdfafa4bccb52b5ac8f9964834c8561515e96f5b4c89e9b4a99d1441d0094。官方保存、末步保存、hf_saver.wait_all_across_ranks保留；patch/provenance输出可审计。该生成器已在隔离副本静态试跑，**新GPU上的正式save训练未跑**。

不要把train_benchmark.py或train_diag.py作为正式训练入口：它们到max_steps直接return，末步不保存。仅YAML的train_expert_only也不等价原benchmark全部冻结白名单。

### 全量VLM+expert：未做容量保证的起始验证配置

~~~bash
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/make_training_configs.py" --mode full --world-size 2 --micro-batch 1 --global-batch 32 --steps 10000 --save-steps 1000 | tee "$LINGBOT_ROOT/results/selected_training_configs.json"
~~~

full使用官方tasks/vla/train_lingbotvla.py，不插冻结白名单；train_expert_only=false、freeze_vision_encoder=false、全局gradient checkpointing=true，align保留。这是需要GPU smoke验证的起点，不代表2×40GB肯定放得下，也不代表推荐新训练技巧。全量optimizer/grad内存显著增加，按现场卡数/显存排障；不能用冻结测试的显存或step time当全量结论。

配置生成器输出两个**完整YAML**：正式预算配置和末尾_smoke.yaml的11步保存验收配置。10000是示例训练预算，不是承诺一个epoch；完整数据的实际步数由loader决定。两份都eval.enabled=false、command=null、enable_resume=false、新output_dir、canonical clean_norm、全50任务train list，不会自动启动评测或接着旧run训练。

### 11步checkpoint smoke（用户安排训练模式后执行）

将CONFIG_SMOKE设为刚生成的完整_smoke.yaml路径，不是泛指旧两任务benchmark config：

~~~bash
export CONFIG_SMOKE="$("$LINGBOT_ENV/bin/python" -c 'import os,json; print(json.load(open(os.environ["LINGBOT_ROOT"]+"/results/selected_training_configs.json"))["checkpoint_smoke_config"])')"
bash "$LINGBOT_ROOT/scripts/launch_clean_training.sh" "$CONFIG_SMOKE"
~~~

这是11步真实优化和保存测试，会占GPU并写完整checkpoint；本手册生成时未在当前无GPU旧机运行。启动器会先要求DATA_TRAIN_READY_50、gbs等式、允许base、eval禁用、新空输出目录和checkpoint卷>60GiB空余；不满足就退出。不要绕过空间门槛。

必须核查：

- 11 steps正常结束、action loss有限、参数冻结范围与所选模式一致；两个rank无异常，不用grep到“Step11”替代成功退出码。
- checkpoint与global_step、optimizer/scheduler/RNG/dataloader状态实际保存；结束前等待HF导出完成，不能只看到目录名。
- 导出的hf_ckpt权重index列出的safetensors均存在、大小非零，允许base的来源/训练数据manifest和norm provenance记录保留。
- 用同入口、同模式、同卡数和同mb/GA做一次独立输出目录的恢复测试，明确验证step编号继续增长、optimizer/dataloader/scheduler恢复。用train.load_checkpoint_path指定**具体step目录：旧smoke的 output_dir/checkpoints/global_step_11**，不是把hf_ckpt当optimizer checkpoint；enable_resume=true/allow_partial_checkpoint=false。原入口显式load_checkpoint_path直接加载该目录；enable_resume未给显式路径才扫描当前output_dir/checkpoints。它捕获加载错误后可能从头训练，必须核对明确的Load distributed checkpoint成功日志，不能只按退出码当恢复通过。
- save/restore失败就继续排障并保留日志；只有通过后才报告TRAIN_READY_50。smoke结果不是收敛或比赛分数验收。


保存后做结构检查，再生成只追加1步的独立resume测试配置（用户安排后执行）：

~~~bash
export WORLD_SIZE=2  # 按此次同一data-parallel卡数修改
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/verify_saved_checkpoint.py" "$CONFIG_SMOKE"
export CONFIG_RESUME="$("$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/prepare_resume_smoke_config.py" "$CONFIG_SMOKE")"
bash "$LINGBOT_ROOT/scripts/launch_clean_training.sh" "$CONFIG_RESUME"
"$LINGBOT_ENV/bin/python" "$LINGBOT_ROOT/scripts/verify_saved_checkpoint.py" "$CONFIG_RESUME"
~~~

还必须读resume日志确认Load distributed checkpoint成功、从step11到12，不能出现Starting training from scratch/Failed to load checkpoint。结构检查只load可信本地extra_state和safetensors header，不把6B模型搬CPU；它不替代GPU恢复执行。原入口保存/恢复的是CPU Torch RNG，未保证CUDA RNG/所有非Torch随机源的bitwise续跑，需如实记录，不声称严格逐位重放。

正式长训练只在用户安排后使用正式YAML，同一50任务混训一份权重。保存间隔和训练预算由用户确认计划后填写；不要自动开始正式训练。

## 10. 可选：另建RoboTwin评测环境

这一步不阻塞LingBot训练，且不运行评测/专家采集。若本次只要今晚训练，可后置。

- 包内robotwin.conda-explicit/portable-pins记录的是Py3.10+Torch2.8/cu128、numpy1.26.4、sapien3.0.0b1、mplib0.2.1、setuptools69.5.1的实际栈；不再套旧cu121 setup_gpu.sh。
- cuRobo固定d64c4b005459db10c5dd867d8b30a87d5bda9bdb，PyTorch3D固定75ebeeaea0908c5527e7b1e305fbc7681382db47。
- 新env隔离安装Torchcu128、robotwin.portable-pins与源包；用独立cuda128-build的CUDA12.8 + gcc/gxx12，CUDA_HOME/CUDACXX/CC/CXX指向它。设置CPATH为targets/x86_64-linux/include、LIBRARY_PATH为对应lib；不要改宿主CUDA或driver。
- TORCH_CUDA_ARCH_LIST从新机实际torch.cuda.get_device_capability逐卡获取，不固定8.0。MAX_JOBS默认2，内存不足降1，不盲目反复编译。
- 最小构建命令是robotwin env里的pip --no-deps --no-build-isolation -e curobo，以及FORCE_CUDA=1 pip --no-deps --no-build-isolation 本地pytorch3d。编译完成用包内check_gpu.sh检查Warp/cuRobo/PyTorch3D算子，不创建仿真环境。
- simulation下载器仅simulation scope准备manifest中的三个官方资源ZIP，并用prepare_sim_assets.py逐archive离线解压；训练不读取sim资产。
- 编译和加载未在新机验证时明确SIM_GPU_PENDING，不要把“能训练”写成“RoboTwin闭环评测已跑通”。

## 11. 新agent交付与失败处理

交付docs/setup_report_NEWDATE.md、实际conda/pip exports、source revision/diff、模型来源与hash、DATA_TRAIN_READY_50记录、GPU小算子日志、选定完整YAML、checkpoint save/restore日志、准确激活/正式启动命令。长任务记录状态、pid、log，不静默后台丢失。

常见阻塞：

| 现象 | 处理 |
|---|---|
| UUID不匹配/未挂载 | 确认新持久化位置、实际UUID和symlink；不关闭guard |
| .pth/editable找旧路径 | 用新固定checkout重新editable安装，避免复制旧env二进制 |
| pip check已知声明冲突 | 完整记录为KNOWN_METADATA_CONFLICT；未知差异必须解决 |
| LeRobot文件名不是episode_*.parquet | 实际v3.0是chunk/file parquet与mp4；按info.json/native loader读取，不重转成错误布局 |
| 48目录缺失 | 完成resume_clean_conversion，不用50行txt冒充50个实际dataset |
| partial输出 | 保留、检查、隔离后重试，不自动删除 |
| CUDA/flash失败 | 检查platform driver、Torch CUDA、Python/ABI/架构；不装宿主驱动或改attention模型绕过 |
| full训练OOM | frozen测试不适用；缩microbatch/重新评估容量，不私自冻结VLM或关align |
| 没有正式checkpoint | 检查误用了timing benchmark；正式副本保留原save逻辑 |

不要再以“50份ZIP齐全”“manifest2500齐全”“baseline norm有文件”任一单项称train-ready。只有全50数据实际可读、正式norm来源一致、GPU小检查与所选入口save/restore验证都通过，才能交付能训练状态。

本次交接构建已做Bash/Python静态检查、冻结入口/配置生成器隔离试跑；没有重新安装旧机环境、没有继续转换48任务、没有运行GPU/训练/仿真。原机当前GPU已释放，nvidia-smi不可用，不能用这台无GPU旧机替代新卡验收。
