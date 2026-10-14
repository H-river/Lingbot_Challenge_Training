# LingBot 训练部署说明（8×A100）

写给帮忙跑训练的队友。谢谢你！

**你需要动手的时间大约 1 小时**（不含下载）。之后机器会自己跑，你只需要偶尔看一眼。

整个流程就是：装好 Docker → 导入镜像 → 解压权重和数据 → 运行 3 条命令。训练代码、依赖环境、比赛合规检查都已经封装在镜像里，**不需要读代码，也不需要改任何配置**。遇到问题先看文末的「常见问题」，解决不了就把报错发给 Hu。

---

## 你会收到什么

Google Drive 文件夹里有这些文件：

| 文件 | 是什么 |
|---|---|
| `lingbot-train_deploy-*.tar.gz` | Docker 镜像：训练环境 + 代码 |
| `weights.tar` | 模型权重（约 29 GB） |
| `data_clean.tar` | 训练数据（比赛官方 clean 数据，已经转换好） |
| `run.sh` | 你唯一需要用的脚本 |
| `DEPLOY.md` | 就是本文 |
| `SHA256SUMS` | 校验文件，用来确认下载完整 |

**磁盘要求**：部署目录所在的盘至少留 **700 GB** 空闲。其中权重 + 数据约 100 GB，训练存档 400 GB 以上，下载的压缩包解压完可以删。另外，Docker 自己的目录（一般是 `/var/lib/docker`）还要留约 40 GB。

---

## 第 0 步：确认机器环境（5 分钟）

```bash
nvidia-smi          # 应该看到 8 张 A100，右上角 Driver Version ≥ 570
docker --version    # 需要 ≥ 23
df -h               # 看看哪块盘空间够，作为部署目录
```

- 驱动低于 570：请找服务器管理员升级。**不要自己装驱动**。
- 没有 Docker，或者第 2 步出现 `could not select device driver` 报错：需要安装 Docker 和 NVIDIA Container Toolkit。官方安装说明：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

如果 `docker` 命令提示 permission denied，下文所有命令前面加 `sudo` 就行。

## 第 1 步：建部署目录，放好文件

部署目录放在哪都行，下面以 `/data/lingbot` 为例：

```bash
mkdir -p /data/lingbot && cd /data/lingbot
# 把 Drive 上的所有文件下载到这里，然后校验：
sha256sum -c SHA256SUMS            # 每一行都应该是 OK
chmod +x run.sh
```

## 第 2 步：导入镜像（10–20 分钟）

```bash
docker load -i lingbot-train_deploy-*.tar.gz
docker image ls lingbot-train      # 应该看到 lingbot-train:deploy
docker run --rm --gpus all --entrypoint nvidia-smi lingbot-train:deploy   # 容器里能看到 8 张卡就对了
```

## 第 3 步：解压权重和数据

```bash
cd /data/lingbot
tar -xf weights.tar
tar -xf data_clean.tar
# 确认没问题后，可以删掉 3 个压缩包省空间
```

解压完，目录应该是这样：

```text
/data/lingbot/
├── run.sh
├── DEPLOY.md
├── models/
│   ├── lingbot-vla-v2-6b/
│   ├── Qwen3-VL-4B-Instruct-config/
│   └── MoGe-2-vitb-normal/
└── data/
    ├── clean_processed/lerobot/   (50 个任务)
    └── clean_raw/extracted/       (50 个任务)
```

`checkpoints/`、`logs/`、`results/`、`cache/` 这几个目录会在第一次运行时自动创建。

## 第 4 步：检查 `./run.sh check`（3–5 分钟）

```bash
cd /data/lingbot
./run.sh check
```

会依次检查：GPU 和驱动、GPU 计算、8 卡通信、权重、数据、磁盘、内存，以及比赛合规。

最后一行显示 `CHECK PASSED` 就可以进入下一步。中间的黄色 `WARN` 不影响继续，但请把这些行发给 Hu 看一下。出现红色 `FAIL` 的话，查文末的「常见问题」。

想把权重逐字节校验一遍可以运行 `./run.sh check --full`，会多花几分钟。

## 第 5 步：冒烟测试 `./run.sh smoke`（30–60 分钟）

```bash
./run.sh smoke
```

它会用正式训练的 batch 设置，在 8 张卡上做两件事：先训练 10 步并存档，再从这个存档续训 3 步并存档。这样能确认训练、存档、续训三个环节都正常，同时测出显存占用和每步耗时。

- 任务在**后台**运行，关掉终端或断开 SSH 都不影响。
- 按 `Ctrl-C` 只是停止查看日志，任务本身会继续跑。想重新看日志：`./run.sh logs`
- 跑完会打印一个 `SMOKE RESULT` 方框。**请把整个方框复制或截图发给 Hu**。Hu 会根据里面的时间估算告诉你正式训练跑多少步（`MAX_STEPS`），有时还会让你加一两个参数。
- 冒烟产生的存档（约 100 GB）测试通过后会自动删掉，日志会保留。

## 第 6 步：正式训练

用 Hu 发给你的那一行命令，例如：

```bash
MAX_STEPS=3000 ./run.sh train
# Hu 可能会给你带更多参数的版本，例如：
MBS=8 PRECISION=bf16 MAX_STEPS=3000 ./run.sh train
```

同样在后台运行。之后每天看一两次就行：

```bash
./run.sh status     # 有没有在跑、跑到第几步、GPU 占用
./run.sh logs       # 实时看日志（Ctrl-C 退出查看）
```

---

## 训练中你会看到什么

正常的日志每一步一行，大概长这样：

```text
Step 1250/537, Epoch 3, Loss 0.21, VLA_Loss 0.19, ... StepTime 41.3s ...
```

- `Step` 后面斜杠右边的数字是**每个 epoch 的步数**，不是总步数，这是正常的。总步数就是你设的 `MAX_STEPS`。
- `Loss` 整体应该缓慢下降，单步上下跳动是正常的。如果出现 `nan`，或者长时间没有新的一行，请告诉 Hu。
- 每隔 `SAVE_STEPS` 步（默认 1000）会存一次档，位置是 `checkpoints/<运行名>/checkpoints/global_step_N/`，每个约 49 GB。

**中途断了怎么办？**（断电、报错、被 stop、机器重启都算）

不用慌，从最近一次存档接着跑：

```bash
ls /data/lingbot/checkpoints/*/checkpoints/            # 找到编号最大的 global_step_N
./run.sh resume checkpoints/<运行名>/checkpoints/global_step_N
```

续训会自动沿用原来那次运行的全部设置，不需要再写 `MAX_STEPS`。

## 跑完之后

日志里出现 `TRAINING FINISHED` 后，告诉 Hu 就行。**什么都不要删**。Hu 需要的是：

- `checkpoints/<运行名>/`：模型存档
- `checkpoints/<运行名>.config/`：这次运行的实际配置（比赛复现要用）
- `logs/<运行名>.log`
- `results/`

之后的评测也会在这台机器上跑，Hu 会另外发说明。

---

## 请不要做的事

这些和比赛规则有关，违反了可能导致成绩作废：

1. **不要往 `models/` 里放其他权重**，尤其是名字里带 `robotwin` 的。比赛禁止使用这类权重，`check` 发现后会直接报错。
2. **不要改 `data/` 里的任何文件，也不要换成别的数据。** 数据目录是以只读方式挂进容器的，启动时也会做审计，但请不要尝试。
3. **不要绕开 `run.sh`**，自己去调用 `train.sh`、`torchrun`，或者用仓库里官方的 `robotwin.yaml`。官方那份配置会读 randomized 数据，在比赛里是违规的。
4. **Hu 没说之前，不要删除 `checkpoints/` 里的任何东西。**
5. 这些文件请不要发给队伍以外的人。

---

## 常见问题

| 你看到的 | 原因 | 怎么办 |
|---|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | 没装 NVIDIA Container Toolkit | 按第 0 步的链接安装，然后执行 `sudo systemctl restart docker` |
| `permission denied ... docker.sock` | 当前用户没有 docker 权限 | 命令前加 `sudo` |
| `Image lingbot-train:deploy not found` | 镜像还没导入 | 做第 2 步 |
| check：`driver ... too old` | 驱动低于 570 | 找管理员升级，不要自己装 |
| check：weights `missing` / `size mismatch` | 权重没下完整或没解压对 | `sha256sum -c SHA256SUMS`，然后重新解压 `weights.tar` |
| check：`data audit failed` | 数据没解压完整，或者目录位置不对 | 对照第 3 步的目录结构，重新解压 `data_clean.tar` |
| check：`NCCL ... failed or hung` | 有别的程序占着卡，或者卡之间的通信有问题 | 用 `nvidia-smi` 看是不是有人在用；把 `logs/check_nccl_*.log` 发给 Hu |
| check：`forbidden robotwin ... weights` | `models/` 里混进了禁用权重 | 把那个文件夹移出 `models/` |
| smoke 报 `CUDA out of memory` | 显存不够 | 运行 `MBS=2 ./run.sh smoke`，并告诉 Hu |
| 只能用其中几张卡（别人也在用） | — | `GPUS=0,1,2,3 ./run.sh smoke`。如果提示 `not divisible`，把提示里给的 GBS 选项发给 Hu |
| `already running` | 已经有一个任务在跑 | `./run.sh status` 看一下；确实要停就 `./run.sh stop` |
| `Output exists` | 运行名重复了 | 换个名字：`RUN_NAME=train_v2 MAX_STEPS=... ./run.sh train`。旧的不要删 |
| `CHECKPOINT_SPACE_PENDING` / 磁盘不足 | 存档盘空间不够 | 清理别的文件，或者问 Hu 哪些旧存档可以删 |
| GPU 利用率一直很低，训练很慢 | CPU 读数据跟不上 | 告诉 Hu，可能要设 `NUM_WORKERS=2` 或 `6` |
| 其他报错 | — | 把 `logs/` 里最新那个日志的最后 200 行发给 Hu：`tail -200 logs/<文件名>` |

联系人：Hu Jiamo（联系方式：__________）

---

<details>
<summary>给好奇的人：镜像里有什么</summary>

- 环境：Ubuntu 24.04、Python 3.12.15、PyTorch 2.8.0+cu128、flash-attn 2.8.3。版本冻结自 2026-10-14 A100 冒烟测试通过的那台机器，见 `docs/versions/`。
- 代码：官方 `Robbyant/lingbot-vla-v2` commit `be969b8`，未做任何改动，构建时会逐文件校验哈希；外加本仓库的 clean-only 数据审计和启动脚本。
- 训练配置：`configs/train_8xA100.yaml`。和官方配方相比只改了 batch、显存和存档相关的设置，文件头逐项列出了这些差异。每次运行的实际配置和覆盖项都会存进 `checkpoints/<运行名>.config/`。
- 镜像构建方法：`bash docker/build.sh --save <输出目录>`
</details>
