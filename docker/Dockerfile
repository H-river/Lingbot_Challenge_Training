# LingBot-VLA v2 clean-only training image (environment + code; NO data, NO weights).
# Build with docker/build.sh: it fetches pinned installers into docker/vendor/ and passes them
# as the extra build context "vendor" (bind-mounted, never copied into the image).
# Needs BuildKit (Docker >= 23). Do not set DOCKER_BUILDKIT=0.
#
# Mirrors (all optional, for mainland-China build hosts):
#   --build-arg BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.8.1-base-ubuntu24.04
#   --build-arg APT_MIRROR=mirrors.aliyun.com
#   --build-arg CONDA_FORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
#   --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
# Package versions are pinned by docs/versions/* (frozen from the machine where the
# 2026-10-14 A100 smoke passed) and must not be changed here.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
FROM ${BASE_IMAGE}

ARG APT_MIRROR=
ARG CONDA_FORGE_URL=https://conda.anaconda.org/conda-forge
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG LINGBOT_GIT_COMMIT=unknown

ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai LANG=C.UTF-8 \
    LINGBOT_ROOT=/root/lingbot_challenge \
    LINGBOT_ENV=/root/lingbot_challenge/envs/lingbotvla \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- system packages (ffmpeg: torchcodec video decoding; build-essential: torch.compile/triton)
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
 && if [[ -n "$APT_MIRROR" ]]; then \
      sed -i "s#http://archive.ubuntu.com#http://$APT_MIRROR#g; s#http://security.ubuntu.com#http://$APT_MIRROR#g" \
        /etc/apt/sources.list.d/ubuntu.sources; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git build-essential ffmpeg \
      libgl1 libglib2.0-0t64 tini procps less tzdata \
 && rm -rf /var/lib/apt/lists/*

# --- conda base + exact Python 3.12.15 env from the frozen explicit spec
COPY docs/versions/lingbotvla.conda-explicit.txt /tmp/pins/
RUN --mount=type=bind,from=vendor,target=/tmp/vendor \
    bash /tmp/vendor/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
 && sed "s#https://conda.anaconda.org/conda-forge#${CONDA_FORGE_URL%/}#" /tmp/pins/lingbotvla.conda-explicit.txt > /tmp/pins/explicit.txt \
 && /opt/conda/bin/conda create -y -p "$LINGBOT_ENV" --file /tmp/pins/explicit.txt \
 && /opt/conda/bin/conda clean -afy

# --- pip stack, same order and pins as scripts/setup_training_from_handoff.sh
COPY docs/versions/lingbotvla.portable-pins.txt configs/flash-attn-wheel.json /tmp/pins/
RUN --mount=type=bind,from=vendor,target=/tmp/vendor \
    PY="$LINGBOT_ENV/bin/python" \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1' \
 && "$PY" -m pip install --index-url "$TORCH_INDEX_URL" 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps -r /tmp/pins/lingbotvla.portable-pins.txt \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps /tmp/vendor/lerobot-v0.4.2.tar.gz \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps /tmp/vendor/utils3d-3fab839f0be9931dac7c8488eb0e1600c236e183.tar.gz \
 && wheel=$("$PY" -c 'import json;print(json.load(open("/tmp/pins/flash-attn-wheel.json"))["name"])') \
 && digest=$("$PY" -c 'import json;print(json.load(open("/tmp/pins/flash-attn-wheel.json"))["digest"].split(":")[1])') \
 && echo "$digest  /tmp/vendor/$wheel" | sha256sum --check \
 && "$PY" -m pip install --no-deps "/tmp/vendor/$wheel"

# --- pinned upstream source snapshot (verified byte-for-byte), editable installs
WORKDIR /root/lingbot_challenge
COPY repos/ repos/
COPY docs/upstream_source_sha256.json docs/
COPY docker/verify_source.py docker/
RUN "$LINGBOT_ENV/bin/python" docker/verify_source.py
RUN PY="$LINGBOT_ENV/bin/python"; repo=repos/lingbot-vla-v2 \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps -e "$repo" \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps -e "$repo/lingbotvla/models/vla/vision_models/lingbot-depth" \
 && "$PY" -m pip install --index-url "$PIP_INDEX_URL" --no-deps -e "$repo/lingbotvla/models/vla/vision_models/MoGe"

# --- the rest of this repository (scripts, configs, manifests, evidence, docs)
COPY . .
RUN printf '%s\n' \
      'export LINGBOT_ROOT=/root/lingbot_challenge' \
      'export LINGBOT_ENV=$LINGBOT_ROOT/envs/lingbotvla' \
      '# RoboTwin (simulation/eval) is not part of this training image; the variable is only required by common.sh.' \
      'export ROBOTWIN_ENV=$LINGBOT_ROOT/envs/robotwin-not-in-training-image' \
      'export HF_HOME=$LINGBOT_ROOT/cache/huggingface XDG_CACHE_HOME=$LINGBOT_ROOT/cache/xdg' \
      'export TORCH_HOME=$LINGBOT_ROOT/cache/torch TORCH_EXTENSIONS_DIR=$LINGBOT_ROOT/cache/torch_extensions' \
      'export TMPDIR=$LINGBOT_ROOT/cache/tmp PYTHONNOUSERSITE=1 PIP_NO_INPUT=1' \
      'export LINGBOT_STORAGE_MOUNT= LINGBOT_STORAGE_UUID=' \
      > configs/paths.env \
 && echo "$LINGBOT_GIT_COMMIT" > /root/lingbot_challenge/IMAGE_COMMIT \
 && chmod +x docker/*.sh docker/*.py train.sh scripts/*.sh

# --- build-time verification (CPU only; GPU checks run in ./run.sh check)
RUN "$LINGBOT_ENV/bin/python" docker/verify_build.py

ENV LINGBOT_GIT_COMMIT=${LINGBOT_GIT_COMMIT}
ENTRYPOINT ["/usr/bin/tini", "--", "/root/lingbot_challenge/docker/entrypoint.sh"]
CMD ["help"]
