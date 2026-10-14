#!/usr/bin/env bash
# Build the clean-only training image and (optionally) export it as a handoff tarball.
#
#   bash docker/build.sh                      # build lingbot-train:deploy
#   bash docker/build.sh --save /path/outdir  # build + write lingbot-train_<tag>.tar.gz + .sha256
#
# Optional environment (useful on mainland-China hosts):
#   GH_PROXY=https://ghfast.top/              prefix for GitHub downloads
#   MINIFORGE_URL=...                         Miniforge installer URL (default: GitHub latest release)
#   LINGBOT_WHEEL_CACHE=/root/lingbot_challenge/cache/wheels   reuse an already-downloaded flash-attn wheel
#   BASE_IMAGE APT_MIRROR CONDA_FORGE_URL PIP_INDEX_URL TORCH_INDEX_URL   -> passed as --build-arg (see Dockerfile)
set -Eeuo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
V="$REPO/docker/vendor"
SAVE_DIR=""
[[ "${1:-}" == "--save" ]] && SAVE_DIR="${2:?--save needs an output directory}"
mkdir -p "$V"

command -v docker >/dev/null || { echo "docker not found"; exit 2; }
docker buildx version >/dev/null 2>&1 || { echo "docker buildx (BuildKit) is required: Docker >= 23 or the buildx plugin"; exit 2; }

# fetch URL OUT [SHA256]  - download once into docker/vendor, verify pinned hash when known
fetch() {
  local url=$1 out="$V/$2" sha=${3:-}
  if [[ ! -s "$out" ]]; then
    echo "download $2"
    curl --fail --location --retry 5 --retry-delay 5 --connect-timeout 20 -o "$out.partial" "$url"
    mv "$out.partial" "$out"
  fi
  if [[ -n "$sha" ]]; then
    echo "$sha  $out" | sha256sum --check --quiet || { echo "SHA256 MISMATCH: $out (delete it and retry)"; exit 2; }
  fi
}

GH=${GH_PROXY:-}
fetch "${MINIFORGE_URL:-${GH}https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}" \
  Miniforge3-Linux-x86_64.sh
fetch "${GH}https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz" lerobot-v0.4.2.tar.gz \
  2ed4981213f0931e741df1dfc3986a36af2fe44db8540d723108c6fd2defc14b
fetch "${GH}https://github.com/EasternJournalist/utils3d/archive/3fab839f0be9931dac7c8488eb0e1600c236e183.tar.gz" \
  utils3d-3fab839f0be9931dac7c8488eb0e1600c236e183.tar.gz

read -r WHEEL WHEEL_URL WHEEL_SHA < <(python3 - "$REPO/configs/flash-attn-wheel.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));print(m['name'],m['browser_download_url'],m['digest'].split(':')[1])
PY
)
if [[ ! -s "$V/$WHEEL" && -n "${LINGBOT_WHEEL_CACHE:-}" && -s "$LINGBOT_WHEEL_CACHE/$WHEEL" ]]; then
  cp "$LINGBOT_WHEEL_CACHE/$WHEEL" "$V/$WHEEL"
fi
fetch "${GH}${WHEEL_URL}" "$WHEEL" "$WHEEL_SHA"
(cd "$V" && sha256sum -- * | grep -v SHA256SUMS > SHA256SUMS)
echo "vendor files:"; cat "$V/SHA256SUMS"

COMMIT=unknown
if git -C "$REPO" rev-parse HEAD >/dev/null 2>&1; then
  COMMIT=$(git -C "$REPO" rev-parse HEAD)
  if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
    COMMIT="${COMMIT}-dirty"
    echo "WARNING: building from uncommitted changes; the image will be labelled ${COMMIT}"
  fi
fi
TAG="deploy-$(date -u +%Y%m%d)-${COMMIT:0:7}"

args=(--build-arg "LINGBOT_GIT_COMMIT=$COMMIT")
for k in BASE_IMAGE APT_MIRROR CONDA_FORGE_URL PIP_INDEX_URL TORCH_INDEX_URL; do
  [[ -n "${!k:-}" ]] && args+=(--build-arg "$k=${!k}")
done

docker buildx build --load -f "$REPO/docker/Dockerfile" --build-context "vendor=$V" "${args[@]}" \
  -t "lingbot-train:$TAG" -t lingbot-train:deploy "$REPO"
docker image ls lingbot-train
echo "BUILT lingbot-train:$TAG (also tagged lingbot-train:deploy), repo commit $COMMIT"

if [[ -n "$SAVE_DIR" ]]; then
  mkdir -p "$SAVE_DIR"
  out="$SAVE_DIR/lingbot-train_${TAG}.tar.gz"
  zip=gzip; command -v pigz >/dev/null && zip=pigz
  echo "exporting to $out (this takes a while) ..."
  docker save lingbot-train:deploy "lingbot-train:$TAG" | "$zip" -1 > "$out.partial"
  mv "$out.partial" "$out"
  (cd "$SAVE_DIR" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
  ls -lh "$out"; cat "$out.sha256"
fi
