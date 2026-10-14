#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run setup_training_handoff
repo="$LINGBOT_ROOT/repos/lingbot-vla-v2"
clone_pin() {
 local url=$1 dest=$2 rev=$3
 if [[ "$dest" == "$repo" && -d "$dest" && ! -e "$dest/.git" ]]; then
  python3 - "$dest" "$LINGBOT_ROOT/docs/upstream_source_sha256.json" <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads(pathlib.Path(sys.argv[2]).read_text())
for name,sha in manifest.items():
 p=root/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'SOURCE_MISMATCH: '+name
print('PINNED_SOURCE_SNAPSHOT_VERIFIED')
VERIFY
  return
 fi
 if [[ ! -e "$dest" ]]; then git clone --filter=blob:none --no-checkout "$url" "$dest"; git -C "$dest" checkout --detach "$rev"; fi
 [[ $(git -C "$dest" rev-parse HEAD) == "$rev" ]] || { echo "REVISION_MISMATCH: $dest"; exit 2; }
 git -C "$dest" diff --quiet && git -C "$dest" diff --cached --quiet || { echo "DIRTY_REPO: $dest"; exit 2; }
 git -C "$dest" submodule update --init --recursive
}
clone_pin https://github.com/Robbyant/lingbot-vla-v2.git "$repo" be969b8fd117fb70550c5d4bf4bc328211b5b1b6
clone_pin https://github.com/RoboTwin-Platform/RoboTwin.git "$LINGBOT_ROOT/repos/RoboTwin" 13c3c47ff4312dd62484bcd51be034af55c062d1
if [[ ! -e "$LINGBOT_ENV" ]]; then
 "$CONDA_EXE" create -y -p "$LINGBOT_ENV" --file "$LINGBOT_ROOT/docs/versions/lingbotvla.conda-explicit.txt"
 touch "$LINGBOT_ENV/.handoff-managed"
elif [[ ! -f "$LINGBOT_ENV/.handoff-managed" && ! -f "$LINGBOT_ENV/.lingbot-managed" ]]; then
 echo 'UNMANAGED_EXISTING_ENV: choose another prefix'; exit 2
fi
py="$LINGBOT_ENV/bin/python"
"$py" -m pip install 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$py" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
"$py" -m pip install --no-deps -r "$LINGBOT_ROOT/docs/versions/lingbotvla.portable-pins.txt"
"$py" -m pip install --no-deps 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz#sha256=2ed4981213f0931e741df1dfc3986a36af2fe44db8540d723108c6fd2defc14b'
"$py" -m pip install --no-deps -e "$repo"
"$py" -m pip install --no-deps -e "$repo/lingbotvla/models/vla/vision_models/lingbot-depth"
"$py" -m pip install --no-deps -e "$repo/lingbotvla/models/vla/vision_models/MoGe"
"$py" -m pip install --no-deps 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
"$py" - <<'PY'
import sys,torch
assert sys.version_info[:2]==(3,12)
assert torch.__version__.split('+')[0]=='2.8.0' and torch.version.cuda=='12.8'
assert torch._C._GLIBCXX_USE_CXX11_ABI
print('TORCH_STACK',torch.__version__,torch.version.cuda,'ABI TRUE')
PY
bash "$LINGBOT_ROOT/scripts/install_flash_wheel.sh"
"$py" - <<'PY'
import os,pathlib,subprocess,json,importlib.metadata
r=pathlib.Path(os.environ['LINGBOT_ROOT']);p=subprocess.run([os.environ['LINGBOT_ENV']+'/bin/python','-m','pip','check'],capture_output=True,text=True);actual=p.stdout+p.stderr
(r/'results/pip-check.training-handoff.txt').write_text(actual)
lines=set(actual.replace('2.8.0+cu128','2.8.0').replace('0.23.0+cu128','0.23.0').splitlines());known=set((r/'configs/known_lingbot_pip_check.txt').read_text().splitlines());unknown=lines-known
if p.returncode==0:print('PIP_CHECK_PASS')
else:
 assert p.returncode==1 and not unknown, 'UNKNOWN_DEPENDENCY_CONFLICT: '+repr(unknown)
 print('KNOWN_METADATA_CONFLICT: official no-deps recipe; not a clean pip-check pass')
for line in (r/'docs/versions/lingbotvla.portable-pins.txt').read_text().splitlines():
 name,version=line.split('==');assert importlib.metadata.version(name)==version,(name,version,importlib.metadata.version(name))
import cv2,h5py,pyarrow,av,transformers,torchcodec,lerobot,accelerate,mlflow,utils3d,moge,mdm,flash_attn
import moge.model.v2,mdm.model.v2
print('CPU_TRAINING_DEPENDENCY_IMPORTS_PASS; no GPU execution claimed')
PY
"$py" -m pip freeze --all > "$LINGBOT_ROOT/results/training-handoff.pip-freeze.txt"
echo 'TRAINING_ENV_INSTALLED: data audit and GPU checks still required; no training launched.'
