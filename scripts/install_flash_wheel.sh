#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"
log_run install_flash_wheel
mkdir -p "$LINGBOT_ROOT/cache/locks"
exec 9>"$LINGBOT_ROOT/cache/locks/flash-wheel.lock"
flock 9
readarray -t wheel < <(python3 - "$LINGBOT_ROOT/configs/flash-attn-wheel.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));print(m['name']);print(m['browser_download_url']);print(m['digest'].split(':')[1])
PY
)
mkdir -p "$LINGBOT_ROOT/cache/wheels"
file="$LINGBOT_ROOT/cache/wheels/${wheel[0]}"
if [[ ! -f "$file" ]]; then
 curl --fail --location --retry 3 --connect-timeout 15 --continue-at - --output "$file.partial" "${wheel[1]}"
 printf '%s  %s\n' "${wheel[2]}" "$file.partial" | sha256sum --check
 mv "$file.partial" "$file"
fi
printf '%s  %s\n' "${wheel[2]}" "$file" | sha256sum --check
[[ ${1:-} != --download-only ]] || exit 0
"$LINGBOT_ENV/bin/python" - <<'PY'
import torch,sys
assert sys.version_info[:2]==(3,12)
assert torch.__version__.split('+')[0]=='2.8.0'
assert torch.version.cuda and torch.version.cuda.startswith('12.')
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
print('Verified wheel compatibility: Python 3.12, Torch 2.8, CUDA 12, CXX11 ABI TRUE')
PY
"$LINGBOT_ENV/bin/python" -m pip install --no-deps "$file"
