#!/usr/bin/env bash
# Read-only inventory. No simulation, package installation, or secrets/config dumps.
set -uo pipefail
section() { printf '\n## %s\n' "$1"; }
probe() {
  printf '\n$'; printf ' %q' "$@"; printf '\n'
  "$@"
  local rc=$?
  if (( rc != 0 )); then printf 'PROBE_UNAVAILABLE exit=%s\n' "$rc"; fi
  return 0
}
section identity
probe hostname
probe id
probe uname -a
probe cat /etc/os-release
section resources
probe lscpu
probe nproc
probe free -h
probe df -hT
probe df -i
probe findmnt -r -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL
probe lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT
section candidate_directories_not_persistence_proof
for d in /root /data /mnt /mnt/data /workspace /opt /home; do
  if [[ -d "$d" ]]; then probe ls -ld "$d"; probe ls -A "$d"; fi
done
section tools
for c in conda mamba micromamba python python3 pip gcc g++ cmake ninja git ffmpeg nvidia-smi nvcc rocm-smi hipcc; do
  if command -v "$c" >/dev/null 2>&1; then command -v "$c"; else printf 'NOT_FOUND %s\n' "$c"; fi
done
for c in python python3 gcc g++ cmake git nvcc hipcc; do
  if command -v "$c" >/dev/null 2>&1; then probe "$c" --version; fi
done
section isolated_environments
if command -v conda >/dev/null 2>&1; then probe conda env list --json; fi
for c in /root/miniconda3/bin/conda /root/anaconda3/bin/conda /opt/conda/bin/conda; do
  if [[ -x "$c" ]]; then probe "$c" env list --json; fi
done
section accelerators_absence_is_expected
for d in /dev/nvidia* /dev/kfd /dev/dri; do [[ ! -e "$d" ]] || probe ls -ld "$d"; done
if command -v nvidia-smi >/dev/null 2>&1; then probe nvidia-smi; fi
if command -v rocm-smi >/dev/null 2>&1; then probe rocm-smi; fi
for d in /usr/local/cuda* /opt/rocm*; do [[ ! -e "$d" ]] || probe ls -ld "$d"; done
section connectivity
for url in https://github.com https://raw.githubusercontent.com https://huggingface.co https://modelscope.cn https://pypi.org/simple/ https://repo.anaconda.com https://conda.anaconda.org/conda-forge/ https://download.pytorch.org/whl/; do
  printf '%s\n' "$url"
  probe curl -L -sS -o /dev/null --connect-timeout 8 --max-time 20 -w 'http=%{http_code} seconds=%{time_total}\n' "$url"
done
section constraints
printf 'Read AGENTS.md, CLAUDE.md and competition rules in the selected project and its ancestors before making changes. Mount inventory alone does not establish persistence.\n'
