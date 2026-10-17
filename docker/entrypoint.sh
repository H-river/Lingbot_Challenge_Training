#!/usr/bin/env bash
# Container entrypoint. Teammates call ./run.sh on the host, which calls this.
#   check [--full]          environment, GPUs, NCCL, weights, data audit, disk
#   smoke                   2-phase real-batch test: train 10 steps + save, resume +3 steps + save
#   train                   formal training (MAX_STEPS required)
#   resume <global_step_N>  continue a stopped run from one of its checkpoints
#   shell                   interactive bash
set -Eeuo pipefail
R=/root/lingbot_challenge
export LINGBOT_PATHS_FILE="$R/configs/paths.env"
source "$LINGBOT_PATHS_FILE"
PY="$LINGBOT_ENV/bin/python"
export PATH="$LINGBOT_ENV/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR="$R/cache/inductor" TRITON_CACHE_DIR="$R/cache/triton"
TS=$(date -u +%Y%m%dT%H%M%SZ)
cd "$R"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }
ylw() { printf '\033[33m%s\033[0m\n' "$*"; }

prepare_dirs() {
  mkdir -p "$R"/{logs,results,checkpoints,cache/tmp,cache/inductor,cache/triton}
  # The data audit reads per-task conversion validation records from results/.
  # They are the same records committed under evidence/; seed them if absent.
  mkdir -p "$R/results/conversion_validation"
  cp -n "$R"/evidence/conversion_validation/*.json "$R/results/conversion_validation/"
}

gpu_count() { "$PY" -c 'import torch; print(torch.cuda.device_count())'; }
gpu_names() { nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | paste -sd';' -; }
export LINGBOT_GPU_NAMES="$(gpu_names 2>/dev/null || echo unknown)"

# ---------------------------------------------------------------- check
cmd_check() {
  local full="" fails=0 warns=0
  [[ "${1:-}" == "--full" ]] && full="--full"
  pass() { grn "PASS $*"; }
  warn() { ylw "WARN $*"; warns=$((warns + 1)); }
  fail() { red  "FAIL $*"; fails=$((fails + 1)); }

  echo "== 1/8 GPUs and driver"
  if ! nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi does not work inside the container (is nvidia-container-toolkit installed? did you pass --gpus?)"
  else
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    local n drv
    n=$(gpu_count 2>/dev/null || echo 0); drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if (( drv >= 570 )); then pass "driver $drv (>= 570, CUDA 12.8 OK)"
    elif (( drv >= 525 )); then warn "driver $drv is older than 570; CUDA 12.8 may work via compatibility but is untested"
    else fail "driver $drv is too old for CUDA 12.8 (need >= 570)"; fi
    if (( n == 8 )); then pass "8 GPUs visible"
    elif (( n >= 1 )); then warn "$n GPUs visible (config is tuned for 8; batch math adapts, speed will differ)"
    else fail "PyTorch sees 0 GPUs"; fi
    nvidia-smi --query-gpu=name --format=csv,noheader | grep -q A100 || warn "GPUs are not A100; memory/speed numbers in DEPLOY.md do not apply"
  fi

  echo "== 2/8 CUDA math, flash-attn, model code import"
  if bash "$R/scripts/check_training_gpu.sh" >"$R/logs/check_gpu_$TS.log" 2>&1; then pass "torch/flash-attn GPU kernels and LingBot model import"
  else fail "GPU kernel check failed; see logs/check_gpu_$TS.log"; tail -5 "$R/logs/check_gpu_$TS.log"; fi

  echo "== 3/8 Multi-GPU communication (NCCL)"
  local n; n=$(gpu_count 2>/dev/null || echo 0)
  if (( n > 1 )); then
    if timeout 300 "$PY" -m torch.distributed.run --standalone --nproc_per_node="$n" "$R/docker/nccl_check.py" \
         >"$R/logs/check_nccl_$TS.log" 2>&1; then
      grep -E "^(PASS|WARN)" "$R/logs/check_nccl_$TS.log" | while read -r l; do [[ $l == WARN* ]] && ylw "$l" || grn "$l"; done
      grep -q "^WARN" "$R/logs/check_nccl_$TS.log" && warns=$((warns + 1))
    else fail "NCCL all-reduce failed or hung; see logs/check_nccl_$TS.log"; tail -5 "$R/logs/check_nccl_$TS.log"; fi
  else ylw "SKIP single GPU"; fi

  echo "== 4/8 Model weights"
  if "$PY" "$R/docker/check_assets.py" $full; then :; else fails=$((fails + 1)); fi

  echo "== 5/8 Training data (clean-only audit, the same one every training launch runs)"
  if "$PY" "$R/scripts/audit_training_ready.py" >"$R/logs/check_data_$TS.log" 2>&1; then
    pass "data audit: $(grep -o '"frames": [0-9]*' "$R/logs/check_data_$TS.log") , 50 tasks, 2500 clean episodes, seen instructions only"
  else fail "data audit failed; see logs/check_data_$TS.log"; tail -3 "$R/logs/check_data_$TS.log"; fi

  echo "== 6/8 Disk space for checkpoints (one checkpoint ≈ 49 GB)"
  local free; free=$(( $(df -B1 --output=avail "$R/checkpoints" | tail -1) / 1024**3 ))
  if (( free < 120 )); then fail "only ${free} GiB free; smoke needs >= 120 GiB"
  elif (( free < 400 )); then warn "${free} GiB free; enough for smoke, but a long run keeps several checkpoints (>= 400 GiB recommended)"
  else pass "${free} GiB free"; fi
  if touch "$R/checkpoints/.write_test" 2>/dev/null; then rm -f "$R/checkpoints/.write_test"; else fail "checkpoints/ is not writable"; fi

  echo "== 7/8 Shared memory, CPU and RAM"
  local shm cpus ram
  shm=$(( $(df -B1 --output=size /dev/shm | tail -1) / 1024**3 ))
  cpus=$(nproc); ram=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024**2 ))
  (( shm >= 16 )) && pass "/dev/shm ${shm} GiB" || warn "/dev/shm only ${shm} GiB (run.sh uses --ipc=host; if you started docker yourself, add it)"
  local want=$(( (n > 0 ? n : 1) * 5 ))
  (( cpus >= want )) && pass "${cpus} CPU cores" || warn "${cpus} CPU cores for ${n} GPUs; if training stalls on data, set NUM_WORKERS=2"
  (( ram >= 256 )) && pass "${ram} GiB RAM" || warn "${ram} GiB RAM; 8 training processes may need >256 GiB at startup"

  echo "== 8/8 Compliance guard"
  if grep -q "randomized" "$R/configs/train_clean.txt"; then fail "train list mentions randomized data"
  else pass "train list = 50 clean tasks only; launcher refuses any other data/model path"; fi

  echo
  if (( fails == 0 )); then grn "CHECK PASSED ($warns warnings). Next: ./run.sh smoke"; else red "CHECK FAILED: $fails problem(s), $warns warning(s). Fix the FAIL lines above (see DEPLOY.md, '常见问题')."; fi
  (( fails == 0 ))
}

# ---------------------------------------------------------------- shared launch
launch() {  # $1 config, $2 log file
  bash "$R/scripts/launch_clean_training.sh" "$1" 2>&1 | tee "$2"
}

# ---------------------------------------------------------------- smoke
cmd_smoke() {
  prepare_dirs
  local n free; n=$(gpu_count)
  free=$(( $(df -B1 --output=avail "$R/checkpoints" | tail -1) / 1024**3 ))
  (( free >= 120 )) || { red "Need >= 120 GiB free in checkpoints/ for smoke, have ${free} GiB"; exit 2; }
  local A="$R/checkpoints/smoke_${TS}_a" B="$R/checkpoints/smoke_${TS}_b"
  local la="$R/logs/smoke_${TS}_a.log" lb="$R/logs/smoke_${TS}_b.log" gpucsv="$R/logs/smoke_${TS}_gpu_mem.csv"
  echo "SMOKE on $n GPU(s): phase A = 10 real-batch steps + checkpoint, phase B = resume + 3 steps + checkpoint."
  echo "This takes roughly 30-60 minutes. Logs: logs/smoke_${TS}_*.log"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 >"$gpucsv" 2>/dev/null &
  local sampler=$!
  trap 'kill $sampler 2>/dev/null || true' EXIT

  local cfga cfgb
  cfga=$("$PY" "$R/docker/make_run_config.py" --kind smoke-a --out-dir "$A" --ngpu "$n" --max-steps 10 --save-steps 10)
  echo "---- phase A: $cfga"
  launch "$cfga" "$la" || { red "SMOKE FAILED in phase A (fresh training). Send logs/smoke_${TS}_a.log to Hu."; exit 1; }
  cfgb=$("$PY" "$R/docker/make_run_config.py" --kind smoke-b --out-dir "$B" --ngpu "$n" --max-steps 13 --save-steps 1000 \
         --load-checkpoint "$A/checkpoints/global_step_10")
  echo "---- phase B: $cfgb"
  launch "$cfgb" "$lb" || { red "SMOKE FAILED in phase B (resume). Send logs/smoke_${TS}_b.log to Hu."; exit 1; }
  kill "$sampler" 2>/dev/null || true

  local rc=0
  "$PY" "$R/docker/smoke_summary.py" "$la" "$lb" "$gpucsv" "$cfga" "$R/results/smoke_summary_${TS}.json" || rc=$?
  if (( rc == 0 )) && [[ "${KEEP_SMOKE:-0}" != 1 ]]; then
    # Only the two smoke run directories created above; configs and logs are kept.
    case "$A$B" in *"/checkpoints/smoke_${TS}_a"*"/checkpoints/smoke_${TS}_b"*) rm -rf -- "$A" "$B";; esac
    echo "Smoke checkpoints deleted to free ~100 GB (set KEEP_SMOKE=1 to keep them)."
  fi
  exit "$rc"
}

# ---------------------------------------------------------------- train / resume
cmd_train() {
  prepare_dirs
  local n name out cfg log
  n=$(gpu_count)
  name="${RUN_NAME:-train_${TS}}"
  [[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || { red "RUN_NAME may only contain letters, digits, . _ -"; exit 2; }
  out="$R/checkpoints/$name"; log="$R/logs/${name}.log"
  cfg=$("$PY" "$R/docker/make_run_config.py" --kind train --out-dir "$out" --ngpu "$n")
  echo "TRAIN $name on $n GPU(s). Config: checkpoints/${name}.config/run_config.yaml  Log: logs/${name}.log"
  local start; start=$(date -u +%FT%TZ)
  if launch "$cfg" "$log"; then
    printf '{"run":"%s","start":"%s","end":"%s","status":"finished"}\n' "$name" "$start" "$(date -u +%FT%TZ)" >"$R/results/${name}.status.json"
    grn "TRAINING FINISHED: checkpoints/$name/checkpoints/ . Tell Hu."
  else
    printf '{"run":"%s","start":"%s","end":"%s","status":"failed"}\n' "$name" "$start" "$(date -u +%FT%TZ)" >"$R/results/${name}.status.json"
    red "TRAINING STOPPED WITH AN ERROR. Send the last 200 lines of logs/${name}.log to Hu. Resume: ./run.sh resume <checkpoint>"
    exit 1
  fi
}

cmd_resume() {
  local ck="${1:?Usage: resume checkpoints/<run>/checkpoints/global_step_N}"
  [[ "$ck" = /* ]] || ck="$R/${ck#./}"
  [[ -d "$ck" && "$(basename "$ck")" == global_step_* ]] || { red "Not a checkpoint directory: $ck"; exit 2; }
  local run_dir prov
  run_dir=$(dirname "$(dirname "$ck")")
  prov="$(dirname "$run_dir")/$(basename "$run_dir").config/run_provenance.json"
  if [[ -f "$prov" ]]; then
    # Inherit the original run's batch/precision/steps so the resumed run is the same experiment.
    eval "$("$PY" - "$prov" <<'PY'
import json,shlex,sys,os
e=json.load(open(sys.argv[1]))["effective"]
m={"MBS":e["micro_batch_size"],"GBS":e["global_batch_size"],"MAX_STEPS":e["max_steps"],"SAVE_STEPS":e["save_steps"],
   "PRECISION":e["precision"],"NUM_WORKERS":e["num_workers"],"GRAD_CKPT":"1" if e["gradient_checkpointing"] else "0"}
for k,v in m.items():
    if not os.environ.get(k,"").strip(): print(f"export {k}={shlex.quote(str(v))}")
PY
)"
    echo "Resuming with the original run's settings: MBS=$MBS GBS=$GBS MAX_STEPS=$MAX_STEPS PRECISION=$PRECISION"
  else
    ylw "No run_provenance.json next to $run_dir; using current environment settings."
  fi
  RUN_NAME="${RUN_NAME:-$(basename "$run_dir")_resume_${TS}}"
  export RUN_NAME
  prepare_dirs
  local n out cfg log
  n=$(gpu_count); out="$R/checkpoints/$RUN_NAME"; log="$R/logs/${RUN_NAME}.log"
  cfg=$("$PY" "$R/docker/make_run_config.py" --kind resume --out-dir "$out" --ngpu "$n" --load-checkpoint "$ck")
  echo "RESUME from $ck -> checkpoints/$RUN_NAME  Log: logs/${RUN_NAME}.log"
  launch "$cfg" "$log" && grn "TRAINING FINISHED: checkpoints/$RUN_NAME/checkpoints/ . Tell Hu." \
    || { red "TRAINING STOPPED WITH AN ERROR. Send the last 200 lines of logs/${RUN_NAME}.log to Hu."; exit 1; }
}

case "${1:-help}" in
  check)  shift; prepare_dirs; cmd_check "$@" ;;
  smoke)  cmd_smoke ;;
  train)  cmd_train ;;
  resume) shift; cmd_resume "$@" ;;
  shell)  prepare_dirs; exec bash ;;
  *) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
