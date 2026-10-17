#!/usr/bin/env bash
# LingBot clean-only training — the only command a deployer needs. See DEPLOY.md.
#
#   ./run.sh check [--full]   check machine, GPUs, weights and data (a few minutes)
#   ./run.sh smoke            real-batch test on all GPUs + resume test (~30-60 min)
#   MAX_STEPS=3000 ./run.sh train          start formal training in the background
#   ./run.sh resume checkpoints/<run>/checkpoints/global_step_N   continue a stopped run
#   ./run.sh logs             follow the running job's log (Ctrl-C only stops watching)
#   ./run.sh status           is something running? GPU usage
#   ./run.sh stop             stop the running job (asks first)
#   ./run.sh shell            bash inside the container (for debugging)
#
# Put this file in the deploy folder that contains models/ and data/ (or set LINGBOT_HOME).
# Optional: GPUS=0,1,2,3  IMAGE=lingbot-train:deploy  MBS GBS SAVE_STEPS PRECISION NUM_WORKERS GRAD_CKPT KEEP_SMOKE RUN_NAME
set -Eeuo pipefail
HOME_DIR="${LINGBOT_HOME:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE="${IMAGE:-lingbot-train:deploy}"
C=/root/lingbot_challenge
PASS_ENV=(MBS GBS MAX_STEPS SAVE_STEPS PRECISION NUM_WORKERS GRAD_CKPT KEEP_SMOKE RUN_NAME)

die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 2; }
info() { printf '\033[36m%s\033[0m\n' "$*"; }

preflight() {
  command -v docker >/dev/null || die "docker is not installed. Install Docker Engine + nvidia-container-toolkit first (DEPLOY.md step 0)."
  docker info >/dev/null 2>&1 || die "Cannot talk to docker. Try: sudo ./run.sh $*   (or add yourself to the docker group)"
  docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "Image $IMAGE not found. Load it first: docker load -i lingbot-train_*.tar.gz   (DEPLOY.md step 2)"
  [[ -d "$HOME_DIR/models/lingbot-vla-v2-6b" ]] || die "Missing $HOME_DIR/models/lingbot-vla-v2-6b — extract weights.tar here (DEPLOY.md step 3)."
  [[ -d "$HOME_DIR/data/clean_processed/lerobot" ]] || die "Missing $HOME_DIR/data/clean_processed — extract data_clean.tar here (DEPLOY.md step 3)."
  mkdir -p "$HOME_DIR"/{checkpoints,logs,results,cache}
}

running() { docker ps --filter "name=^lingbot-" --format '{{.Names}}' | head -1; }

docker_args() {
  local gpus="all"
  [[ -n "${GPUS:-}" ]] && gpus="\"device=${GPUS}\""
  ARGS=(--gpus "$gpus" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        -v "$HOME_DIR/models:$C/models:ro"
        -v "$HOME_DIR/data:$C/data:ro"
        -v "$HOME_DIR/checkpoints:$C/checkpoints"
        -v "$HOME_DIR/logs:$C/logs"
        -v "$HOME_DIR/results:$C/results"
        -v "$HOME_DIR/cache:$C/cache"
        -e "LINGBOT_IMAGE_TAG=$IMAGE")
  local k
  for k in "${PASS_ENV[@]}"; do [[ -n "${!k:-}" ]] && ARGS+=(-e "$k=${!k}"); done
  return 0
}

background() {  # $1 container name, rest = entrypoint args
  local name=$1; shift
  local r; r=$(running)
  [[ -z "$r" ]] || die "Container $r is already running. Use ./run.sh logs, or ./run.sh stop first."
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker_args
  docker run -d --name "$name" "${ARGS[@]}" "$IMAGE" "$@" >/dev/null
  info "Started '$name' in the background. It keeps running if you close this terminal or SSH."
  info "Watching the log now — press Ctrl-C to stop WATCHING (the job continues). Re-attach: ./run.sh logs"
  trap 'echo; info "Stopped watching. Job still running: ./run.sh logs | ./run.sh status"; exit 0' INT
  docker logs -f "$name" 2>&1 || true
  trap - INT
  local code; code=$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo "?")
  [[ "$code" == 0 ]] && info "'$name' finished OK." || die "'$name' exited with code $code. Log files are in $HOME_DIR/logs/."
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  check)
    preflight "$cmd"; docker_args
    docker run --rm "${ARGS[@]}" "$IMAGE" check "$@" ;;
  smoke)
    preflight "$cmd"; background lingbot-smoke smoke ;;
  train)
    [[ -n "${MAX_STEPS:-}" ]] || die "Set the number of steps Hu gave you, e.g.:  MAX_STEPS=3000 ./run.sh train"
    preflight "$cmd"; background lingbot-train train ;;
  resume)
    [[ -n "${1:-}" ]] || die "Usage: ./run.sh resume checkpoints/<run>/checkpoints/global_step_N"
    ck="${1#./}"; ck="${ck#"$HOME_DIR"/}"
    [[ -d "$HOME_DIR/$ck" ]] || die "Not found: $HOME_DIR/$ck  (list with: ls $HOME_DIR/checkpoints/*/checkpoints/)"
    preflight "$cmd"; background lingbot-train resume "$ck" ;;
  logs)
    r=$(running); [[ -n "$r" ]] || { r=$(docker ps -a --filter "name=^lingbot-" --format '{{.Names}}' | head -1); }
    [[ -n "$r" ]] || die "No lingbot job found."
    info "Following $r (Ctrl-C to stop watching)"; docker logs -f --tail 200 "$r" ;;
  status)
    r=$(running)
    if [[ -n "$r" ]]; then info "RUNNING: $r (since $(docker inspect -f '{{.State.StartedAt}}' "$r"))"
      docker logs --tail 400 "$r" 2>&1 | grep -E "Step [0-9]+/" | tail -1 || true
    else info "Nothing running."; docker ps -a --filter "name=^lingbot-" --format '  last: {{.Names}}  {{.Status}}'; fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || true ;;
  stop)
    r=$(running); [[ -n "$r" ]] || die "Nothing running."
    read -r -p "Stop $r? Training stops; you can continue later from the last saved checkpoint with ./run.sh resume. [y/N] " a
    [[ "$a" == y || "$a" == Y ]] && docker stop -t 60 "$r" && info "Stopped." ;;
  shell)
    preflight "$cmd"; docker_args; docker run --rm -it "${ARGS[@]}" "$IMAGE" shell ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
