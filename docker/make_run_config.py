#!/usr/bin/env python3
"""Build the per-run training config from configs/train_8xA100.yaml.

Only the knobs listed in OVERRIDABLE may change; every other hyperparameter is
copied unchanged from the reviewed base config. The result and a provenance
record are written next to the run's checkpoints so the run can be reproduced.

Usage (inside the container):
  make_run_config.py --kind train --out-dir /root/lingbot_challenge/checkpoints/train_X \
      --ngpu 8 [--load-checkpoint PATH]
Overrides come from the environment: MBS, GBS, MAX_STEPS, SAVE_STEPS,
PRECISION (fp32|bf16), NUM_WORKERS, GRAD_CKPT (1|0).
"""
import argparse, datetime, hashlib, json, os, pathlib, sys
import yaml

ROOT = pathlib.Path(os.environ.get("LINGBOT_ROOT", "/root/lingbot_challenge"))
BASE = ROOT / "configs/train_8xA100.yaml"
OVERRIDABLE = ["MBS", "GBS", "MAX_STEPS", "SAVE_STEPS", "PRECISION", "NUM_WORKERS", "GRAD_CKPT"]


def die(msg):
    print(f"CONFIG_ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) <= 0:
        die(f"{name} must be a positive integer, got {raw!r}")
    return int(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True, choices=["smoke-a", "smoke-b", "train", "resume"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ngpu", type=int, required=True)
    p.add_argument("--load-checkpoint")
    p.add_argument("--max-steps", type=int, help="forced value (smoke only)")
    p.add_argument("--save-steps", type=int, help="forced value (smoke only)")
    a = p.parse_args()

    if a.ngpu < 1:
        die('--ngpu must be positive')
    if a.kind.startswith('smoke') and (not a.max_steps or a.max_steps < 1 or not a.save_steps or a.save_steps < 1):
        die('smoke requires positive --max-steps and --save-steps')
    cfg = yaml.safe_load(BASE.read_text())
    tr, data = cfg["train"], cfg["data"]

    mbs = env_int("MBS", tr["micro_batch_size"])
    gbs = env_int("GBS", tr["global_batch_size"])
    if gbs % (mbs * a.ngpu):
        ok = sorted({mbs * a.ngpu * k for k in range(1, 4096)}, key=lambda g: abs(g - gbs))[:4]
        die(f"global batch {gbs} is not divisible by MBS({mbs}) x GPUs({a.ngpu}). "
            f"Ask Hu which to use: GBS={' / GBS='.join(map(str, sorted(ok)))}, or a different MBS.")
    gas = gbs // (mbs * a.ngpu)

    if a.kind.startswith("smoke"):
        max_steps, save_steps = a.max_steps, a.save_steps
    else:
        if not os.environ.get("MAX_STEPS", "").strip():
            die("MAX_STEPS is required for train/resume (e.g. MAX_STEPS=3000). "
                "Use the ETA table printed by ./run.sh smoke to choose it.")
        max_steps = env_int("MAX_STEPS", 0)
        save_steps = env_int("SAVE_STEPS", tr["save_steps"])

    precision = os.environ.get("PRECISION", "fp32").strip().lower() or "fp32"
    if precision not in ("fp32", "bf16"):
        die(f"PRECISION must be fp32 or bf16, got {precision!r}")
    grad_ckpt = os.environ.get("GRAD_CKPT", "1").strip()
    if grad_ckpt not in ("0", "1"):
        die("GRAD_CKPT must be 0 or 1")

    out = pathlib.Path(a.out_dir)
    tr.update(
        output_dir=str(out),
        micro_batch_size=mbs,
        global_batch_size=gbs,
        gradient_accumulation_steps=gas,
        max_steps=max_steps,
        save_steps=save_steps,
        enable_fp32=(precision == "fp32"),
        enable_gradient_checkpointing=(grad_ckpt == "1"),
        enable_resume=False,
    )
    data["num_workers"] = env_int("NUM_WORKERS", data["num_workers"])
    if a.load_checkpoint:
        ck = pathlib.Path(a.load_checkpoint)
        if not ck.is_dir() or not ck.name.startswith("global_step_"):
            die(f"--load-checkpoint must be an existing .../checkpoints/global_step_N directory: {ck}")
        tr["load_checkpoint_path"] = str(ck)
    else:
        tr.pop("load_checkpoint_path", None)  # launcher contract: omit, never null

    # Same invariant launch_clean_training.sh asserts, checked early with a readable message.
    assert tr["global_batch_size"] == tr["micro_batch_size"] * a.ngpu * tr["gradient_accumulation_steps"]

    # Kept beside (not inside) the run dir: the launcher requires output_dir to be empty at start.
    staging = out.parent / f"{out.name}.config"
    staging.mkdir(parents=True, exist_ok=False)
    run_cfg = staging / "run_config.yaml"
    run_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=200))
    prov = {
        "kind": a.kind,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_config": str(BASE),
        "base_config_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "run_config_sha256": hashlib.sha256(run_cfg.read_bytes()).hexdigest(),
        "env_overrides": {k: os.environ[k] for k in OVERRIDABLE if os.environ.get(k, "").strip()},
        "ngpu": a.ngpu,
        "effective": {"micro_batch_size": mbs, "global_batch_size": gbs, "gradient_accumulation_steps": gas,
                      "max_steps": max_steps, "save_steps": save_steps, "precision": precision,
                      "gradient_checkpointing": grad_ckpt == "1", "num_workers": data["num_workers"],
                      "load_checkpoint_path": tr.get("load_checkpoint_path")},
        "image": {"tag": os.environ.get("LINGBOT_IMAGE_TAG"), "repo_commit": os.environ.get("LINGBOT_GIT_COMMIT")},
        "gpus": os.environ.get("LINGBOT_GPU_NAMES"),
    }
    (staging / "run_provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(run_cfg)


if __name__ == "__main__":
    main()
