#!/usr/bin/env python3
"""Summarise a two-phase smoke run (phase A: fresh start + save; phase B: resume).

Reads the trainer logs and the nvidia-smi memory samples, prints a human-readable
verdict, and writes results/smoke_summary_<ts>.json.
Usage: smoke_summary.py LOG_A LOG_B GPU_SAMPLES_CSV RUN_CONFIG_A OUT_JSON
"""
import json, pathlib, re, statistics, sys
import yaml

STEP = re.compile(r"Step (\d+)/\d+, Epoch .*?Loss ([0-9.eE+-]+|nan|inf).*?StepTime ([0-9.]+)s")
VRAM = re.compile(r"VRAM usage after epoch .*? max ([0-9.]+)GB")


def parse(path):
    text = pathlib.Path(path).read_text(errors="replace")
    steps = [(int(m.group(1)), m.group(2), float(m.group(3))) for m in STEP.finditer(text)]
    vram = [float(v) for v in VRAM.findall(text)]
    saved = "Distributed checkpoint saved at" in text
    return steps, vram, saved, text


def main():
    log_a, log_b, samples, cfg_a, out_json = sys.argv[1:6]
    cfg = yaml.safe_load(pathlib.Path(cfg_a).read_text())["train"]
    sa, va, saved_a, _ = parse(log_a)
    sb, vb, saved_b, text_b = parse(log_b)

    checks = []
    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("phase A trained steps", len(sa) >= 3, f"{len(sa)} steps logged")
    finite = all(l.lower() not in ("nan", "inf") for _, l, _ in sa + sb)
    check("loss is finite", finite and (sa or sb))
    check("phase A saved checkpoint", saved_a)
    check("phase B loaded the phase A checkpoint", "Load distributed checkpoint from" in text_b)
    first_b = sb[0][0] if sb else None
    resume_from = cfg["max_steps"]
    check("phase B resumed from the saved step", first_b == resume_from + 1,
          f"first step after resume = {first_b}, expected {resume_from + 1}")
    check("phase B saved checkpoint", saved_b)

    per_gpu = {}
    p = pathlib.Path(samples)
    if p.is_file():
        for line in p.read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                per_gpu[int(parts[0])] = max(per_gpu.get(int(parts[0]), 0), int(parts[1]))
    peak_gib = max(per_gpu.values()) / 1024 if per_gpu else (max(va + vb) if va + vb else None)

    # Exclude compile warm-up: use steps after the first 3 of phase A.
    warm = [t for s, _, t in sa if s > 3] or [t for _, _, t in sa]
    step_s = statistics.median(warm) if warm else None

    ok = all(c["ok"] for c in checks)
    summary = {
        "verdict": "PASS" if ok else "FAIL",
        "checks": checks,
        "config": {k: cfg[k] for k in ("micro_batch_size", "global_batch_size", "gradient_accumulation_steps",
                                       "enable_fp32", "enable_gradient_checkpointing")},
        "median_step_seconds": step_s,
        "peak_gpu_memory_GiB_nvidia_smi": peak_gib,
        "peak_gpu_memory_GiB_per_gpu": {k: round(v / 1024, 1) for k, v in sorted(per_gpu.items())},
        "eta_hours": {str(n): round(step_s * n / 3600, 1) for n in (1000, 2000, 3000, 5000, 10000, 50000)} if step_s else {},
        "steps_per_24h": int(86400 / step_s) if step_s else None,
    }
    pathlib.Path(out_json).write_text(json.dumps(summary, indent=2) + "\n")

    print("\n================ SMOKE RESULT ================")
    for c in checks:
        print(f"  [{'OK' if c['ok'] else 'XX'}] {c['check']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    c = summary["config"]
    print(f"  batch: micro {c['micro_batch_size']} x GPUs x accum {c['gradient_accumulation_steps']} = {c['global_batch_size']}"
          f" | fp32={c['enable_fp32']} | grad_ckpt={c['enable_gradient_checkpointing']}")
    if peak_gib is not None:
        print(f"  peak GPU memory: {peak_gib:.1f} GiB of ~80 GiB")
        if peak_gib < 50:
            print("    -> plenty of headroom: Hu may try a larger MBS (e.g. MBS=8) for speed")
        elif peak_gib > 75:
            print("    -> very close to the limit: long runs may OOM; consider MBS=2")
    if step_s:
        print(f"  median step time: {step_s:.1f} s   (~{summary['steps_per_24h']} steps per 24 h)")
        print("  estimated training time:  " + "   ".join(f"{k} steps ≈ {v} h" for k, v in summary["eta_hours"].items()))
    print(f"  VERDICT: {summary['verdict']}")
    print(f"  saved: {out_json}")
    print("  -> send this whole block to Hu.")
    print("==============================================\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
