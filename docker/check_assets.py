#!/usr/bin/env python3
"""Check mounted model weights against configs/assets.manifest.json.

Default: every expected file exists with the exact byte size.
--full : also verify sha256 of every LFS (large) file. Takes a few minutes.
Always: refuse any weights whose name contains "robotwin" (the official
post-trained lingbot-vla-v2-6b-robotwin weights are forbidden for training).
"""
import hashlib, json, os, pathlib, sys

ROOT = pathlib.Path(os.environ.get("LINGBOT_ROOT", "/root/lingbot_challenge"))
DEST = {"models": "models/lingbot-vla-v2-6b", "qwen": "models/Qwen3-VL-4B-Instruct-config",
        "moge": "models/MoGe-2-vitb-normal"}
# Docs/images in the model repos are not read by training; tolerate their absence.
OPTIONAL_SUFFIXES = (".md", ".png", ".gitattributes")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    full = "--full" in sys.argv
    manifest = json.loads((ROOT / "configs/assets.manifest.json").read_text())
    problems, checked, hashed = [], 0, 0

    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        print(f"FAIL models directory not mounted: {models_dir}")
        return 1
    forbidden = [str(p) for p in models_dir.rglob("*") if "robotwin" in p.name.lower()]
    if forbidden:
        problems.append("forbidden robotwin post-trained weights present (remove them): " + ", ".join(forbidden[:5]))

    for scope, rel in DEST.items():
        base = ROOT / rel
        for f in manifest[scope]["files"]:
            name = f["rfilename"]
            path = base / name
            if not path.is_file():
                if name.endswith(OPTIONAL_SUFFIXES):
                    continue
                problems.append(f"missing {rel}/{name}")
                continue
            checked += 1
            if path.stat().st_size != f["size"]:
                problems.append(f"size mismatch {rel}/{name}: {path.stat().st_size} != {f['size']}")
                continue
            if full and "lfs" in f:
                hashed += 1
                print(f"  sha256 {rel}/{name} ...", flush=True)
                if sha256(path) != f["lfs"]["sha256"]:
                    problems.append(f"sha256 mismatch {rel}/{name}")

    for p in problems:
        print("FAIL", p)
    if problems:
        return 1
    print(f"PASS weights: {checked} files sized correctly" + (f", {hashed} large files sha256-verified" if full else
          " (run './run.sh check --full' to also verify sha256)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
