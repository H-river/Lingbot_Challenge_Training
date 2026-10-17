#!/usr/bin/env python3
"""Build-time (CPU only) check that the image matches the frozen environment.
GPU kernels (flash-attn, NCCL) are checked at runtime by ./run.sh check."""
import importlib.metadata, pathlib, subprocess, sys
import torch

assert sys.version_info[:3] == (3, 12, 15), sys.version
assert torch.__version__.split("+")[0] == "2.8.0" and torch.version.cuda == "12.8", (torch.__version__, torch.version.cuda)
assert torch._C._GLIBCXX_USE_CXX11_ABI
for line in pathlib.Path("docs/versions/lingbotvla.portable-pins.txt").read_text().splitlines():
    name, version = line.split("==")
    got = importlib.metadata.version(name)
    assert got == version, (name, version, got)
for name, version in [("lerobot", "0.4.2"), ("flash_attn", "2.8.3")]:
    got = importlib.metadata.version(name).split("+")[0]
    assert got == version, (name, got)
p = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
lines = set((p.stdout + p.stderr).replace("2.8.0+cu128", "2.8.0").replace("0.23.0+cu128", "0.23.0").splitlines())
known = set(pathlib.Path("configs/known_lingbot_pip_check.txt").read_text().splitlines())
unknown = {l for l in lines - known if l.strip()}
assert p.returncode == 0 or not unknown, "UNKNOWN_DEPENDENCY_CONFLICT: " + repr(unknown)
import cv2, h5py, pyarrow, av, transformers, torchcodec, lerobot, accelerate, utils3d, moge, mdm  # noqa: F401,E401
import moge.model.v2, mdm.model.v2  # noqa: F401,E401
print("BUILD_VERIFY_PASS torch", torch.__version__, "| only known pip metadata conflicts")
