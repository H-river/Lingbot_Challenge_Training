#!/usr/bin/env python3
"""Verify patched files against the separate augmentation hash manifest."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'docs/patched_source_sha256.json').read_text())
for name,expected in manifest.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit(f'PATCHED_SOURCE_MISMATCH {name}')
print(f'PATCHED_SOURCE_VERIFIED {len(manifest)} files')
