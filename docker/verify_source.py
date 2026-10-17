#!/usr/bin/env python3
"""Build-time: the vendored upstream source must match docs/upstream_source_sha256.json byte-for-byte."""
import hashlib, json, pathlib
root = pathlib.Path("repos/lingbot-vla-v2")
manifest = json.loads(pathlib.Path("docs/upstream_source_sha256.json").read_text())
for name, sha in manifest.items():
    p = root / name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha, "SOURCE_MISMATCH: " + name
print("PINNED_SOURCE_SNAPSHOT_VERIFIED", len(manifest), "files")
