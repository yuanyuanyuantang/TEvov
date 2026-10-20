#!/usr/bin/env python3
"""Hash on-disk model runtime files; never claim identity of live GPU weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tevo.experiment.suite import atomic_json
from tevo.reproducibility import canonical_hash, sha256_file

# Include inference code, weights, configuration and tokenizer assets, while
# avoiding hidden download caches and incidental documentation/log files.
RUNTIME_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx",
                    ".json", ".model", ".txt", ".tiktoken", ".jinja", ".py"}


def inventory(root: Path) -> dict[str, Path]:
    paths = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() in RUNTIME_SUFFIXES and path.is_file():
            paths[relative.as_posix()] = path
    if not paths:
        raise ValueError("No supported model runtime files found")
    return paths


def signature(path: Path) -> tuple:
    current, link = path.stat(), path.lstat()
    if not stat.S_ISREG(current.st_mode):
        raise ValueError("Model artifact is not a regular file")
    return tuple((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for s in (current, link))


def validate_indexes(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        if not name.endswith((".safetensors.index.json", ".bin.index.json")):
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid weight_map in model index: {name}")
        if any(not isinstance(shard, str) for shard in mapping.values()):
            raise ValueError(f"Invalid shard name in model index: {name}")
        for shard in set(mapping.values()):
            relative = PurePosixPath(shard)
            if relative.is_absolute() or ".." in relative.parts or "\\" in shard:
                raise ValueError(f"Unsafe shard name in model index: {name}")
            expected = (PurePosixPath(name).parent / relative).as_posix()
            if expected not in paths:
                raise ValueError(f"Missing referenced shard: {expected}")


def capture(model_dir: Path, model_id: str) -> dict:
    if not model_id.strip():
        raise ValueError("model-id must not be empty")
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("model-dir must be an existing directory")
    started = time.monotonic()
    paths = inventory(root)
    initial = {name: signature(path) for name, path in paths.items()}
    validate_indexes(paths)
    files = {}
    for name, path in paths.items():
        if signature(path) != initial[name]:
            raise ValueError(f"Model artifact changed before hashing: {name}")
        digest = sha256_file(path)
        if signature(path) != initial[name]:
            raise ValueError(f"Model artifact changed while hashing: {name}")
        files[name] = {"bytes": initial[name][0][2], "sha256": digest}
    if inventory(root).keys() != paths.keys() or any(signature(path) != initial[name] for name, path in paths.items()):
        raise ValueError("Model artifact inventory changed during capture")
    return {"schema_version": 1, "model_id": model_id, "files": files,
            "files_manifest_sha256": canonical_hash({"model_id": model_id, "files": files}),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "capture_seconds": time.monotonic() - started,
            "runtime_suffixes": sorted(RUNTIME_SUFFIXES),
            "live_gpu_weights_verified": False,
            "scope": "Selected on-disk weights/configuration/tokenizer/inference files only; model_id is a declared serving ID. This does not prove which weights a live GPU service loaded."}


def capture_to_file(model_dir: Path, model_id: str, output: Path) -> dict:
    # Refuse before reading potentially large weights, and again atomically when
    # publishing. Symlink outputs are never followed or silently overwritten.
    if os.path.lexists(output):
        raise FileExistsError("Model artifact output exists; choose a new path")
    manifest = capture(model_dir, model_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tevo-model-capture-", dir=output.parent) as directory:
        temporary = Path(directory) / "manifest.json"
        atomic_json(temporary, manifest)
        os.link(temporary, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help="Declared serving ID; not a GPU identity attestation")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = capture_to_file(args.model_dir, args.model_id, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Model artifact capture refused: {error}\n")
    print(json.dumps({"files": len(manifest["files"]), "files_manifest_sha256": manifest["files_manifest_sha256"],
                      "live_gpu_weights_verified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
