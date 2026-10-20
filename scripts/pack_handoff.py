#!/usr/bin/env python3
"""Package current code and required inputs, without Git, weights or run outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"tevo", "scripts", "tests", "configs", "benchmark", "artifacts", "results", "third_party", ".github"}
ROOT_FILES = {"README.md", "HANDOFF.md", ".gitignore", ".env.example", "pyproject.toml", "constraints-observed.txt"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", "logs", ".venv"}
SECRET = re.compile(rb"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def pack(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("choose an output outside the project")
    selected = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if set(relative.parts) & EXCLUDED_PARTS:
            continue
        allowed = relative.parts[0] in DIRECTORIES or relative.as_posix() in ROOT_FILES or (
            len(relative.parts) == 1 and path.name.startswith("requirements") and path.suffix == ".txt")
        if not allowed:
            continue
        if path.is_symlink():
            raise ValueError(f"linked file is not portable: {relative}")
        if path.is_file() and path.suffix not in {
                ".pyc", ".pyo", ".log", ".sqlite", ".zip", ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".key", ".secret"
        } and path.name != ".env":
            selected.append(path)
    files = {}
    # Read before opening the archive so a rejected credential leaves no partial package.
    contents = {}
    for path in selected:
        name, raw = path.relative_to(root).as_posix(), path.read_bytes()
        if SECRET.search(raw):
            raise ValueError(f"credential-like content: {name}")
        contents[name] = raw
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if not {"README.md", "HANDOFF.md", "pyproject.toml"}.issubset(files):
        raise ValueError("required handoff entry files are missing")
    manifest = {"kind": "tevo_collaboration_handoff", "files": files, "file_count": len(files),
                "model_weights_included": False, "historical_outputs_included": False,
                "note": "Historical metadata contains original paths; current execution uses the portable configs."}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in contents.items():
            archive.writestr("TEvov/" + name, raw)
        archive.writestr("TEvov/HANDOFF_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC verification failed")
        for name, item in files.items():
            if hashlib.sha256(archive.read("TEvov/" + name)).hexdigest() != item["sha256"]:
                raise ValueError("archive content verification failed")
    return {"file_count": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(ROOT, args.output), ensure_ascii=False, indent=2))
