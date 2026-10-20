#!/usr/bin/env python3
"""Materialize the pinned BFCL source and reviewed local patch without pip installs.

Use --source-dir for an offline Git object source. Its dirty working tree is never
copied. Existing destinations are verified and never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "third_party" / "bfcl.lock.json"
DEFAULT_DESTINATION = ROOT / "benchmark" / "bfcl" / "gorilla"
MANIFEST_NAME = "TEVO_BFCL_MANIFEST.json"


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_tree(destination: Path, lock: dict) -> None:
    """Require the locked bytes, including checker and environment implementations."""
    for name, expected_hash in lock["files"].items():
        path = destination / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or nonregular locked BFCL file: {name}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"BFCL file SHA256 mismatch: {name}")
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != MANIFEST_NAME
    }
    unexpected = actual - lock["files"].keys()
    if unexpected:
        raise ValueError(f"Unexpected BFCL files: {sorted(unexpected)[:5]}")


def extract_archive(data: bytes, destination: Path, allowed_files: set[str],
                    excluded_files: dict[str, dict] | None = None) -> None:
    """Extract regular locked files only; reject links and traversal paths."""
    excluded_files = excluded_files or {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe path in BFCL archive")
            if member.isdir():
                continue
            if not member.isfile() or member.name not in allowed_files | excluded_files.keys():
                raise ValueError(f"Unexpected member in BFCL archive: {member.name}")
            content = archive.extractfile(member)
            if content is None:
                raise ValueError(f"Unreadable BFCL archive member: {member.name}")
            raw = content.read()
            if member.name in excluded_files:
                if hashlib.sha256(raw).hexdigest() != excluded_files[member.name]["sha256"]:
                    raise ValueError("Excluded upstream BFCL artifact SHA256 mismatch")
                continue
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    if cwd is not None:
        # git apply otherwise discovers the enclosing TEvo repository and silently
        # skips paths outside the staging directory's repository-relative prefix.
        environment["GIT_CEILING_DIRECTORIES"] = str(cwd.parent)
    result = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *arguments], cwd=cwd, check=False, env=environment,
        capture_output=True, timeout=180,
    )
    if result.returncode:
        # Git stderr can include local path names or credentials from configuration.
        raise RuntimeError(f"Git operation {arguments[0]} failed (exit {result.returncode})")
    return result.stdout


def bootstrap(destination: Path, *, source_dir: Path | None = None,
              lock_path: Path = DEFAULT_LOCK) -> dict:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    patch = lock_path.parent / lock["patch"]
    if sha256_file(patch) != lock["patch_sha256"]:
        raise ValueError("Reviewed BFCL patch SHA256 mismatch")
    destination = destination.resolve()
    if destination.exists():
        verify_tree(destination, lock)
        return {"status": "verified", "commit": lock["commit"], "files": len(lock["files"])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bfcl-bootstrap-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        source = source_dir.resolve() if source_dir else staging / "repository.git"
        if source_dir is None:
            _run_git(["init", "--bare", str(source)])
            _run_git(["-C", str(source), "fetch", "--depth=1", lock["repository"], lock["commit"]])
        resolved_commit = _run_git(["-C", str(source), "rev-parse", f"{lock['commit']}^{{commit}}"])
        if resolved_commit.decode().strip() != lock["commit"]:
            raise ValueError("BFCL source does not contain the locked commit")
        data = _run_git([
            "-C", str(source), "archive", "--format=tar", lock["commit"],
            "berkeley-function-call-leaderboard", "LICENSE",
        ])
        tree = staging / "tree"
        tree.mkdir()
        extract_archive(data, tree, set(lock["files"]), lock.get("excluded_files"))
        _run_git(["apply", "--check", str(patch)], cwd=tree)
        _run_git(["apply", str(patch)], cwd=tree)
        verify_tree(tree, lock)
        manifest = {
            "schema_version": 1, "repository": lock["repository"], "commit": lock["commit"],
            "patch_sha256": lock["patch_sha256"], "file_count": len(lock["files"]),
            "lock_sha256": sha256_file(lock_path),
            "excluded_files": lock.get("excluded_files", {}),
        }
        (tree / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # The target must remain absent; an existing installation is never replaced.
        if destination.exists():
            raise FileExistsError("BFCL destination appeared during bootstrap")
        os.rename(tree, destination)
    return {"status": "created", **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Offline official Gorilla Git repository")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    arguments = parser.parse_args()
    try:
        result = bootstrap(arguments.destination, source_dir=arguments.source_dir, lock_path=arguments.lock)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"BFCL bootstrap failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
