"""Offline bootstrap checks: reject altered source bytes and unsafe tar members."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_bfcl.py"
spec = importlib.util.spec_from_file_location("bootstrap_bfcl", SCRIPT)
bootstrap_bfcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap_bfcl)


def test_locked_tree_detects_tampering_and_extras(tmp_path):
    source = tmp_path / "file.py"
    source.write_bytes(b"original")
    lock = {"files": {"file.py": hashlib.sha256(b"original").hexdigest()}}
    bootstrap_bfcl.verify_tree(tmp_path, lock)
    source.write_bytes(b"altered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bootstrap_bfcl.verify_tree(tmp_path, lock)
    source.write_bytes(b"original")
    (tmp_path / "unexpected.py").write_text("pass")
    with pytest.raises(ValueError, match="Unexpected"):
        bootstrap_bfcl.verify_tree(tmp_path, lock)


@pytest.mark.parametrize("name,kind", [("../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)])
def test_archive_rejects_traversal_and_links(tmp_path, name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../outside" if kind == tarfile.SYMTYPE else ""
        archive.addfile(member)
    with pytest.raises(ValueError):
        bootstrap_bfcl.extract_archive(stream.getvalue(), tmp_path, {name})


def test_patch_applies_inside_an_enclosing_repository(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    tree = tmp_path / "nested" / "tree"
    tree.mkdir(parents=True)
    (tree / "file.txt").write_text("before\n")
    patch = tmp_path / "change.patch"
    patch.write_text("--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-before\n+after\n")
    bootstrap_bfcl._run_git(["apply", "--check", str(patch)], cwd=tree)
    bootstrap_bfcl._run_git(["apply", str(patch)], cwd=tree)
    assert (tree / "file.txt").read_text() == "after\n"


def test_excluded_upstream_artifact_is_verified_before_discard(tmp_path):
    stream = io.BytesIO()
    raw = b"obsolete benchmark data"
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("obsolete.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    excluded = {"obsolete.json": {"sha256": hashlib.sha256(raw).hexdigest(), "reason": "unused"}}
    bootstrap_bfcl.extract_archive(stream.getvalue(), tmp_path, set(), excluded)
    assert not (tmp_path / "obsolete.json").exists()
    excluded["obsolete.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Excluded upstream"):
        bootstrap_bfcl.extract_archive(stream.getvalue(), tmp_path, set(), excluded)
