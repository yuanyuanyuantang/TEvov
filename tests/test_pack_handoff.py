"""The collaboration package includes inputs and rejects links/secrets/overwrite."""
import json
import zipfile

import pytest

from scripts.pack_handoff import pack


def test_handoff_includes_inputs_and_excludes_history(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    names = ["README.md", "HANDOFF.md", "pyproject.toml", "tevo/module.py",
             "artifacts/train_trajectories/trial.json", "benchmark/bfcl/data/val.jsonl",
             ".env.example", ".env", "outputs/private.json", ".git/config", "models/weights.bin",
             "scripts/__pycache__/cached.pyc", "benchmark/bfcl/logs/private.json"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    output = tmp_path / "handoff.zip"
    result = pack(root, output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("TEvov/HANDOFF_MANIFEST.json"))
        assert result["file_count"] == manifest["file_count"] == 7
        assert "artifacts/train_trajectories/trial.json" in manifest["files"]
        assert "outputs/private.json" not in manifest["files"]
    with pytest.raises(FileExistsError):
        pack(root, output)
    with pytest.raises(ValueError, match="outside"):
        pack(root, root / "nested.zip")


def test_handoff_refuses_credentials_and_symlinks(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in ["README.md", "HANDOFF.md", "pyproject.toml"]:
        (root / name).write_text("entry")
    secret = root / "scripts/private.py"
    secret.parent.mkdir()
    secret.write_text("sk-" + "a" * 32)
    output = tmp_path / "handoff.zip"
    with pytest.raises(ValueError, match="credential"):
        pack(root, output)
    assert not output.exists()
    secret.unlink()
    secret.symlink_to(root / "README.md")
    with pytest.raises(ValueError, match="linked"):
        pack(root, output)
