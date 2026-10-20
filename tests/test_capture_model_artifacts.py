"""Disk provenance must reject missing shards, concurrent writes and overwrite."""

import hashlib
import json

import pytest

from scripts import capture_model_artifacts as module


@pytest.fixture
def model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"fixture"}\n')
    (root / "tokenizer.json").write_text('{"vocab":{}}\n')
    (root / "model-00001-of-00001.safetensors").write_bytes(b"fixture-weights")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}))
    return root


def test_capture_hashes_relative_runtime_assets_and_publishes_once(model, tmp_path):
    (model / "README.md").write_text("incidental documentation")
    output = tmp_path / "artifacts.json"
    manifest = module.capture_to_file(model, "fixture-model", output)
    assert len(manifest["files"]) == 4 and "README.md" not in manifest["files"]
    assert manifest["live_gpu_weights_verified"] is False
    for name, record in manifest["files"].items():
        content = (model / name).read_bytes()
        assert record == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    assert json.loads(output.read_text()) == manifest
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="exists"):
        module.capture_to_file(tmp_path / "nonexistent-model", "fixture-model", output)
    assert output.read_bytes() == before


def test_capture_rejects_missing_or_unsafe_index_shards(model):
    shard = model / "model-00001-of-00001.safetensors"
    shard.unlink()
    with pytest.raises(ValueError, match="Missing referenced shard"):
        module.capture(model, "fixture-model")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../outside.bin"}}))
    with pytest.raises(ValueError, match="Unsafe shard"):
        module.capture(model, "fixture-model")


def test_capture_detects_mutation_while_hashing_and_leaves_no_output(model, tmp_path, monkeypatch):
    real_hash = module.sha256_file

    def changing_hash(path):
        digest = real_hash(path)
        path.write_bytes(path.read_bytes() + b"changed")
        return digest

    monkeypatch.setattr(module, "sha256_file", changing_hash)
    output = tmp_path / "artifacts.json"
    with pytest.raises(ValueError, match="changed while hashing"):
        module.capture_to_file(model, "fixture-model", output)
    assert not output.exists()


def test_capture_detects_late_inventory_addition(model, monkeypatch):
    real_hash = module.sha256_file

    def adding_file(path):
        digest = real_hash(path)
        (model / "added_tokens.json").write_text('{}\n')
        return digest

    monkeypatch.setattr(module, "sha256_file", adding_file)
    with pytest.raises(ValueError, match="inventory changed"):
        module.capture(model, "fixture-model")
