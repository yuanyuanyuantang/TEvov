import json

import pytest

from tevo.io_utils import atomic_write_text
from tevo.memory.procedural_store import ProceduralMemoryStore
from test_memory_record import FakeEmbedding, _make_record


def test_invalid_bank_never_silently_becomes_empty_or_overwrites_source(tmp_path):
    directory = tmp_path / "memory"
    directory.mkdir()
    source = directory / "memories.json"
    source.write_text('{"truncated":')
    original = source.read_bytes()
    with pytest.raises(ValueError, match="Invalid memory records"):
        ProceduralMemoryStore(directory, FakeEmbedding())
    assert source.read_bytes() == original


def test_mismatched_generation_of_records_and_vectors_is_rejected(tmp_path):
    store = ProceduralMemoryStore(tmp_path, FakeEmbedding())
    store.add(_make_record("a", "condition", "action"))
    index = json.loads(store.index_path.read_text())
    index["a"]["search_text"] = "old generation"
    store.index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="text mismatch"):
        ProceduralMemoryStore(tmp_path, FakeEmbedding())


def test_failed_atomic_replace_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "bank.json"
    atomic_write_text(path, "original")
    def fail_replace(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr("tevo.io_utils.os.replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_text(path, "replacement")
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]
