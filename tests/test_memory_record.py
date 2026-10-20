"""MemoryRecord and its single-file store, using deterministic test embeddings."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tevo.memory.procedural_schema import (
    CONFIDENCE_FROM_FAILURE,
    CONFIDENCE_FROM_SUCCESS,
    DEFAULT_TAG,
    STALE_THRESHOLD,
    MemoryRecord,
    normalize_tag,
)
from tevo.memory.procedural_store import ProceduralMemoryStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

_DIM = 64


class FakeEmbedding:
    """确定性词袋 embedding：token 哈希到固定维度。"""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * _DIM
        tokens = Counter(
            tok for tok in "".join(
                c.lower() if c.isalnum() else " " for c in str(text)
            ).split()
            if tok
        )
        for token, count in tokens.items():
            slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % _DIM
            vector[slot] += float(count)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]




def _make_record(
    record_id: str,
    when_to_use: str,
    content: str,
    *,
    type_tag: str = "failure_repair",
    tool_tags: list[str] | None = None,
    confidence: float = CONFIDENCE_FROM_SUCCESS,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        type_tag=type_tag,
        tool_tags=list(tool_tags or []),
        confidence=confidence,
        when_to_use=when_to_use,
        content=content,
        source="manual",
        source_task="t_test",
    )


@pytest.fixture()
def store(tmp_path) -> ProceduralMemoryStore:
    return ProceduralMemoryStore(tmp_path / "memory", FakeEmbedding())




# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

def test_normalize_tag_known_and_legacy_names():
    assert normalize_tag("failure_repair") == "failure_repair"
    assert normalize_tag("FailureRepairRecord") == "failure_repair"
    assert normalize_tag("nonsense") == DEFAULT_TAG
    assert normalize_tag("") == DEFAULT_TAG


def test_record_roundtrip_and_render():
    record = _make_record("mem_1", "When cd fails with No such directory", "mkdir first, then cd", tool_tags=["cd", "mkdir"])
    data = record.to_dict()
    restored = MemoryRecord.from_dict(json.loads(json.dumps(data)))
    assert restored.record_id == "mem_1"
    assert restored.tool_tags == ["cd", "mkdir"]
    assert restored.search_text == record.search_text
    markdown = record.to_markdown()
    assert "[failure_repair | tools: cd, mkdir]" in markdown
    assert "When to use:" in markdown


def test_search_text_is_the_scenario_sentence_only():
    """v6：检索键 = 仅 when_to_use 场景句；策略正文不进检索文本（ReMe 对齐）。"""
    record = _make_record("mem_1", "When a directory move fails", "create the directory with mkdir first", tool_tags=["cd"])
    assert record.search_text == "When a directory move fails"
    assert "mkdir first" not in record.search_text


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_store_add_retrieve_and_persistence(tmp_path):
    store = ProceduralMemoryStore(tmp_path / "mem", FakeEmbedding())
    assert store.add(_make_record("mem_a", "cd fails with No such directory", "create the directory with mkdir first", tool_tags=["cd", "mkdir"]))
    assert store.total_count() == 1

    hits = store.retrieve(None, "cd fails with No such directory", top_k=3)
    assert [r.record_id for r in hits] == ["mem_a"]

    # tag 过滤
    assert store.retrieve("failure_repair", "cd fails with No such directory", top_k=3)
    assert store.retrieve("success_trajectory", "cd fails with No such directory", top_k=3) == []

    # 重新加载后仍在（单文件持久化）
    store2 = ProceduralMemoryStore(tmp_path / "mem", FakeEmbedding())
    assert store2.total_count() == 1
    assert store2.get("mem_a").content.startswith("create the directory")


def test_store_dedup_merges_near_duplicates(store):
    store.add(_make_record("mem_a", "cd fails with No such directory", "create the directory with mkdir first", confidence=0.5))
    added = store.add(_make_record("mem_b", "cd fails with No such directory", "create the directory with mkdir first", tool_tags=["cd"], confidence=0.6))
    assert added is False
    assert store.total_count() == 1
    merged = store.get("mem_a")
    # 未知episode的legacy事件不伪称独立证据，也不增加merge confidence。
    assert merged.confidence == pytest.approx(0.6)
    assert "cd" in merged.tool_tags
    assert merged.metadata["evidence_count"] == 0
    assert merged.metadata["legacy_evidence_events"] == 2
    assert merged.metadata["evidence_count_unit"] == "distinct_episode_ids_unknown"
    assert "mem_b" in merged.metadata["merged_from"]


def test_store_dedup_keeps_different_notes(store):
    store.add(_make_record("mem_a", "cd fails with No such directory", "create the directory with mkdir first"))
    assert store.add(_make_record("mem_b", "echo writes text into a file", "pass content and file_name to echo", type_tag="parameter_template", tool_tags=["echo"]))
    assert store.total_count() == 2


def test_store_attribution_lifecycle(store):
    store.add(_make_record("mem_a", "cd fails", "mkdir first", confidence=0.6))
    store.add(_make_record("mem_b", "echo fails", "check file_name", confidence=0.25))

    # exposure：只计数不动 confidence
    assert store.record_exposure_any("mem_a") == "failure_repair"
    assert store.get("mem_a").usage_count == 1
    assert store.get("mem_a").confidence == pytest.approx(0.6)

    # credited：+0.05，success_count+1
    store.update_on_use_any("mem_a", True)
    assert store.get("mem_a").confidence == pytest.approx(0.65)
    assert store.get("mem_a").success_count == 1

    # blamed：-0.1；跌破 0.2 → stale，检索排除
    store.update_on_use_any("mem_b", False)
    store.update_on_use_any("mem_b", False)
    assert store.get("mem_b").confidence == pytest.approx(0.05)
    assert store.get("mem_b").stale is True
    hits = store.retrieve(None, "echo fails", top_k=5)
    assert all(r.record_id != "mem_b" for r in hits)

    # 物理清理
    assert store.cleanup_stale() == 1
    assert store.get("mem_b") is None


def test_store_cleanup_low_utility(store):
    record = _make_record("mem_a", "cd fails", "mkdir first")
    store.add(record)
    for _ in range(5):
        store.record_exposure_any("mem_a")
    # 5 次曝光 0 次成功 → utility 0 <= 0.5 → 删除
    assert store.cleanup_low_utility(freq_threshold=5, utility_threshold=0.5) == 1
    assert store.total_count() == 0


def test_store_retrieve_all_types_buckets_by_tag(store):
    store.add(_make_record("mem_a", "cd fails with No such directory", "mkdir first", type_tag="failure_repair"))
    store.add(_make_record("mem_b", "task needs a new directory then a file", "mkdir then touch then echo", type_tag="success_trajectory"))
    buckets = store.retrieve_all_types("cd fails with No such directory", top_k_per_type=2)
    assert buckets["failure_repair"][0].record_id == "mem_a"
    assert set(buckets.keys()) >= {"failure_repair", "success_trajectory", "generic_rule"}
