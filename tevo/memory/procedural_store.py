"""Procedural Memory Bank — 单文件 JSON 持久化 + 向量索引存储。

2026-09 重构：7 类记录塌缩为统一 MemoryRecord；按类型分文件改为单文件
（memories.json + memories_index.json）。类型只是 tag，检索时可按 tag 过滤。

生命周期规则（信封字段，全部代码维护）：
- exposure（被检索展示）→ usage_count += 1
- credited（归因命中且任务成功）→ confidence += 0.05（封顶 1.0），success_count += 1
- blamed（归因命中但任务失败）→ confidence -= 0.1
- confidence < 0.2 → stale，检索排除；cleanup_stale 定期物理清理
- 去重：入库时 cosine >= dedup_threshold(默认0.85) → 合并进已有记录
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from tevo.io_utils import atomic_write_text
from tevo.memory.embedding import EmbeddingModelProtocol, cosine_similarity
from tevo.memory.procedural_schema import (
    CONFIDENCE_INCREMENT,
    CONFIDENCE_PENALTY,
    MEMORY_TAGS,
    STALE_THRESHOLD,
    MemoryRecord,
)

_DEFAULT_DEDUP_THRESHOLD = 0.85


class ProceduralMemoryStore:
    """统一管理程序性记忆的本地存储（单文件 + 单一向量索引）。"""

    def __init__(self, base_dir: str | Path, embedding_model: EmbeddingModelProtocol) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self._query_vector_cache: dict[str, list[float]] = {}
        # 并发保护：多个 worker 会同时检索/归因/入库（服务端线程池），内存字典
        # 与落盘都不是线程安全的。RLock 允许同类方法内部互相调用。
        self._lock = threading.RLock()

        self._records: dict[str, MemoryRecord] = {}   # record_id -> record（保持插入序）
        self._index: dict[str, dict[str, Any]] = {}   # record_id -> {search_text, vector}
        self._load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        """供调用方在需要跨多次写操作保持一致（如全库归并）的场景下复用。"""
        return self._lock

    @property
    def records_path(self) -> Path:
        return self.base_dir / "memories.json"

    @property
    def index_path(self) -> Path:
        return self.base_dir / "memories_index.json"

    def _load(self) -> None:
        if self.records_path.exists():
            try:
                raw = json.loads(self.records_path.read_text(encoding="utf-8"))
                if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
                    raise ValueError("memory records must be a list of objects")
                for item in raw:
                    record = MemoryRecord.from_dict(item)
                    if not record.record_id or record.record_id in self._records:
                        raise ValueError("missing or duplicate memory record id")
                    self._records[record.record_id] = record
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid memory records: {self.records_path}; restore a verified snapshot") from exc
        if self.index_path.exists():
            try:
                self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
                if not isinstance(self._index, dict):
                    raise ValueError("memory index must be an object")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid memory index: {self.index_path}; restore a verified snapshot") from exc
        # Each file is atomically replaced, but a crash can occur between the two
        # replacements. Fail closed instead of silently dropping memories or
        # retrieving vectors from a different bank generation.
        if set(self._index) != set(self._records):
            raise ValueError("Memory records/index IDs differ; restore a verified snapshot or explicitly rebuild the index")
        dimensions = set()
        for record_id, item in self._index.items():
            if not isinstance(item, dict) or item.get("search_text") != self._records[record_id].search_text:
                raise ValueError(f"Memory index text mismatch: {record_id}")
            vector = item.get("vector")
            if not isinstance(vector, list) or not vector or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector
            ):
                raise ValueError(f"Invalid memory vector: {record_id}")
            dimensions.add(len(vector))
        if len(dimensions) > 1:
            raise ValueError("Memory index contains inconsistent vector dimensions")

    def _save(self) -> None:
        with self._lock:
            atomic_write_text(
                self.records_path,
                json.dumps([r.to_dict() for r in self._records.values()], ensure_ascii=False, indent=2),
            )
            atomic_write_text(
                self.index_path,
                json.dumps(self._index, ensure_ascii=False),
            )

    def save_all(self) -> None:
        self._save()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add(self, record: MemoryRecord | dict[str, Any], *, dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD) -> bool:
        """添加一条记忆。相似记录（cosine >= threshold）合并而非新增。

        返回 True 表示新增，False 表示合并到已有记录。
        """
        if isinstance(record, dict):
            record = MemoryRecord.from_dict(record)
        if not isinstance(record, MemoryRecord):
            raise ValueError(f"Unsupported record type: {type(record).__name__}")
        if not record.search_text.strip():
            return False

        with self._lock:
            if self._records:
                existing = self._find_duplicate(record, dedup_threshold)
                if existing is not None:
                    self._merge_record(existing, record)
                    return False

            self._records[record.record_id] = record
            self._index_record(record)
            self._save()
            return True

    def remove(self, record: MemoryRecord | dict[str, Any]) -> bool:
        record_id = record.get("record_id") if isinstance(record, dict) else getattr(record, "record_id", None)
        with self._lock:
            if not record_id or record_id not in self._records:
                return False
            del self._records[record_id]
            self._index.pop(record_id, None)
            self._save()
            return True

    def _index_record(self, record: MemoryRecord) -> None:
        search_text = record.search_text
        vector = self.embedding_model.embed(search_text)
        self._index[record.record_id] = {"search_text": search_text, "vector": vector}

    def _find_duplicate(self, record: MemoryRecord, threshold: float) -> MemoryRecord | None:
        query_vector = self.embedding_model.embed(record.search_text)
        best: MemoryRecord | None = None
        best_sim = -1.0
        for record_id, item in self._index.items():
            vector = item.get("vector", [])
            if not vector:
                continue
            similarity = cosine_similarity(query_vector, [float(v) for v in vector])
            if similarity >= threshold and similarity > best_sim:
                existing = self._records.get(record_id)
                if existing is not None:
                    new_meta, old_meta = record.metadata or {}, existing.metadata or {}
                    if "generated_text_sha256" in new_meta or "generated_text_sha256" in old_meta:
                        # Similar wording is not evidence that different goal
                        # quantities, contracts or observed procedures agree.
                        fields = ("claim_kind", "goal_sha256", "static_contract_sha256", "generated_text_sha256")
                        if any(key not in new_meta or key not in old_meta or new_meta[key] != old_meta[key] for key in fields):
                            continue
                    best, best_sim = existing, similarity
        return best

    def _merge_record(self, existing: MemoryRecord, new: MemoryRecord) -> None:
        """合并兼容观察；新增不同episode才增加独立证据和merge confidence。"""
        with self._lock:
            self._merge_record_locked(existing, new)

    def _merge_record_locked(self, existing: MemoryRecord, new: MemoryRecord) -> None:
        metadata = dict(existing.metadata or {})
        new_metadata = new.metadata or {}
        old_episodes = {value for value in metadata.get("evidence_episode_ids", []) if isinstance(value, str) and value}
        new_episodes = {value for value in new_metadata.get("evidence_episode_ids", []) if isinstance(value, str) and value}
        if old_episodes or new_episodes:
            episodes = old_episodes | new_episodes
            metadata["evidence_episode_ids"] = sorted(episodes)
            metadata["evidence_count"] = len(episodes)
            metadata["evidence_count_unit"] = "distinct_episode_ids"
            if new_episodes - old_episodes:
                existing.confidence = min(max(existing.confidence, new.confidence) + CONFIDENCE_INCREMENT, 1.0)
            refs = metadata.get("evidence_refs", []) + new_metadata.get("evidence_refs", [])
            by_ref = {(ref["episode_id"], ref["turn_id"], ref["call_index"]): ref for ref in refs}
            metadata["evidence_refs"] = list(by_ref.values())
        else:
            # Legacy counters lack an episode identity: preserve event counts
            # separately, and do not pretend another merge is independent data.
            metadata["legacy_evidence_events"] = int(metadata.get("legacy_evidence_events", metadata.get("evidence_count", 1)) or 1) + int(new_metadata.get("evidence_count", 1) or 1)
            metadata["evidence_count"] = 0
            metadata["evidence_count_unit"] = "distinct_episode_ids_unknown"
            existing.confidence = max(existing.confidence, new.confidence)
        existing.usage_count += new.usage_count
        existing.success_count += new.success_count
        for tool in new.tool_tags:
            if tool not in existing.tool_tags:
                existing.tool_tags.append(tool)
        metadata["source_task_ids"] = sorted(set(metadata.get("source_task_ids", [])) | {existing.source_task, new.source_task})
        merged_from = set(str(x) for x in metadata.get("merged_from", []) if str(x))
        merged_from.add(new.record_id)
        metadata["merged_from"] = sorted(merged_from)[:12]
        existing.metadata = metadata
        existing.updated_at = time.time()
        if existing.confidence >= STALE_THRESHOLD:
            existing.stale = False
        self._save()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        record_type: str | None = None,
        query: str = "",
        top_k: int = 5,
        tags: set[str] | None = None,
    ) -> list[MemoryRecord]:
        """向量召回 + confidence 重排。record_type 作为 tag 过滤（None=不过滤）。

        tags 参数保留兼容：非空时要求记录 type_tag 或 tags 元数据有交集。
        """
        if top_k <= 0 or not query:
            return []
        query_vector = self._embed_query(query)
        with self._lock:
            if not self._index:
                return []
            candidates: list[tuple[str, float, float]] = []
            for record_id, item in self._index.items():
                record = self._records.get(record_id)
                if record is None or record.stale:
                    continue
                if record_type and record_type not in ("all", "*") and record.type_tag != record_type:
                    continue
                if tags:
                    record_tags = set(record.metadata.get("tags", []) or []) | {record.type_tag}
                    if not (tags & record_tags):
                        continue
                vector = item.get("vector", [])
                if not vector:
                    continue
                similarity = cosine_similarity(query_vector, [float(v) for v in vector])
                quality = max(0.0, min(float(record.confidence), 1.0))
                candidates.append((record_id, similarity, quality))
            candidates.sort(key=lambda x: x[1], reverse=True)
            shortlist = candidates[: max(top_k * 5, top_k)]
            shortlist.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return [self._records[rid] for rid, *_ in shortlist[:top_k] if rid in self._records]

    def retrieve_all_types_scored(
        self, query: str, top_k_per_type: int = 3
    ) -> dict[str, list[tuple[MemoryRecord, float]]]:
        """一次相似度计算，按 tag 分桶取 top_k_per_type，保留相似度。

        每类取 top_k_per_type 后，各类候选里按相似度全局排序取前 K 条，等价于
        全库纯相似度 top-K（一个类别最多贡献 K 条，所以全局前 K 必在其类内前 K）。
        """
        result: dict[str, list[tuple[MemoryRecord, float]]] = {tag: [] for tag in MEMORY_TAGS}
        if top_k_per_type <= 0 or not query:
            return result
        query_vector = self._embed_query(query)
        with self._lock:
            if not self._index:
                return result
            scored: list[tuple[MemoryRecord, float]] = []
            for record_id, item in self._index.items():
                record = self._records.get(record_id)
                if record is None or record.stale:
                    continue
                vector = item.get("vector", [])
                if not vector:
                    continue
                scored.append((record, cosine_similarity(query_vector, [float(v) for v in vector])))
            scored.sort(key=lambda x: x[1], reverse=True)
            counts: dict[str, int] = {}
            for record, similarity in scored:
                tag = record.type_tag
                if counts.get(tag, 0) >= top_k_per_type:
                    continue
                result.setdefault(tag, []).append((record, similarity))
                counts[tag] = counts.get(tag, 0) + 1
            return result

    def retrieve_all_types(self, query: str, top_k_per_type: int = 3) -> dict[str, list[MemoryRecord]]:
        """同 retrieve_all_types_scored，但只返回记录（相似度丢弃）。"""
        return {
            tag: [record for record, _ in items]
            for tag, items in self.retrieve_all_types_scored(query, top_k_per_type).items()
        }

    def _embed_query(self, query: str) -> list[float]:
        key = str(query or "")
        with self._lock:
            cached = self._query_vector_cache.get(key)
        if cached is not None:
            return cached
        vector = self.embedding_model.embed(key)
        with self._lock:
            if len(self._query_vector_cache) >= 256:
                self._query_vector_cache.pop(next(iter(self._query_vector_cache)))
            self._query_vector_cache[key] = vector
        return vector

    # ------------------------------------------------------------------
    # 归因反馈 / 生命周期
    # ------------------------------------------------------------------

    def get(self, record_id: str) -> MemoryRecord | None:
        with self._lock:
            return self._records.get(str(record_id))

    def update_on_use(self, record_id: str, record_type: str | None, success: bool) -> None:
        """归因命中后的方向性反馈：credited → +0.05 / blamed → -0.1。"""
        with self._lock:
            record = self._records.get(str(record_id))
            if record is None:
                return
            delta = CONFIDENCE_INCREMENT if success else -CONFIDENCE_PENALTY
            record.confidence = max(min(record.confidence + delta, 1.0), 0.0)
            if success:
                record.success_count += 1
            if record.confidence < STALE_THRESHOLD:
                record.stale = True
            record.updated_at = time.time()
            self._save()

    def update_on_use_any(self, record_id: str, success: bool) -> str | None:
        record = self._records.get(str(record_id))
        if record is None:
            return None
        self.update_on_use(record_id, None, success)
        return record.type_tag

    def record_exposure(self, record_id: str, record_type: str | None = None) -> None:
        """被检索展示（无方向性反馈）。"""
        with self._lock:
            record = self._records.get(str(record_id))
            if record is None:
                return
            record.usage_count += 1
            metadata = dict(record.metadata or {})
            metadata["exposure_count"] = int(metadata.get("exposure_count", 0) or 0) + 1
            metadata["last_exposed_at"] = time.time()
            record.metadata = metadata
            record.updated_at = time.time()
            self._save()

    def record_exposure_any(self, record_id: str) -> str | None:
        record = self._records.get(str(record_id))
        if record is None:
            return None
        self.record_exposure(record_id)
        return record.type_tag

    def cleanup_low_utility(self, *, freq_threshold: int = 5, utility_threshold: float = 0.5) -> int:
        """ReMe 式效用删除：曝光 >= alpha 且成功率 <= beta 的记忆物理删除。"""
        alpha = max(int(freq_threshold or 0), 1)
        beta = float(utility_threshold)
        with self._lock:
            removed_ids: set[str] = set()
            for record_id, record in self._records.items():
                freq = int(record.usage_count or 0)
                if freq >= alpha:
                    rate = record.success_count / max(freq, 1)
                    if rate <= beta:
                        removed_ids.add(record_id)
            for record_id in removed_ids:
                del self._records[record_id]
                self._index.pop(record_id, None)
            if removed_ids:
                self._save()
            return len(removed_ids)

    def cleanup_stale(self, *, max_stale_age: int | None = None) -> int:
        """物理清理 stale 记忆；max_stale_age 秒内的新 stale 记录可保留。"""
        now = time.time()
        with self._lock:
            removed_ids = {
                record_id
                for record_id, record in self._records.items()
                if record.stale
                and (max_stale_age is None or now - float(record.updated_at or 0) >= max_stale_age)
            }
            for record_id in removed_ids:
                del self._records[record_id]
                self._index.pop(record_id, None)
            if removed_ids:
                self._save()
            return len(removed_ids)

    def mark_stale_below(self, record_type: str | None = None, threshold: float = STALE_THRESHOLD) -> int:
        with self._lock:
            marked = 0
            for record in self._records.values():
                if record_type and record.type_tag != record_type:
                    continue
                if record.confidence < threshold and not record.stale:
                    record.stale = True
                    marked += 1
            if marked:
                self._save()
            return marked

    # ------------------------------------------------------------------
    # 查询 / 统计
    # ------------------------------------------------------------------

    def list_all(self, record_type: str | None = None) -> list[MemoryRecord]:
        """按 tag 列出记录（含 stale）；record_type=None 返回全部。"""
        with self._lock:
            if record_type is None or record_type in ("all", "*"):
                return list(self._records.values())
            return [r for r in self._records.values() if r.type_tag == record_type]

    def count(self, record_type: str) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if r.type_tag == record_type)

    def total_count(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------
    # 快照（记忆增长曲线分析）
    # ------------------------------------------------------------------

    def save_snapshot(self, task_index: int) -> None:
        snapshot_path = self.base_dir / "snapshots.jsonl"
        with self._lock:
            snapshot = {
                "task_index": task_index,
                "total_count": self.total_count(),
                **{tag: self.count(tag) for tag in MEMORY_TAGS},
                "stale_count": sum(1 for r in self._records.values() if r.stale),
            }
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with snapshot_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    def load_snapshots(self) -> list[dict[str, Any]]:
        snapshot_path = self.base_dir / "snapshots.jsonl"
        if not snapshot_path.exists():
            return []
        snapshots = []
        with snapshot_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshots.append(json.loads(line))
        return snapshots
