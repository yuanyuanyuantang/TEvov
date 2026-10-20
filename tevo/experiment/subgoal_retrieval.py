"""Bounded first-query fragments and fixed MMR; bank records stay unchanged."""

import math
import re

from tevo.memory.embedding import cosine_similarity


def split_queries(query: str) -> list[str]:
    """Keep the full query and literal fragments, never inferred subgoals."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("a nonempty first-user query is required")
    pieces = re.split(r"[.!?](?=\s|$)|;|\b(?:and|then|once|before|after|also)\b", query,
                      flags=re.IGNORECASE)
    queries = [query]
    for piece in pieces:
        piece = piece.strip()
        if len(piece) >= 12 and piece not in queries:
            queries.append(piece)
    if len(queries) - 1 > 64:
        raise ValueError("more than 64 first-query fragments")
    return queries


def select_subgoal_memories(records: list[dict], index: dict,
                            query_vectors: list[list[float]], *,
                            available_tools: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Top20 union, max relevance, 0.7/0.3 MMR, at most five text-unique slots.

    Callers validate the frozen bank/index first, as for whole-query retrieval.
    With explicit available_tools, filter incompatible tags and weight pairwise
    redundancy by tag Jaccard. Without it the original prototype is unchanged.
    Tags are a coarse overlap signal, not a proof of procedural dependence.
    The trace is private metadata; the existing formatter controls actor text.
    """
    from tevo.experiment.suite import rank_memories

    if available_tools is not None and (
        not isinstance(available_tools, (set, frozenset)) or not available_tools
        or any(not isinstance(t, str) or not t.strip() for t in available_tools)
    ):
        raise ValueError("available_tools must be a nonempty set of tool names")
    dimensions = {len(item["vector"]) for item in index.values()}
    if not query_vectors or not dimensions or len(dimensions) != 1:
        raise ValueError("query vectors and a consistent nonempty index are required")
    dimension = next(iter(dimensions))
    if not dimension or any(len(v) != dimension or not all(math.isfinite(x) for x in v)
                            for v in query_vectors):
        raise ValueError("query vector dimension or finite-value mismatch")
    pool = {}
    for vector in query_vectors:
        for row in rank_memories(records, index, vector, 20, "tevo"):
            if available_tools is not None:
                tags = row.get("tool_tags")
                if (not isinstance(tags, list) or not tags
                    or any(not isinstance(t, str) or not t for t in tags)
                    or not set(tags) <= available_tools):
                    continue
            pool[row["id"]] = row
    # ponytail: quadratic scan is bounded by the small frozen bank; no ANN service.
    candidates = [identity for identity in index if identity in pool]
    relevance = {i: max(cosine_similarity(q, index[i]["vector"]) for q in query_vectors)
                 for i in candidates}
    selected, trace, seen = [], [], set()
    while candidates and len(selected) < 5:
        scores = {}
        for identity in candidates:
            penalties = [0.0]
            for previous in selected:
                overlap = 1.0
                if available_tools is not None:
                    left, right = set(pool[identity]["tool_tags"]), set(previous["tool_tags"])
                    overlap = len(left & right) / len(left | right)
                penalties.append(overlap * cosine_similarity(index[identity]["vector"],
                                                              index[previous["id"]]["vector"]))
            redundancy = max(penalties)
            scores[identity] = (0.7 * relevance[identity] - 0.3 * redundancy, redundancy)
        identity = max(candidates, key=lambda i: scores[i][0])
        candidates.remove(identity)
        row = pool[identity]
        text_key = tuple(" ".join(row[key].split()) for key in ("when_to_use", "content"))
        if text_key in seen:
            continue
        seen.add(text_key)
        selected.append({**row, "similarity": round(relevance[identity], 6)})
        trace.append({"id": identity, "relevance": relevance[identity],
                      "redundancy": scores[identity][1], "mmr": scores[identity][0]})
    return selected, trace
