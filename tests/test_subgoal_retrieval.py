"""Checks that constrain the finite retrieval prototype, with no network."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from tevo.experiment.subgoal_retrieval import select_subgoal_memories, split_queries


def test_literal_split_preserves_decimal_and_late_subgoal_and_bounds():
    query = "Add 15.5 liters and check pressures. Then start the engine; also read battery voltage."
    queries = split_queries(query)
    assert queries[0] == query
    assert "Add 15.5 liters" in queries
    assert "read battery voltage" in queries
    assert all(q in query for q in queries)
    assert split_queries("short") == ["short"]
    assert len(split_queries("read the status and read the status")) == 2
    with pytest.raises(ValueError, match="64"):
        split_queries(" and ".join(f"read distinct status {i}" for i in range(65)))


def test_mmr_relevance_diversity_dedup_ties_and_immutability():
    index = {"a": {"vector": [1., 0.]}, "duplicate": {"vector": [1., 0.]},
             "close": {"vector": [.99, .01]}, "b": {"vector": [0., 1.]}}
    rows = {i: {"id": i, "when_to_use": "scene " + i, "content": "instruction " + i}
            for i in index}
    rows["duplicate"].update(when_to_use="scene  a", content="instruction  a")
    original = deepcopy((index, rows))
    calls = []

    def rank(records, bank_index, vector, k, method):
        calls.append((vector, k, method))
        return list(rows.values())

    with patch("tevo.experiment.suite.rank_memories", side_effect=rank):
        chosen, trace = select_subgoal_memories([], index, [[1., 0.], [0., 1.]])
    assert [r["id"] for r in chosen] == ["a", "b", "close"]
    assert calls == [([1., 0.], 20, "tevo"), ([0., 1.], 20, "tevo")]
    assert trace[0]["mmr"] == .7 and trace[1]["redundancy"] == 0.
    assert (index, rows) == original
    for invalid in ([], [[1.]], [[float("nan"), 0.]]):
        with pytest.raises(ValueError):
            select_subgoal_memories([], index, invalid)


def test_five_slot_cap_and_negative_similarity_redundancy_floor():
    rows = [{"id": str(i), "when_to_use": str(i), "content": str(i)} for i in range(8)]
    index = {r["id"]: {"vector": [1., 0.] if i % 2 == 0 else [-1., 0.]}
             for i, r in enumerate(rows)}
    with patch("tevo.experiment.suite.rank_memories", return_value=rows):
        chosen, trace = select_subgoal_memories([], index, [[1., 0.], [-1., 0.]])
    assert len(chosen) == 5 and [r["id"] for r in chosen[:2]] == ["0", "1"]
    assert trace[1]["redundancy"] == 0.


def test_tool_filter_rejects_unavailable_mixed_missing_and_malformed_tags():
    tags = [["read"], ["book"], ["read", "book"], [], None, [None]]
    rows = [{"id": str(i), "when_to_use": str(i), "content": str(i), "tool_tags": t}
            for i, t in enumerate(tags)]
    index = {r["id"]: {"vector": [1., 0.]} for r in rows}
    with patch("tevo.experiment.suite.rank_memories", return_value=rows):
        selected, _ = select_subgoal_memories([], index, [[1., 0.]], available_tools={"read"})
        assert [r["id"] for r in selected] == ["0"]
        assert len(select_subgoal_memories([], index, [[1., 0.]])[0]) == 5
    for invalid in (set(), ["read"], {None}, {""}):
        with pytest.raises(ValueError, match="available_tools"):
            select_subgoal_memories([], index, [[1., 0.]], available_tools=invalid)


def test_complementary_tool_tags_do_not_penalize_each_other():
    rows = [{"id": i, "when_to_use": i, "content": i, "tool_tags": [tag]}
            for i, tag in [("a", "open"), ("same", "open"), ("b", "confirm")]]
    index = {r["id"]: {"vector": [1., 0.]} for r in rows}
    with patch("tevo.experiment.suite.rank_memories", return_value=rows):
        selected, trace = select_subgoal_memories([], index, [[1., 0.]],
                                                  available_tools={"open", "confirm"})
    assert [r["id"] for r in selected] == ["a", "b", "same"]
    assert trace[1]["redundancy"] == 0. and trace[2]["redundancy"] == 1.


def test_partial_tool_overlap_uses_jaccard_and_preserves_records():
    rows = [{"id": "a", "when_to_use": "a", "content": "a", "tool_tags": ["open"]},
            {"id": "b", "when_to_use": "b", "content": "b", "tool_tags": ["open", "confirm"]}]
    index = {r["id"]: {"vector": [1., 0.]} for r in rows}
    original = deepcopy((rows, index))
    with patch("tevo.experiment.suite.rank_memories", return_value=rows):
        _, trace = select_subgoal_memories([], index, [[1., 0.]], available_tools={"open", "confirm"})
    assert trace[1]["redundancy"] == .5
    assert trace[1]["mmr"] == pytest.approx(.7 - .3 * .5)
    assert (rows, index) == original
