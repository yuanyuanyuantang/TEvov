"""Per-turn memory refresh hook: single-block replacement, frozen query rule, fail-closed.

CPU-only; the LLM/env/memory HTTP surfaces are monkeypatched exactly like
tests/test_bfcl_agent_reproducibility.py.  These tests pin the two-arm trial's
pre-implementation locks: first-turn arms stay byte-identical, per-turn arms
replace the ONE memory block in the first message (no accumulation), the refresh
query is the verbatim cumulative user-text prefix (no injected memory, no future
user messages, no completion markers), and a lookup miss is an error, not a
silent wrong-turn context.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = Path(os.environ.get("TEVO_BFCL_ROOT", ROOT / "benchmark/bfcl/gorilla/berkeley-function-call-leaderboard"))
sys.path.insert(0, str(ROOT / "benchmark/bfcl"))
if BFCL_ROOT.exists():
    sys.path.insert(0, str(BFCL_ROOT))
import bfcl_agent as module
from tevo.reproducibility import canonical_hash

Agent = module.LocalBFCLAgent

HEADER = "Some Related Experience to help you to complete the task:\n"


def make_agent(tmp_path, monkeypatch, **overrides):
    for name in ("TEVO_LLM_SEED", "TEVO_INJECT_TOOL_PROFILES", "TEVO_ALLOW_VAL_OBSERVER_LEARNING",
                 "TEVO_DEFER_TRAIN_OBSERVER_LEARNING", "TEVO_DISABLE_MEMORY_USAGE_FEEDBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEVO_DISABLE_MEMORY_USAGE_FEEDBACK", "1")
    monkeypatch.setenv("TEVO_COMPACT_BFCL_RESULTS", "0")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: SimpleNamespace())
    data = tmp_path / "val.jsonl"
    ids = overrides.pop("ids", ["multi_turn_base_1"])
    rows = overrides.pop("rows", None)
    if rows is None:
        rows = [{"id": task_id, "messages": [{"role": "user", "content": "goal " + task_id}],
                 "tools": [], "extra": {"id": task_id, "involved_classes": []}} for task_id in ids]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    kwargs = dict(index=0, task_ids=ids, experiment_name="bfcl-multi-turn-base-val_perturn",
                  data_path=str(data), num_trials=1)
    kwargs.update(overrides)
    return Agent(**kwargs)


def run_agent(agent, monkeypatch, env_outputs):
    prompts = []
    monkeypatch.setattr(agent, "call_llm", lambda messages, schema: prompts.append(
        [dict(message) for message in messages]) or {"role": "assistant", "content": "noop"})
    outputs = iter(env_outputs)
    monkeypatch.setattr(agent, "env_step", lambda *args: next(outputs))
    monkeypatch.setattr(agent, "get_reward", lambda *args: 0.0)
    monkeypatch.setattr(agent, "_reset_bfcl_instance_cache", lambda *args, **kwargs: None)
    entry = agent.execute()[0]
    return entry, prompts


def test_flag_off_is_byte_identical_and_never_refreshes(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, use_memory=True, max_interactions=6)
    retrievals = []

    def retrieve(run_id, task_index, query=None):
        retrievals.append(query)
        return "FROZEN-BLOCK"

    monkeypatch.setattr(agent, "_retrieve_tevo_memory_context", retrieve)
    entry, prompts = run_agent(agent, monkeypatch, [
        {"messages": [{"role": "user", "content": "second goal"}]},
        {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]},
    ])
    expected = "Task:\ngoal multi_turn_base_1\n\n" + HEADER + "FROZEN-BLOCK"
    assert retrievals == [None]  # one retrieval, first turn only, no query override
    assert [row[0]["content"] for row in prompts] == [expected, expected]
    assert entry["memory_injection_log"] == [{
        "turn": 0,
        "query_sha256": canonical_hash("goal multi_turn_base_1"),
        "context_sha256": canonical_hash("FROZEN-BLOCK"),
        "replaced_previous_block": False,
        "first_message_content_sha256": canonical_hash(expected),
    }]


def test_per_turn_refreshes_replace_single_block_with_cumulative_user_query(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, use_memory=True, per_turn_memory=True, max_interactions=6)
    queries = []

    def retrieve(run_id, task_index, query=None):
        queries.append(query)
        return "BLOCK-0" if query is None else "BLOCK:" + query

    monkeypatch.setattr(agent, "_retrieve_tevo_memory_context", retrieve)
    entry, prompts = run_agent(agent, monkeypatch, [
        {"messages": [{"role": "user", "content": "second goal"}]},
        {"messages": [{"role": "user", "content": "third goal"}]},
        {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]},
    ])
    assert queries[0] is None  # first turn keeps the historical no-query path
    assert queries[1] == "goal multi_turn_base_1\nsecond goal"
    assert queries[2] == "goal multi_turn_base_1\nsecond goal\nthird goal"
    contents = [row[0]["content"] for row in prompts]
    assert contents[0] == "Task:\ngoal multi_turn_base_1\n\n" + HEADER + "BLOCK-0"
    assert contents[1] == "Task:\ngoal multi_turn_base_1\n\n" + HEADER + "BLOCK:goal multi_turn_base_1\nsecond goal"
    assert contents[2] == ("Task:\ngoal multi_turn_base_1\n\n" + HEADER
                           + "BLOCK:goal multi_turn_base_1\nsecond goal\nthird goal")
    for content in contents:
        assert content.count("Some Related Experience") == 1  # the single block, never 5 x turns
        assert content.count("Task:") == 1  # the stored verbatim task text, never nested re-wrapping
    for query in queries[1:]:
        assert "BLOCK" not in query and "Some Related Experience" not in query
        assert "[CONVERSATION_COMPLETED]" not in query
    log = entry["memory_injection_log"]
    assert [row["turn"] for row in log] == [0, 1, 2]
    assert [row["replaced_previous_block"] for row in log] == [False, True, True]
    assert log[1]["query_sha256"] == canonical_hash("goal multi_turn_base_1\nsecond goal")
    assert log[2]["query_sha256"] == canonical_hash("goal multi_turn_base_1\nsecond goal\nthird goal")
    assert [row["first_message_content_sha256"] for row in log] == [canonical_hash(content) for content in contents]


def test_tool_only_env_steps_do_not_count_as_user_turns(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, use_memory=True, per_turn_memory=True, max_interactions=6)
    queries = []

    def retrieve(run_id, task_index, query=None):
        queries.append(query)
        return "BLOCK-0" if query is None else "BLOCK:" + query

    monkeypatch.setattr(agent, "_retrieve_tevo_memory_context", retrieve)
    entry, prompts = run_agent(agent, monkeypatch, [
        {"messages": [{"role": "tool", "content": "result", "tool_call_id": "fake"}]},
        {"messages": [{"role": "user", "content": "second goal"}]},
        {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]},
    ])
    assert queries == [None, "goal multi_turn_base_1\nsecond goal"]
    assert [row["turn"] for row in entry["memory_injection_log"]] == [0, 1]


def test_frozen_lookup_miss_is_an_error_not_a_wrong_turn_context(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch, use_memory=True, per_turn_memory=True, max_interactions=6)
    frozen = {"goal multi_turn_base_1": "BLOCK-0"}

    def retrieve(run_id, task_index, query=None):
        if query is None:
            return frozen["goal multi_turn_base_1"]
        if query not in frozen:
            raise ValueError(f"per-turn query is not in the frozen per-turn contexts: {query[:160]!r}")
        return frozen[query]

    monkeypatch.setattr(agent, "_retrieve_tevo_memory_context", retrieve)
    entry, _ = run_agent(agent, monkeypatch, [
        {"messages": [{"role": "user", "content": "divergent runtime text"}]},
        {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]},
    ])
    assert entry["reward"] == 0.0
    assert "per-turn query is not in the frozen per-turn contexts" in entry["error"]
    assert [row["turn"] for row in entry["memory_injection_log"]] == [0]


def test_per_turn_memory_requires_use_memory():
    with pytest.raises(ValueError, match="per_turn_memory requires use_memory"):
        Agent(index=0, task_ids=["multi_turn_base_1"], experiment_name="bfcl-train",
              data_path="unused.jsonl", per_turn_memory=True)


def test_init_state_requires_verbatim_first_user_message(tmp_path, monkeypatch):
    rows = [{"id": "multi_turn_base_1", "messages": [], "tools": [],
             "extra": {"id": "multi_turn_base_1", "involved_classes": []}}]
    with pytest.raises(ValueError, match="verbatim first user message"):
        make_agent(tmp_path, monkeypatch, use_memory=True, per_turn_memory=True, rows=rows)
    # Without the per-turn flag the legacy path accepts the same task shape.
    agent = make_agent(tmp_path, monkeypatch, use_memory=True, rows=rows)
    assert agent.user_turn_texts == [[[]]]