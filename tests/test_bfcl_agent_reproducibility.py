"""BFCL acquisition boundaries and request accounting, without model/Ray requests."""

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
from tevo.reproducibility import canonical_hash, derive_seed

Agent = module.LocalBFCLAgent


@pytest.mark.parametrize("empty_position", [0, 1, 2])
def test_eval_preserves_empty_first_middle_and_final_task_turn(empty_position):
    history = []
    for turn in range(3):
        history.append({"role": "user", "content": f"goal {turn}"})
        if turn == empty_position:
            history.append({"role": "assistant", "content": "I cannot perform this step."})
        else:
            history.append({"role": "assistant", "tool_calls": [{"id": f"c{turn}",
                "function": {"name": "read", "arguments": json.dumps({"item": turn})}}]})
            history.append({"role": "tool", "tool_call_id": f"c{turn}", "content": "ok"})
    history.append({"role": "user", "content": "[CONVERSATION_COMPLETED]"})
    expected = [[] if i == empty_position else [f"read(item={i})"] for i in range(3)]
    assert module.extract_multi_turn_responses(history) == expected


def test_eval_keeps_call_order_across_batches_and_skips_transport_error_turn():
    history = [{"role": "user", "content": "goal"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "first", "arguments": "{}"}},
                                               {"function": {"name": "second", "arguments": "{}"}}]},
        {"role": "tool", "content": "one"}, {"role": "tool", "content": "two"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "last", "arguments": "{}"}}]},
        {"role": "tool", "content": "three"}, {"role": "user", "content": "[ERROR] internal failure"}]
    assert module.extract_multi_turn_responses(history) == [["first()", "second()", "last()"]]


def test_eval_keeps_delivered_but_unanswered_goal_without_fabricating_future_turns():
    assert module.extract_multi_turn_responses([
        {"role": "user", "content": "goal"}, {"role": "assistant", "content": "cannot"},
        {"role": "user", "content": "next goal"}]) == [[], []]
    assert module.extract_multi_turn_responses([
        {"role": "user", "content": "[CONVERSATION_COMPLETED]"}]) == []


def make_agent(tmp_path, monkeypatch, **overrides):
    for name in ("TEVO_LLM_SEED", "TEVO_INJECT_TOOL_PROFILES", "TEVO_ALLOW_VAL_OBSERVER_LEARNING", "TEVO_DEFER_TRAIN_OBSERVER_LEARNING"):
        monkeypatch.delenv(name, raising=False)
    captured = {}
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: captured.update(kwargs) or SimpleNamespace())
    data = tmp_path / "train.jsonl"
    ids = ["multi_turn_base_1", "multi_turn_base_2"]
    data.write_text("".join(json.dumps({
        "id": task_id, "messages": [{"role": "user", "content": "goal " + task_id}],
        "tools": [], "extra": {"id": task_id, "involved_classes": []},
    }) + "\n" for task_id in ids))
    kwargs = dict(index=0, task_ids=ids, experiment_name="bfcl-train", data_path=str(data), num_trials=2)
    kwargs.update(overrides)
    return Agent(**kwargs), captured


def response(usage=None):
    message = SimpleNamespace(model_dump=lambda **kwargs: {"role": "assistant", "content": "done"})
    usage_obj = SimpleNamespace(model_dump=lambda **kwargs: usage) if usage is not None else None
    return SimpleNamespace(id="fake-response", model="fake-model", system_fingerprint="fake-server",
                           choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage_obj)


def test_actor_native_context_headroom_records_effective_payload(tmp_path, monkeypatch):
    from tevo.llm.context_budget import ContextBudgetError
    policy = {"policy": "server_tokenize_headroom", "context_window_tokens": 32768}
    agent, _ = make_agent(tmp_path, monkeypatch, max_tokens=8192, context_budget=policy, max_retries=99)
    agent.thinking_extra_body_mode = "chat_template_kwargs"
    tokenizations, generations = [], []
    def tokenize(body):
        tokenizations.append(body)
        return json.dumps({"count": 24577, "max_model_len": 32768, "tokens": [1] * 24577})
    agent._request_tokenization = tokenize
    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: generations.append(kwargs) or response({"prompt_tokens": 24577, "total_tokens": 24580}))))
    agent.call_llm([{"role": "user", "content": "goal"}], [])
    assert len(tokenizations) == 1 and generations[0]["max_tokens"] == 8191
    assert agent.request_metrics[0]["requested_max_tokens"] == 8192
    assert agent.request_metrics[0]["effective_max_tokens"] == 8191
    assert agent.context_budget_metrics[0]["generation_prompt_count_matches"] is True
    agent._request_tokenization = lambda body: (_ for _ in ()).throw(TimeoutError("private"))
    with pytest.raises(ContextBudgetError):
        agent.call_llm([{"role": "user", "content": "goal"}], [])
    assert len(generations) == 1 and agent.request_metrics[-1]["provider_request_sent"] is False


def test_trial_seed_timeout_and_usage_are_recorded(tmp_path, monkeypatch):
    agent, client_options = make_agent(tmp_path, monkeypatch, trial_seeds=[11, 23], max_tokens=128,
                                       request_timeout_seconds=7, max_retries=1, retry_backoff_seconds=0)
    assert client_options["timeout"] == 7 and client_options["max_retries"] == 0
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("fake timeout")
        return response({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})

    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    messages = [{"role": "user", "content": "a goal"}]
    assert agent.call_llm(messages, [])["content"] == "done"
    first_events = list(agent.request_metrics)
    assert [event["status"] for event in first_events] == ["error", "success"]
    assert first_events[0]["usage"] is None and first_events[1]["usage"]["total_tokens"] == 13
    assert agent._usage_summary()["total_tokens"] is None  # Unknown failed request is not fabricated as zero.
    agent._active_run_id = 1
    agent.request_metrics = []
    agent.call_llm(messages, [])
    assert [call["seed"] for call in calls] == [derive_seed(11, agent.task_ids[0]), derive_seed(11, agent.task_ids[0]), derive_seed(23, agent.task_ids[0])]
    assert all(call["max_tokens"] == 128 for call in calls)
    assert first_events[1]["prompt_sha256"] == agent.request_metrics[0]["prompt_sha256"]
    assert first_events[1]["request_sha256"] != agent.request_metrics[0]["request_sha256"]
    assert agent._usage_summary()["total_tokens"] == 13
    agent.llm_client.chat.completions.create = lambda **kwargs: response()
    agent.request_metrics = []
    agent.call_llm(messages, [])
    assert agent.request_metrics[0]["usage"] is None
    assert agent._convert_conversation_to_eval_format(1, 0)["input_token_count"] is None


def test_thinking_false_is_explicit_in_the_actual_request(tmp_path, monkeypatch):
    agent, _ = make_agent(tmp_path, monkeypatch)
    agent.thinking_extra_body_mode = "chat_template_kwargs"
    requests = []
    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: requests.append(kwargs) or response())))
    for enabled in (False, True):
        agent.enable_thinking = enabled
        agent.call_llm([{"role": "user", "content": "a goal"}], [])
    assert [request["extra_body"]["chat_template_kwargs"]["enable_thinking"] for request in requests] == [False, True]
    assert agent.request_metrics[0]["request_sha256"] != agent.request_metrics[1]["request_sha256"]


def test_suite_fixed_repeat_sends_seed_zero_in_actual_actor_payload(tmp_path, monkeypatch):
    from tevo.experiment.suite import _run_trial
    # Use the real actor constructor and call_llm, replacing only the provider
    # transport and task execution/scoring (no model or environment requests).
    agent, _ = make_agent(tmp_path, monkeypatch)
    payloads = []
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: payloads.append(kwargs) or response()))))
    def execute(self):
        self.call_llm([{"role": "user", "content": "goal"}], [])
        return [{"task_id": self.task_ids[0], "reward": 1., "sampling": self._sampling_metadata()}]
    monkeypatch.setattr(Agent, "execute", execute)
    plan = {"config": {"sampling_seed_mode": "fixed", "budgets": {
                "max_workers": 1, "max_requests": 20, "max_total_tokens": 10000, "wall_seconds": 60},
                "model": {"name": "fake", "temperature": .9, "max_tokens": 128}},
            "dataset_path": agent.data_path, "answer_path": str(agent.answer_path)}
    (tmp_path / "out").mkdir()
    for repeat in range(4):
        job = {"key": f"job{repeat}", "ordinal": repeat, "arm": {"method": "nomem"},
               "experiment": "test", "task_id": agent.task_ids[0], "base_seed": 0,
               "trial_seed": 0, "trial_index": repeat, "repeat_index": repeat,
               "llm_base_url": "http://localhost:1/v1", "endpoint_index": 0}
        row = _run_trial(job, plan, str(tmp_path / "out"), {"tevo": {}})
        assert row["status"] == "complete", row
        assert row["sampling"]["effective_seed"] == 0
        assert row["run_id"] == row["repeat_index"] == repeat
    assert len(payloads) == 4
    assert all(p["seed"] == 0 and p["temperature"] == .9 for p in payloads)


def test_train_sampling_finishes_before_learning_or_comparative_extraction(tmp_path, monkeypatch):
    agent, _ = make_agent(tmp_path, monkeypatch, use_tevo_observer=True, use_memory_addition=True)
    sampled = []
    learned = []
    comparisons = []
    tool_call = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "fake-call", "type": "function", "function": {"name": "fake_tool", "arguments": "{}"},
    }]}
    monkeypatch.setattr(agent, "call_llm", lambda *args: tool_call.copy())
    monkeypatch.setattr(agent, "env_step", lambda *args: {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]})
    monkeypatch.setattr(agent, "get_reward", lambda run_id, task_index: sampled.append((run_id, task_index)) or 1.0)
    monkeypatch.setattr(agent, "_reset_bfcl_instance_cache", lambda *args, **kwargs: None)

    def finalize(**kwargs):
        assert len(sampled) == 4
        assert kwargs["allow_learning"] is True
        learned.append(kwargs["task_id"])
        return {"success": True}

    def post(**kwargs):
        assert len(sampled) == 4 and len(learned) == 4
        comparisons.append(kwargs["json"])
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(agent, "finalize_tevo_task", finalize)
    monkeypatch.setattr(module.requests, "post", post)
    entries = agent.execute()
    assert len(entries) == 4 and all("error" not in entry for entry in entries)
    assert all(entry["tevo_finalize_defer_scope"] == "train_actor_all_samples" for entry in entries)
    assert len(comparisons) == 2
    assert all(item["user_goal"] == "goal " + item["task_id"] and len(item["trials"]) == 2 for item in comparisons)


def test_multiactor_train_learning_is_rejected_and_evaluation_errors_are_visible(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="max_workers=1"):
        make_agent(tmp_path, monkeypatch, use_tevo_observer=True, use_memory_addition=True, max_workers=2)
    agent, _ = make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "existing-test-key")
    monkeypatch.setattr(module, "OpenAICompletionsHandler", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fake checker failure")))
    assert agent.get_reward(0, 0) == 0
    assert os.environ["OPENAI_API_KEY"] == "existing-test-key"
    entry = agent._build_result_entry(run_id=0, task_index=0, start_time="test", reward=0,
                                      observed_tool_calls=[], tevo_finalize_result=None)
    assert entry["phase"] == "evaluation" and entry["evaluation_error"]["type"] == "RuntimeError"


def test_user_turn_is_not_a_tool_result_and_empty_real_tool_output_is_success(tmp_path, monkeypatch):
    monkeypatch.setenv("TEVO_COMPACT_BFCL_RESULTS", "0")
    agent, _ = make_agent(tmp_path, monkeypatch, task_ids=["multi_turn_base_1"], num_trials=1,
                          use_tevo_observer=True, max_interactions=3)
    outputs = iter([
        {"messages": [{"role": "user", "content": "future goal"}]},
        {"messages": [{"role": "tool", "content": "", "tool_call_id": "fake"}]},
        {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]},
    ])
    monkeypatch.setattr(agent, "call_llm", lambda *args: {"role": "assistant", "content": "", "tool_calls": [{
        "id": "fake", "type": "function", "function": {"name": "fake_tool", "arguments": "{}"}}]})
    monkeypatch.setattr(agent, "env_step", lambda *args: next(outputs))
    monkeypatch.setattr(agent, "get_reward", lambda *args: 0.0)
    monkeypatch.setattr(agent, "finalize_tevo_task", lambda **kwargs: {"success": True})
    entry = agent.execute()[0]
    first, second, _ = entry["observed_tool_calls"]
    assert first["execution_status"] == "not_executed" and first["success"] is False
    assert first["output"] == "" and first["state"]["user_goal"] == "goal multi_turn_base_1"
    assert second["execution_status"] == "executed" and second["success"] is True
    assert second["state"]["user_goal"] == "future goal" and second["turn_id"] == 2
    assert first["episode_id"] == second["episode_id"] == entry["episode_id"]


def test_nonretryable_budget_stop_does_not_retry_or_sleep(tmp_path, monkeypatch):
    agent, _ = make_agent(tmp_path, monkeypatch, max_retries=5)
    class BudgetStop(RuntimeError):
        retryable = False
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise BudgetStop("no request budget")
    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(module.time, "sleep", lambda *args: pytest.fail("budget stop must not sleep"))
    with pytest.raises(BudgetStop):
        agent.call_llm([{"role": "user", "content": "goal"}], [])
    assert len(calls) == len(agent.request_metrics) == 1






def _execute_observed_episodes(agent, monkeypatch):
    """Exercise real episode/call/state/finalizer propagation with local tools."""
    literal = "0123456789abcdef0123456789abcdef"
    goal = "Read the resource named " + literal + " without changing its ID"
    captured = []
    for run_id in range(agent.num_trials):
        for task_index in range(len(agent.task_ids)):
            agent.history[run_id][task_index][0]["content"] = goal
    monkeypatch.setattr(agent, "call_llm", lambda *args: {"role": "assistant", "content": "", "tool_calls": [{
        "id": "local-call", "type": "function", "function": {
            "name": "fake_tool", "arguments": json.dumps({"resource_id": literal})}}]})
    steps = {}
    def step(run_id, task_index, output):
        key = (run_id, task_index)
        steps[key] = steps.get(key, 0) + 1
        if steps[key] == 1:
            return {"messages": [{"role": "tool", "tool_call_id": "local-call",
                                  "content": json.dumps({"resource_id": literal})}]}
        return {"messages": [{"role": "user", "content": "[CONVERSATION_COMPLETED]"}]}
    monkeypatch.setattr(agent, "env_step", step)
    monkeypatch.setattr(agent, "get_reward", lambda run_id, task_index: float(run_id == 0))
    monkeypatch.setattr(agent, "_reset_bfcl_instance_cache", lambda *args: None)
    monkeypatch.setattr(agent, "finalize_tevo_task", lambda **kwargs: captured.append(kwargs) or {"success": True})
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: pytest.fail("No HTTP in episode test"))
    entries = agent.execute()
    assert all("error" not in entry for entry in entries)
    for entry, payload in zip(entries, captured):
        assert payload["episode_id"] == entry["episode_id"]
        for call in payload["tool_calls"]:
            assert call["episode_id"] == payload["episode_id"] == call["state"]["metadata"]["episode_id"]
            assert call["state"]["user_goal"] == goal
            assert call["args"]["resource_id"] == literal
        assert json.loads(payload["tool_calls"][0]["output"])["resource_id"] == literal
    return entries, captured


def test_episode_identity_uses_original_seed_task_and_experiment_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("TEVO_COMPACT_BFCL_RESULTS", "0")
    # A 31-bit sampling-seed collision must not collapse distinct episodes.
    monkeypatch.setattr(module, "derive_seed", lambda *args: 7)
    all_ids = []
    for experiment in ("bfcl-train", "bfcl-train-other"):
        agent, _ = make_agent(tmp_path, monkeypatch, trial_seeds=[11, 23], experiment_name=experiment,
                              use_tevo_observer=True, max_interactions=2)
        entries, _ = _execute_observed_episodes(agent, monkeypatch)
        for entry in entries:
            original_seed = [11, 23][entry["run_id"]]
            assert entry["episode_id"] == canonical_hash([
                "bfcl-explicit-episode-v1", experiment, entry["task_id"], original_seed])[:32]
            all_ids.append(entry["episode_id"])
    assert len(set(all_ids)) == 8


def test_legacy_episode_identity_remains_distinct_without_explicit_trial_seeds(tmp_path, monkeypatch):
    monkeypatch.setenv("TEVO_COMPACT_BFCL_RESULTS", "0")
    identities = []
    for _ in range(2):
        agent, _ = make_agent(tmp_path, monkeypatch, task_ids=["multi_turn_base_1"], num_trials=1,
                              use_tevo_observer=True, max_interactions=2)
        monkeypatch.setenv("TEVO_LLM_SEED", "11")  # environment seed is still the legacy protocol
        entries, _ = _execute_observed_episodes(agent, monkeypatch)
        identities.append(entries[0]["episode_id"])
    assert len(set(identities)) == 2
    assert all(len(identity) == 32 for identity in identities)
