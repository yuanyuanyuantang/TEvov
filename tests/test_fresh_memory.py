"""No network: bind only to available responses, and never accept nonexistent evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fresh_memory", ROOT / "scripts/build_memory_from_scratch.py")
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


def fixture():
    task = {"tools": [{"function": {"name": name, "description": description,
             "parameters": {"type": "object", "properties": parameters}}}
        for name, description, parameters in [
            ("read", "Read current fuel.", {}),
            ("fill", "The tank holds up to 50 gallons.", {"amount": {"type": "number"}})]]}
    trial = {"episode_id": "e", "base_seed": 20261014, "task_history": [
        {"role": "user", "content": "Fill the tank to full."},
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": '{"fuel": 17.5}'},
        {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "fill", "arguments": '{"amount": 32.5}'}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"fuel": 50}'},
    ]}
    candidate = {"when_to_use": "Filling the current tank to full", "analysis_kind": "success",
        "tool_tags": ["read", "fill"], "steps": [
            {"text": "Read the current fuel before filling.",
             "evidence_refs": [{"episode_id": "e", "call_index": 0}], "requires_sources": []},
            {"text": "Compute the missing amount from the current reading.",
             "evidence_refs": [{"episode_id": "e", "call_index": 0}, {"episode_id": "e", "call_index": 1}], "requires_sources": [0]}],
        "parameter_sources": [{"episode_id": "e", "kind": "state_difference", "target_call": 1, "parameter": "amount",
            "source_call": 0, "source_path": ["fuel"], "constant": 50.0,
            "contract_tool": "fill", "contract_quote": "holds up to 50 gallons"}]}
    return task, trial, candidate


def test_difference_renders_rule_without_episode_values():
    task, trial, candidate = fixture()
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    content, sources = checked["content"], checked["parameter_sources"]
    assert '50.0 minus' in content and 'response["fuel"]' in content
    assert "17.5" not in content and "32.5" not in content and len(sources) == 1


def test_tool_output_renders_separate_requests_and_allows_existing_response():
    task, trial, candidate = fixture()
    source = candidate["parameter_sources"][0]
    source["kind"] = "tool_output"
    for key in ("constant", "contract_tool", "contract_quote"):
        del source[key]
    trial["task_history"][3]["tool_calls"][0]["function"]["arguments"] = '{"amount": 17.5}'
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert len(checked["parameter_sources"]) == 1
    content = checked["content"]
    assert "wait before calling fill (not in the same batch)" in content
    assert 'current read response["fuel"]' in content
    assert "Reuse a matching response if already available" in content and "SAME intended object" in content
    assert "17.5" not in content


@pytest.mark.parametrize("output,outcome", [
    ('{"error":"capacity exceeded"}', "explicit_error"),
    ('{"success":false}', "explicit_error"),
    ('null', "unknown"), ('Operation completed', "unknown"),
    ('{"fuel":50}', "local_no_reported_error"),
])
def test_positive_target_gate_keeps_ordinary_failure_lessons(output, outcome):
    task, trial, candidate = fixture()
    trial["reward"] = 0.0  # A failed whole task can contain usable local evidence.
    trial["task_history"][-1]["content"] = output
    calls = fresh.trace_calls(trial)
    assert fresh.target_outcome(calls[-1]) == outcome
    schemas = {t["function"]["name"]: t["function"] for t in task["tools"]}
    lookup = {(c["episode_id"], c["call_index"]): c for c in calls}
    if outcome == "local_no_reported_error":
        assert fresh.check_source(candidate["parameter_sources"][0], schemas, lookup)
    else:
        with pytest.raises(ValueError, match="target not positive support: " + outcome):
            fresh.check_source(candidate["parameter_sources"][0], schemas, lookup)
        candidate["steps"] = [{"text": "Check the tool prerequisites after a rejected operation.",
            "evidence_refs": [{"episode_id": "e", "call_index": 1}], "requires_sources": []}]
        checked = fresh.check_candidate(candidate, task, calls)
        assert not checked["parameter_sources"]
        assert checked["content"] == candidate["steps"][0]["text"]


def test_corrected_bank_filters_mixed_support_without_losing_original_lineage():
    from scripts.correct_parameter_bank import correct_records
    task, trial, _ = fixture()
    trial["task_id"] = "train_a"
    trial["task_history"][0]["content"] = "Fill 32.5 gallons."
    trial["replay_payload"] = {"tool_calls": fresh.trace_calls(trial)}
    bad = copy.deepcopy(trial)
    bad["episode_id"] = "bad"
    bad["task_history"][-1]["content"] = '{"error":"rejected"}'
    bad["replay_payload"]["tool_calls"] = fresh.trace_calls(bad)
    refs = [{"episode_id": t["episode_id"], "call_index": 1,
             "call_sha256": fresh.canonical_hash(t["replay_payload"]["tool_calls"][1])} for t in (trial, bad)]
    meta = {"evidence_refs": refs, "candidate_assessments": [{"task_id": "train_a"}],
        "parameter_sources_by_episode": {t["episode_id"]: [{"episode_id": t["episode_id"],
            "target_call": 1, "parameter": "amount", "kind": "user_input", "user_quote": "32.5"}] for t in (trial, bad)}}
    def record(kind, content, metadata):
        return {"record_id": "mem_" + fresh.canonical_hash(["Fill current tank", content, ["fill"]])[:16],
            "when_to_use": "Fill current tank", "content": content, "tool_tags": ["fill"],
            "type_tag": kind, "metadata": metadata}
    ordinary = record("generic_rule", "Check prerequisites.", {**meta, "parameter_sources_by_episode": {}})
    rule = record("parameter_template", "Old rendered rule.", meta)
    rows = [ordinary, rule]
    index = {r["record_id"]: {"search_text": r["when_to_use"], "vector": [.1, .2]} for r in reversed(rows)}
    before = copy.deepcopy([rows, index])
    kept, subset, audit, mapping = correct_records(rows, index, {"train_a": task}, [trial, bad])
    assert [rows, index] == before and kept[0] == ordinary
    updated = kept[1]["metadata"]
    assert updated["source_record_metadata"] == meta  # Full original evidence is private.
    assert set(updated["parameter_sources_by_episode"]) == {"e"}
    assert [r["episode_id"] for r in updated["evidence_refs"]] == ["e"]
    assert audit[0]["accepted_supports"] == 1 and audit[0]["rejected_supports"][0]["outcome"] == "explicit_error"
    assert list(subset) == [mapping[k] for k in index]  # Preserve tie order across ID remapping.
    assert all(subset[mapping[k]] == index[k] for k in index)
    both_bad = copy.deepcopy(trial)
    both_bad["task_history"][-1]["content"] = 'null'
    both_bad["replay_payload"]["tool_calls"] = fresh.trace_calls(both_bad)
    changed = copy.deepcopy(rows)
    for r in changed:
        r["metadata"]["evidence_refs"][0]["call_sha256"] = fresh.canonical_hash(both_bad["replay_payload"]["tool_calls"][1])
    kept, _, audit, _ = correct_records(changed, index, {"train_a": task}, [both_bad, bad])
    assert len(kept) == 1 and kept[0]["type_tag"] == "generic_rule" and audit[0]["record_id"] is None


def test_formal_collection_grid_is_50_by_8_without_memory():
    model = {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:8010/v1", "http://localhost:8011/v1"]}
    seeds = list(range(20261020, 20261028))
    plan = fresh.collection_plan(model, 50, seeds, 4)
    assert len(plan["task_ids"]) == 50 and len(plan["jobs"]) == 400
    assert len({j["key"] for j in plan["jobs"]}) == 400
    assert len({(j["task_id"], j["base_seed"]) for j in plan["jobs"]}) == 400
    assert plan["bank_paths"] == plan["config"]["frozen"] == {}
    assert {j["arm"]["method"] for j in plan["jobs"]} == {"nomem"}
    for task in plan["task_ids"]:
        rows = [j for j in plan["jobs"] if j["task_id"] == task]
        assert {j["base_seed"] for j in rows} == set(seeds)
        assert [j["endpoint_index"] for j in rows].count(0) == 4
    assert plan["config"]["budgets"]["max_requests"] == 24000


def test_collection_replay_rejects_injection_and_preserves_task_failures():
    task, trial, _ = fixture()
    task["messages"] = [copy.deepcopy(trial["task_history"][0])]
    job = {"key": "k", "task_id": "train_a", "base_seed": 1, "trial_seed": 2,
           "experiment": "fresh_nomem_train", "endpoint_index": 0, "llm_base_url": "http://localhost:1/v1"}
    trial.update(job, status="complete", reward=0.0, memory_injection_log=[],
                 tevo_observer_enabled=False, memory_context_sha256=fresh.canonical_hash({}), per_turn_memory=False)
    replay = fresh.training_replay(trial, job, task)
    assert replay["reward"] == 0.0 and replay["collection_memory_read"] is False
    assert fresh.canonical_hash(replay["replay_payload"]) == replay["replay_sha256"]
    injected = copy.deepcopy(trial)
    injected["memory_injection_log"] = [{"text": "old memory"}]
    with pytest.raises(ValueError, match="no memory"):
        fresh.training_replay(injected, job, task)
    injected = copy.deepcopy(trial)
    injected["task_history"][0]["content"] += " old memory"
    with pytest.raises(ValueError, match="altered"):
        fresh.training_replay(injected, job, task)


def test_collection_resume_preserves_checkpoint_and_rejects_changed_grid(tmp_path, monkeypatch):
    task, trial, _ = fixture()
    task.update(id="train_a", messages=[copy.deepcopy(trial["task_history"][0])])
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps(task) + "\n")
    val.write_text('{"id":"val_b"}\n')
    monkeypatch.setattr(fresh, "TRAIN", train)
    monkeypatch.setattr(fresh, "VAL", val)
    runs, executions = [], []
    def fake_suite(plan, raw, resume=False):
        runs.append(resume)
        raw.mkdir(exist_ok=True)
        fresh.write_private_json(raw / "manifest.json", {"fingerprints": fresh.fingerprints(plan)})
        for job in plan["jobs"]:
            path = raw / "trials" / (job["key"] + ".json")
            if not path.exists():
                row = copy.deepcopy(trial)
                row.update({k: job[k] for k in ("key", "task_id", "base_seed", "trial_seed", "experiment", "endpoint_index", "llm_base_url")})
                row.update(episode_id="e"+str(job["base_seed"]), status="complete", reward=0.0,
                           memory_injection_log=[], tevo_observer_enabled=False,
                           memory_context_sha256=fresh.canonical_hash({}), per_turn_memory=False,
                           manifest_sha256=fresh.canonical_hash(fresh.fingerprints(plan)))
                fresh.write_private_json(path, row, exclusive=True)
                executions.append(job["key"])
            if len(runs) == 1:
                raise RuntimeError("interrupted after paid checkpoint")
        return {"provenance_valid": True, "resources": {"requests": 2}}
    monkeypatch.setattr(fresh, "run_suite", fake_suite)
    import scripts.regrade_suite as grading
    monkeypatch.setattr(grading, "regrade", lambda plan, raw: {"complete_matrix_regraded": True, "matches": 2})
    model = {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]}
    output = tmp_path / "collection"
    with pytest.raises(RuntimeError, match="paid checkpoint"):
        fresh.collect(output, model, 1, [1, 2])
    first = next((output / "raw/trials").glob("*.json"))
    digest = fresh.sha256_file(first)
    with pytest.raises(ValueError, match="resume refused"):
        fresh.collect(output, model, 1, [1, 3], resume=True)
    assert len(runs) == 1
    result = fresh.collect(output, model, 1, [1, 2], resume=True)
    assert result["complete"] and result["trials"] == result["regrade_matches"] == 2
    assert len(executions) == len(set(executions)) == 2
    assert fresh.sha256_file(first) == digest and runs == [False, True]


def test_same_batch_response_is_unavailable_even_with_earlier_call_index():
    task, trial, candidate = fixture()
    trial["task_history"][1]["tool_calls"].extend(trial["task_history"][3]["tool_calls"])
    del trial["task_history"][3]
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert "unavailable" in checked["rejected_sources"][0]["reason"]
    assert checked["content"] == "Read the current fuel before filling."
    assert "missing amount" not in checked["content"]


@pytest.mark.parametrize("mutation,match", [
    ("wrong_value", "does not fit"), ("wrong_path", "invented"),
    ("invented_quote", "quote mismatch"),
])
def test_bad_binding_drops_only_its_dependent_step(mutation, match):
    task, trial, candidate = fixture()
    source = candidate["parameter_sources"][0]
    if mutation == "wrong_value":
        trial["task_history"][3]["tool_calls"][0]["function"]["arguments"] = '{"amount": 31.5}'
    elif mutation == "wrong_path": source["source_path"] = ["invented"]
    else: source["contract_quote"] = "tank capacity is 100 gallons"
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert checked["content"] == "Read the current fuel before filling."
    assert checked["parameter_sources"] == []
    assert match in checked["rejected_sources"][0]["reason"]
    assert checked["rejected_steps"][0]["step_index"] == 1
    assert checked["evidence_refs"] == [{"episode_id": "e", "call_index": 0}]


def test_step_literals_or_bad_citations_do_not_remove_valid_binding():
    task, trial, candidate = fixture()
    candidate["steps"][1]["text"] = "Fill with 32.5 gallons."
    candidate["steps"][0]["evidence_refs"][0]["call_index"] = 99
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert len(checked["rejected_steps"]) == 2
    assert len(checked["parameter_sources"]) == 1 and "50.0 minus" in checked["content"]
    assert "32.5" not in checked["content"]


def test_unknown_global_tool_tag_still_rejected():
    task, trial, candidate = fixture()
    candidate["tool_tags"] = ["invented"]
    with pytest.raises(ValueError, match="unknown tool"):
        fresh.check_candidate(candidate, task, fresh.trace_calls(trial))


def test_general_step_lesson_can_survive_without_parameter_binding():
    task, trial, candidate = fixture()
    candidate["parameter_sources"] = []
    candidate["steps"] = candidate["steps"][:1]
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert checked["parameter_sources"] == [] and checked["content"] == candidate["steps"][0]["text"]


def test_duplicate_and_unknown_tool_responses_rejected():
    _, trial, _ = fixture()
    trial["task_history"].append(copy.deepcopy(trial["task_history"][-1]))
    with pytest.raises(ValueError, match="duplicate tool response"):
        fresh.trace_calls(trial)
    trial["task_history"][-1]["tool_call_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        fresh.trace_calls(trial)


def test_call_ids_can_repeat_across_completed_batches():
    task, trial, candidate = fixture()
    trial["task_history"][3]["tool_calls"][0]["id"] = "a"
    trial["task_history"][4]["tool_call_id"] = "a"
    calls = fresh.trace_calls(trial)
    assert len(calls) == 2 and calls[0]["response_message"] == 2 and calls[1]["response_message"] == 4
    checked = fresh.check_candidate(candidate, task, calls)
    assert "minus" in checked["content"]


def test_historical_array_position_is_not_a_general_object_selection_rule():
    with pytest.raises(ValueError, match="invalid response path"):
        fresh.at_path([{"id": "first"}, {"id": "second"}], [0, "id"])


def test_distillation_uses_qwen_without_external_key_fallback(monkeypatch):
    monkeypatch.delenv("TEVO_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-be-used")
    config = fresh.distillation_config({"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]})
    assert config.model == "Qwen/Qwen3-8B" and config.resolved_api_key() == "dummy"
    assert config.max_retries == 0 and not config.cache_enabled
    assert config.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert "thinking" not in config.extra_body
    with pytest.raises(ValueError, match="requires Qwen3-8B"):
        fresh.distillation_config({"name": "deepseek-flash", "base_urls": ["https://example.invalid/v1"]})


def test_user_input_substring_is_not_proof_of_the_target_value():
    task, trial, candidate = fixture()
    trial["task_history"][0]["content"] = "Fill with 132.5 gallons."
    candidate["parameter_sources"] = [{"episode_id": "e", "kind": "user_input", "target_call": 1, "parameter": "amount", "user_quote": "132.5"}]
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial))
    assert "does not support" in checked["rejected_sources"][0]["reason"]
    assert checked["content"] == "Read the current fuel before filling."


def test_paid_response_persisted_even_when_json_parsing_fails(tmp_path, monkeypatch):
    task, trial, _ = fixture()
    task["id"] = "train_synthetic"
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps(task) + "\n")
    val.write_text('{"id":"val_synthetic"}\n')
    monkeypatch.setattr(fresh, "TRAIN", train)
    monkeypatch.setattr(fresh, "VAL", val)
    trial.update(task_id=task["id"], episode_id="e", collection_memory_read=False,
                 collection_memory_write=False, status="complete", reward=1)
    trial["replay_payload"] = {"episode_id": "e", "task_id": task["id"], "tool_calls": fresh.trace_calls(trial)}
    trial["replay_sha256"] = fresh.canonical_hash(trial["replay_payload"])
    directory = tmp_path / "trials"
    directory.mkdir()
    (directory / "trial.json").write_text(json.dumps(trial))
    class FakeClient:
        def __init__(self, config): pass
        def _request_tokenization(self, payload):
            raise AssertionError("fake chat does not request tokenization")
        def chat(self, *args, **kwargs):
            return {"content": "not JSON", "raw": {"usage": {"total_tokens": 8}, "choices": []}}
    monkeypatch.setattr(fresh, "DeepSeekLLMClient", FakeClient)
    monkeypatch.setattr(fresh, "inspect_providers", lambda model: {"verified": True})
    output = tmp_path / "bank"
    with pytest.raises(json.JSONDecodeError):
        fresh.build(directory, output, "http://localhost:1/v1", {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]})
    assert json.loads((output / "distillation_responses.json").read_text())[0]["response"]["usage"]["total_tokens"] == 8
    assert not (output / "memories.json").exists()


def fresh_trial(task_id, episode, seed, reward):
    _, trial, _ = fixture()
    trial.update(task_id=task_id, episode_id=episode, base_seed=seed, reward=reward,
                 collection_memory_read=False, collection_memory_write=False, status="complete")
    trial["replay_payload"] = {"episode_id": episode, "task_id": task_id, "tool_calls": fresh.trace_calls(trial)}
    trial["replay_sha256"] = fresh.canonical_hash(trial["replay_payload"])
    return trial


@pytest.mark.parametrize("scores,mode", [([1, 0], "success_failure"), ([0, 0], "same_outcome"),
    ([1, 1], "same_outcome"), ([.2, .4], "different_scores")])
def test_task_grouping_preserves_actual_outcomes_and_episode_identity(scores, mode):
    task, _, _ = fixture()
    trials = [fresh_trial("train_a", "a", 2, scores[0]), fresh_trial("train_a", "b", 1, scores[1])]
    groups = fresh.group_trials(trials, {"train_a": task})
    assert len(groups) == 1 and groups[0]["comparison_mode"] == mode
    assert [t["episode_id"] for t in groups[0]["trajectories"]] == ["b", "a"]
    assert all(c["episode_id"] == t["episode_id"] for t in groups[0]["trajectories"] for c in t["calls"])


def test_representatives_keep_ties_and_are_independent_of_file_order():
    task, _, _ = fixture()
    trials = [fresh_trial("a", f"e{s}", s, 0) for s in range(8)]
    first = fresh.group_trials(trials, {"a": task})[0]
    reverse = fresh.group_trials(list(reversed(trials)), {"a": task})[0]
    selected = fresh.procedure_evidence(first)
    assert selected == fresh.procedure_evidence(reverse)
    assert selected["comparison_mode"] == "same_outcome"
    assert [t["episode_id"] for t in selected["trajectories"]] == ["e0", "e7"]
    assert len(first["trajectories"]) == 8  # Compiler evidence is not reduced.
    for t in selected["trajectories"]:
        for compact, original in zip(t["calls"], next(x["calls"] for x in first["trajectories"] if x["episode_id"] == t["episode_id"])):
            assert all(compact[k] == original[k] for k in compact)
    with pytest.raises(ValueError, match="1-8"):
        fresh.group_trials(trials + [fresh_trial("a", "e9", 9, 0)], {"a": task})


def test_preventive_suggestion_is_retained_and_bad_steps_are_local():
    task, trial, candidate = fixture()
    candidate["parameter_sources"] = []
    observed = {**candidate["steps"][0], "evidence_kind": "observed"}
    suggestion = {**observed, "evidence_kind": "preventive_suggestion",
                  "text": "To prevent an invalid fill, inspect the current tank before acting."}
    candidate["steps"] = [observed, suggestion, {**observed, "evidence_kind": "verified_success"},
                          {**observed, "requires_sources": [0]}, candidate["steps"][0]]
    checked = fresh.check_candidate(candidate, task, fresh.trace_calls(trial), ordinary_only=True)
    assert len(checked["retained_steps"]) == 2 and len(checked["rejected_steps"]) == 3
    assert [s["evidence_kind"] for s in checked["step_assessments"]] == ["observed", "preventive_suggestion"]
    assert all(s["semantic_support_verified"] is False for s in checked["step_assessments"])


def test_flat_ordinary_memory_needs_no_tags_classifications_or_empty_lists():
    task, trial, _ = fixture()
    candidate = {"when_to_use": "Before filling the current tank",
                 "content": "To prevent overfilling, inspect the current tank before acting.",
                 "evidence_refs": [{"episode_id": "e", "call_index": 0}]}
    checked = fresh.check_procedure(candidate, task, fresh.trace_calls(trial))
    assert checked["content"] == candidate["content"]
    assert checked["tool_tags"] == ["read"] and checked["analysis_kind"] == "ordinary"
    assert checked["parameter_sources"] == checked["step_assessments"] == []
    assert fresh.check_procedure({**candidate, "confidence": 1, "parameter_sources": []}, task,
                                 fresh.trace_calls(trial))["content"] == candidate["content"]
    for change, match in [({"evidence_refs": []}, "lacks evidence"),
                          ({"evidence_refs": [{"episode_id": "e", "call_index": 99}]}, "invalid evidence call"),
                          ({"content": "Fill with 32.5 gallons."}, "historical numeric"),
                          ({"parameter_sources": [{"kind": "invented"}]}, "parameter dependencies")]:
        with pytest.raises(ValueError, match=match):
            fresh.check_procedure({**candidate, **change}, task, fresh.trace_calls(trial))


def test_literal_guard_accepts_documented_options_and_identifier_boundaries():
    schemas = {"status": {"parameters": {"properties": {"option": {
        "type": "string", "description": 'Choose [Enum]: ["fuel", "doors"]'}}}}}
    calls = [{"args": {"option": "fuel", "field": "first", "ticket": "Ticket7423"}}]
    fresh.screen_text("Check fuel before filling; use the first_name parameter.", calls, schemas, [])
    with pytest.raises(ValueError, match="historical string"):
        fresh.screen_text("Use Ticket7423 from the old episode.", calls, schemas, [])
    with pytest.raises(ValueError, match="historical string"):
        fresh.screen_text("Use first as the person's name.", calls, schemas, [])


def test_literal_guard_catches_nested_tags_goal_thresholds_and_numeric_spellings():
    calls = [{"args": {"tags": ["#FamilyHistory"], "nested": {"amount": 39.5}},
              "goal_snapshot": "Promising means above 300; visit a shop below 40."}]
    for text, reason in [("Post #FamilyHistory.", "historical string"),
                         ("Fill 39.50.", "historical numeric"),
                         ("Buy above 300.", "user-goal number"),
                         ("Visit a shop below 40.0 psi.", "user-goal number")]:
        with pytest.raises(ValueError, match=reason):
            fresh.screen_text(text, calls, {}, [], include_goal_numbers=True)
    fresh.screen_text("Use the current user's threshold and tags.", calls, {}, [], include_goal_numbers=True)
    fresh.screen_text("Use 40 as the documented bound.", calls, {}, [40.0], include_goal_numbers=True)


def test_formal_build_reads_all_400_but_summarizes_only_representatives(tmp_path, monkeypatch):
    task, _, _ = fixture()
    tasks = []
    directory = tmp_path / "trials"
    directory.mkdir()
    for n in range(50):
        current = copy.deepcopy(task)
        current["id"] = f"train_{n}"
        current["tools"][0]["function"]["name"] = f"read_{n}"
        current["tools"][1]["function"]["name"] = f"fill_{n}"
        tasks.append(current)
        for seed in range(8):
            row = fresh_trial(current["id"], f"e{n}_{seed}", seed, float(seed == 7))
            row["task_history"][1]["tool_calls"][0]["function"]["name"] = f"read_{n}"
            row["task_history"][3]["tool_calls"][0]["function"]["name"] = f"fill_{n}"
            if seed == 3:  # Source occurs only outside the LLM representative pair.
                row["task_history"][2]["content"] = '{"fuel":17.5,"needed":32.5}'
            row["replay_payload"]["tool_calls"] = fresh.trace_calls(row)
            row["replay_sha256"] = fresh.canonical_hash(row["replay_payload"])
            (directory / f"{n}_{seed}.json").write_text(json.dumps(row))
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    val.write_text('{"id":"validation"}\n')
    monkeypatch.setattr(fresh, "TRAIN", train)
    monkeypatch.setattr(fresh, "VAL", val)
    requests = []
    class FakeClient:
        def __init__(self, config): pass
        def _request_tokenization(self, payload): return "{}"
        def chat(self, prompt, user_prompt, **kwargs):
            g = json.loads(user_prompt)
            requests.append(g)
            self._request_tokenization({})
            tool = g["tools"][0]["function"]["name"]
            refs = [{"episode_id": t["episode_id"], "call_index": 0} for t in g["trajectories"]]
            candidate = {"when_to_use": f"Before filling after {tool}",
                         "content": f"Inspect the current state with {tool} before filling.", "evidence_refs": refs}
            unseen = {**candidate, "evidence_refs": [{"episode_id": g["trajectories"][0]["episode_id"].replace("_0", "_3"), "call_index": 0}]}
            return {"content": json.dumps({"memories": [candidate, unseen]}), "raw": {"usage": {"total_tokens": 8}, "choices": []}}
    def fake_embedding(texts, settings, ledger, key):
        for start in range(0, len(texts), 32):
            request_id = ledger.reserve(key, "embedding")
            ledger.finish(request_id, None, "success")
        return [[1.0] + [0.0] * 1023 for _ in texts], []
    monkeypatch.setattr(fresh, "DeepSeekLLMClient", FakeClient)
    monkeypatch.setattr(fresh, "inspect_providers", lambda model: {"verified": True})
    monkeypatch.setattr(fresh, "_embedding", fake_embedding)
    output = tmp_path / "bank"
    result = fresh.build(directory, output, "http://localhost:1/v1", {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]})
    assert result["training_trials"] == 400 and len(requests) == 50
    assert all([t["base_seed"] for t in g["trajectories"]] == [0, 7] for g in requests)
    assert result["source_assessment_counts"] == {"unresolved": 350, "validated": 50}
    assert result["memory_type_counts"] == {"generic_rule": 50, "parameter_template": 50}
    assert len(result["rejected"]) == 50  # Invalid flat memory does not remove the other candidate.
    assert all(r["reason"] == "invalid evidence call" for r in result["rejected"])
    assert result["ordinary_semantic_support_verified"] is False
    assert result["ordinary_parameter_relations_verified"] is False
    rows = json.loads((output / "memories.json").read_text())
    assert len(rows) == 100 and result["resources"]["requests"] == 105
    assert all(r["metadata"]["evidence_refs"][0]["episode_id"].endswith("_3") for r in rows if r["type_tag"] == "parameter_template")
    for row in rows:
        if row["type_tag"] == "generic_rule":
            assert {r["episode_id"].rsplit("_", 1)[-1] for r in row["metadata"]["evidence_refs"]} == {"0", "7"}


def test_build_context_overflow_fails_before_generation(tmp_path, monkeypatch):
    task, _, _ = fixture()
    task["id"] = "a"
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps(task) + "\n")
    val.write_text('{"id":"validation"}\n')
    monkeypatch.setattr(fresh, "TRAIN", train)
    monkeypatch.setattr(fresh, "VAL", val)
    directory = tmp_path / "trials"
    directory.mkdir()
    (directory / "a.json").write_text(json.dumps(fresh_trial("a", "e", 1, 0)))
    monkeypatch.setattr(fresh, "inspect_providers", lambda model: {"verified": True})
    monkeypatch.setattr(fresh.DeepSeekLLMClient, "_request_tokenization", lambda self, payload:
                        json.dumps({"count": 32768, "max_model_len": 32768, "tokens": [1] * 32768}))
    def forbid(*args):
        raise AssertionError("overflow must not send a generation request")
    monkeypatch.setattr(fresh.DeepSeekLLMClient, "_request_with_retry", forbid)
    output = tmp_path / "bank"
    with pytest.raises(RuntimeError, match="distillation request failed"):
        fresh.build(directory, output, "http://localhost:1/v1", {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]})
    failure = json.loads((output / "failure.json").read_text())
    assert failure["context_budget"]["prompt_tokens"] == 32768
    assert not (output / "memories.json").exists()


def test_comparative_memory_needs_two_retained_episodes():
    task, trial, candidate = fixture()
    second = copy.deepcopy(trial)
    second["episode_id"] = "f"
    calls = fresh.trace_calls(trial) + fresh.trace_calls(second)
    candidate["analysis_kind"] = "comparative"
    candidate["parameter_sources"] = []
    candidate["steps"] = candidate["steps"][:1]
    with pytest.raises(ValueError, match="two episodes"):
        fresh.check_candidate(candidate, task, calls)
    candidate["steps"].append({"text": "Read the current fuel before filling.",
        "evidence_refs": [{"episode_id": "f", "call_index": 0}], "requires_sources": []})
    checked = fresh.check_candidate(candidate, task, calls)
    assert {r["episode_id"] for r in checked["evidence_refs"]} == {"e", "f"}
    candidate["steps"][1]["evidence_refs"][0]["episode_id"] = "unknown"
    with pytest.raises(ValueError, match="two episodes"):
        fresh.check_candidate(candidate, task, calls)


def test_same_call_indices_across_episodes_do_not_mix_sources():
    task, trial, candidate = fixture()
    second = copy.deepcopy(trial)
    second["episode_id"] = "f"
    second["task_history"][2]["content"] = '{"fuel": 8.0}'
    second["task_history"][3]["tool_calls"][0]["function"]["arguments"] = '{"amount": 42.0}'
    second_source = copy.deepcopy(candidate["parameter_sources"][0])
    second_source["episode_id"] = "f"
    candidate["parameter_sources"].append(second_source)
    calls = fresh.trace_calls(trial) + fresh.trace_calls(second)
    checked = fresh.check_candidate(candidate, task, calls)
    assert len(checked["parameter_sources"]) == 2 and checked["rejected_sources"] == []
    second_source["source_path"] = ["invented"]
    checked = fresh.check_candidate(candidate, task, calls)
    assert [s["episode_id"] for s in checked["parameter_sources"]] == ["e"]
    assert checked["rejected_sources"][0]["source_index"] == 1


def test_no_fallback_when_every_step_depends_on_a_bad_relation():
    task, trial, candidate = fixture()
    candidate["steps"] = candidate["steps"][1:]
    candidate["parameter_sources"][0]["source_path"] = ["invented"]
    with pytest.raises(ValueError, match="no supported"):
        fresh.check_candidate(candidate, task, fresh.trace_calls(trial))


def test_program_sources_preserve_procedures_and_do_not_use_extra_llm_requests(tmp_path, monkeypatch):
    task, _, _ = fixture()
    tasks = [{**task, "id": "train_a"}, {**task, "id": "train_b"}]
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    val.write_text('{"id":"val_c"}\n')
    monkeypatch.setattr(fresh, "TRAIN", train)
    monkeypatch.setattr(fresh, "VAL", val)
    directory = tmp_path / "trials"
    directory.mkdir()
    for task_id in ["train_a", "train_b"]:
        for seed, reward in [(1, 1), (2, 0)]:
            trial = fresh_trial(task_id, task_id + str(seed), seed, reward)
            output = {"fuel": 17.5, "needed": 32.5}
            if seed == 2:
                output["other"] = 32.5
            trial["task_history"][2]["content"] = json.dumps(output)
            trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
            trial["replay_sha256"] = fresh.canonical_hash(trial["replay_payload"])
            (directory / (task_id + str(seed) + ".json")).write_text(json.dumps(trial))
    requests = []
    class FakeClient:
        def __init__(self, config): pass
        def _request_tokenization(self, payload): return "{}"
        def chat(self, prompt, user_prompt, **kwargs):
            group = json.loads(user_prompt)
            requests.append(group)
            self._request_tokenization({})
            candidate = {"when_to_use": "Filling the current tank to full",
                         "content": "Read the current fuel before filling.",
                         "evidence_refs": [{"episode_id": t["episode_id"], "call_index": 0} for t in group["trajectories"]]}
            return {"content": json.dumps({"memories": [candidate]}),
                    "raw": {"usage": {"total_tokens": 8}, "choices": []}}
    def fake_embedding(texts, settings, ledger, key):
        request_id = ledger.reserve(key, "embedding")
        ledger.finish(request_id, None, "success")
        return [[1.0] + [0.0] * 1023 for text in texts], []
    monkeypatch.setattr(fresh, "DeepSeekLLMClient", FakeClient)
    monkeypatch.setattr(fresh, "inspect_providers", lambda model: {"verified": True})
    monkeypatch.setattr(fresh, "_embedding", fake_embedding)
    output = tmp_path / "bank"
    summary = fresh.build(directory, output, "http://localhost:1/v1",
                          {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]})
    assert len(requests) == 2 and all(len(g["trajectories"]) == 2 for g in requests)
    assert {g["task_id"] for g in requests} == {"train_a", "train_b"}
    assert summary["training_trials"] == 4 and summary["memories"] == 2
    assert summary["resources"]["requests"] == 6
    assert summary["resources"]["reported_total_tokens"] == 16
    assert summary["source_assessment_counts"] == {"validated": 2, "ambiguous": 2}
    records = json.loads((output / "memories.json").read_text())
    ordinary = next(r for r in records if r["type_tag"] == "generic_rule")
    assert ordinary["content"] == "Read the current fuel before filling."
    assert len(ordinary["metadata"]["evidence_refs"]) == 4
    assert len(ordinary["metadata"]["candidate_assessments"]) == 2
    parameter = next(r for r in records if r["type_tag"] == "parameter_template")
    assert 'response["needed"]' in parameter["content"]
    assert "32.5" not in parameter["content"]
    assert parameter["source"] == "trace_compile"
    assert parameter["metadata"]["true_or_unique_source_proven"] is False
    assert {r["stage"] for r in json.loads((output / "distillation_responses.json").read_text())} == {"procedures"}
    def no_model(*args, **kwargs):
        raise AssertionError("reuse mode must not construct a client or inspect a model service")
    monkeypatch.setattr(fresh, "DeepSeekLLMClient", no_model)
    monkeypatch.setattr(fresh, "inspect_providers", no_model)
    reused = fresh.build(directory, tmp_path / "reuse", "http://localhost:1/v1",
        {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]}, procedure_bank=output)
    assert reused["memories"] == 2 and reused["resources"]["requests"] == 1
    assert reused["resources"]["reported_total_tokens"] == 0
    assert reused["resources"]["requests_with_unknown_token_usage"] == 1
    assert not (tmp_path / "reuse/distillation_responses.json").exists()
    # A changed reference must fail even if the caller updates the bank's file hash.
    rows = json.loads((output / "memories.json").read_text())
    rows[0]["metadata"]["evidence_refs"][0]["call_sha256"] = "tampered"
    (output / "memories.json").write_text(json.dumps(rows))
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["outputs_sha256"]["memories.json"] = fresh.sha256_file(output / "memories.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence mismatch"):
        fresh.build(directory, tmp_path / "bad_reuse", "http://localhost:1/v1",
            {"name": "Qwen/Qwen3-8B", "base_urls": ["http://localhost:1/v1"]}, procedure_bank=output)
    assert not (tmp_path / "bad_reuse").exists()



def enum_group(trials, task):
    return fresh.group_trials(trials, {"a": task})[0]


def test_enumeration_never_joins_same_call_numbers_across_episodes():
    task, _, _ = fixture()
    first, second = fresh_trial("a", "e", 1, 1), fresh_trial("a", "f", 2, 1)
    first["task_history"][2]["content"] = '{"needed":32.5}'
    second["task_history"][2]["content"] = '{"needed":42.0}'
    second["task_history"][3]["tool_calls"][0]["function"]["arguments"] = '{"amount":42.0}'
    for trial in [first, second]:
        trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
    candidates, audit = fresh.enumerate_parameter_sources(enum_group([first, second], task), task)
    assert len(candidates) == 2 and all(a["status"] == "proposed" for a in audit)
    assert [c[1]["parameter_sources"][0]["episode_id"] for c in candidates] == ["e", "f"]
    second["task_history"][2]["content"] = '{"needed":99.0}'
    second["replay_payload"]["tool_calls"] = fresh.trace_calls(second)
    candidates, audit = fresh.enumerate_parameter_sources(enum_group([first, second], task), task)
    assert len(candidates) == 1 and audit[1]["status"] == "unresolved"


@pytest.mark.parametrize("competing_input", [False, True])
def test_multiple_fitting_sources_abstain(competing_input):
    task, _, _ = fixture()
    trial = fresh_trial("a", "e", 1, 1)
    output = {"needed":32.5}
    if competing_input:
        trial["task_history"][0]["content"] = "Fill with 32.5 gallons."
    else:
        output["also_needed"] = 32.5
    trial["task_history"][2]["content"] = json.dumps(output)
    trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
    candidates, audit = fresh.enumerate_parameter_sources(enum_group([trial], task), task)
    assert candidates == [] and audit[0]["status"] == "ambiguous"
    assert len(audit[0]["fitting_candidates"]) == 2


def test_enumeration_rejects_same_batch_and_does_not_invent_array_rules():
    task, _, _ = fixture()
    trial = fresh_trial("a", "e", 1, 1)
    trial["task_history"][2]["content"] = '{"needed":32.5}'
    trial["task_history"][1]["tool_calls"].extend(trial["task_history"][3]["tool_calls"])
    del trial["task_history"][3]
    trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
    assert fresh.enumerate_parameter_sources(enum_group([trial], task), task)[0] == []
    trial = fresh_trial("a", "e", 1, 1)
    trial["task_history"][2]["content"] = '{"needed":[32.5]}'
    trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
    assert fresh.enumerate_parameter_sources(enum_group([trial], task), task)[0] == []


@pytest.mark.parametrize("goal,expected", [("Fill with 132.5 gallons.", False), ("Fill with 32.50 gallons.", True)])
def test_direct_input_numeric_quotes_are_whole_values(goal, expected):
    task, _, _ = fixture()
    trial = fresh_trial("a", "e", 1, 1)
    trial["task_history"][0]["content"] = goal
    trial["replay_payload"]["tool_calls"] = fresh.trace_calls(trial)
    candidates, audit = fresh.enumerate_parameter_sources(enum_group([trial], task), task)
    assert bool(candidates) is expected
    if expected:
        assert candidates[0][1]["parameter_sources"][0]["user_quote"] == "32.50"


def test_contract_setting_spells_out_endpoint_without_allowing_history_values():
    task, trial, _ = fixture()
    task['tools'][1]['function']['parameters']['properties']['amount']['description'] = (
        'Position between 0 (released) and 1 (fully pressed).')
    trial['task_history'][3]['tool_calls'][0]['function']['arguments'] = '{"amount": 1.0}'
    candidate = {'when_to_use': 'Before operating the actuator',
        'content': 'Use fill with amount set to 1.0 before operating.',
        'evidence_refs': [{'episode_id': 'e', 'call_index': 1}]}
    calls = fresh.trace_calls(trial)
    checked = fresh.check_procedure(candidate, task, calls)
    assert "amount at the documented 'fully pressed' setting" in checked['content']
    assert '1.0' not in checked['content']
    assert checked['contract_setting_rewrites'][0]['contract_quote'] == 'between 0 (released) and 1 (fully pressed)'
    # An unrelated occurrence of the same digit is still screened.
    with pytest.raises(ValueError, match='historical numeric'):
        fresh.check_procedure({**candidate, 'content': candidate['content'] + ' Also copy 1.0.'}, task, calls)
    task['tools'][1]['function']['parameters']['properties']['amount']['description'] = 'Tank holds up to 50 gallons.'
    with pytest.raises(ValueError, match='historical numeric'):
        fresh.check_procedure(candidate, task, calls)


def test_contract_setting_requires_actual_cited_success_and_unambiguous_meaning():
    schema = {'control': {'parameters': {'properties': {'position': {
        'description': 'Position between 0 (released) and 1 (fully pressed).'}}}}}
    call = {'tool_id': 'control', 'args': {'position': 1.0}, 'execution_status': 'executed',
            'response_message': 2, 'output': '{"ok":true}'}
    text = 'Use control with position=1.0.'
    assert fresh.normalize_contract_settings(text, [call], schema)[1]
    for changed in [dict(call, output='{"error":"bad"}'), dict(call, output='null'), dict(call, args={'position': .5})]:
        assert fresh.normalize_contract_settings(text, [changed], schema) == (text, [])
    assert fresh.normalize_contract_settings('Use position=1.0.', [call], schema)[1] == []
    conflicting = copy.deepcopy(schema['control'])
    conflicting['parameters']['properties']['position']['description'] = 'Position between 0 (fully pressed) and 1 (released).'
    assert fresh.normalize_contract_settings(text, [call, dict(call, tool_id='other')],
        {**schema, 'other': conflicting}) == (text, [])
