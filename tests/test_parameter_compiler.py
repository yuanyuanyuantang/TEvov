"""Parameter compiler unit tests use synthetic fixtures, never a historical bank."""
import copy
import json
import socket

import pytest

from tevo.experiment import offline_relations as relations
from tevo.memory.parameter_compiler import compile_records, render_rule
from tevo.reproducibility import canonical_hash
from scripts.compile_parameter_memory import build_bank


@pytest.fixture
def fuel_case():
    calls = [
        {"tool_id": "displayCarStatus", "args": {"option": "fuel"}, "output": '{"fuelLevel":10.5}', "success": True,
         "execution_status": "executed", "turn_id": 0, "state": {"user_goal": "Fill the tank completely"}},
        {"tool_id": "fillFuelTank", "args": {"fuelAmount": 39.5}, "output": '{"fuelLevel":50.0}', "success": True,
         "execution_status": "executed", "turn_id": 0, "state": {"user_goal": "Fill the tank completely"}},
    ]
    unit = {
        "unit_id": "u", "domain": "VehicleControlAPI", "target_tool": "fillFuelTank", "parameter": "fuelAmount",
        "goal": "Fill the tank completely", "condition": "fill_to_full",
        "schemas": {"fillFuelTank": {"parameters": {"properties": {"fuelAmount": {"type": "number"}}}}, "displayCarStatus": {}},
        "prefix": [{"tool": "displayCarStatus", "args": {"option": "fuel"}}],
        "constants": [{"value": 50., "quote": "hold up to 50 gallons"}],
        "all_refs": [{"episode_id": "e", "call_index": 1}],
        "observations": [{"episode_id": "e", "call_index": 1, "prior": calls[:1], "target": calls[1]}],
    }
    record = {"record_id": "m", "when_to_use": "Fill the tank", "content": "fillFuelTank(fuelAmount=39.5)",
              "confidence": .6, "tool_tags": ["fillFuelTank"],
              "metadata": {"evidence_refs": [{"episode_id": "e", "call_index": 1, "success": True}],
                           "evidence_episode_ids": ["e"], "evidence_count": 1}}
    return record, unit, {"e": {"calls": calls}}


def test_fuel_rule_preserves_retrieval_and_source_lineage(fuel_case):
    record, unit, episodes = fuel_case
    snapshot = copy.deepcopy(record)
    output, report = compile_records([record], [unit], episodes)
    result = output[0]
    assert record == snapshot and report["compiled_records"] == 1
    assert result["when_to_use"] == record["when_to_use"] and result["confidence"] == record["confidence"]
    assert "39.5" not in result["content"] and '50.0 - displayCarStatus_response["fuelLevel"]' in result["content"]
    assert result["metadata"]["evidence_refs"] == record["metadata"]["evidence_refs"]
    assert {ref["call_index"] for ref in result["metadata"]["rendered_evidence_refs"]} == {0, 1}
    proof = result["metadata"]["parameter_compilation"]
    assert proof["source_content"] == record["content"] and proof["policy"] == "V0"
    assert result["metadata"]["generated_text_sha256"] == canonical_hash({"when_to_use": result["when_to_use"], "content": result["content"]})
    assert result["metadata"]["semantic_generalization_validated"] is False
    candidate = proof["rules"][0]["candidate"]
    for current, expected in [(17.5, 32.5), (8.0, 42.0)]:
        assert relations.instantiate(candidate, {"displayCarStatus": {"fuelLevel": current}}) == expected


def test_unsupported_and_no_fit_records_remain_byte_equivalent(fuel_case):
    record, unit, episodes = fuel_case
    output, report = compile_records([record], [], episodes)
    assert output == [record] and report["decisions"]["m"]["status"] == "out_of_scope"
    unit["observations"][0]["target"]["args"]["fuelAmount"] = 30.
    output, report = compile_records([record], [unit], episodes)
    assert output == [record] and report["decisions"]["m"]["status"] == "no_candidate"


def test_id_instruction_uses_current_role_and_fixed_candidate_selection():
    unit = {"condition": "current_created_booking", "target_tool": "cancel_booking", "parameter": "booking_id"}
    candidate = {"kind": "copy", "producer": "book_flight", "path": ["booking_id"]}
    text = render_rule(unit, candidate)
    assert "3426812" not in text and "call_index" not in text
    assert 'book_flight_response["booking_id"]' in text and "do not book another flight" in text
    assert relations.instantiate(candidate, {"book_flight": {"booking_id": "7822128"}}) == "7822128"


def test_missing_producer_evidence_is_not_silently_published(fuel_case):
    record, unit, episodes = fuel_case
    # The provided fitting observations cannot substitute for the real calls.
    episodes["e"]["calls"][0] = {"tool_id": "other", "args": {}, "output": "{}"}
    with pytest.raises(ValueError, match="producer"):
        compile_records([record], [unit], episodes)


def test_conflicting_relations_in_one_record_fall_back(fuel_case):
    record, unit, episodes = fuel_case
    other = copy.deepcopy(unit)
    other["unit_id"] = "u2"
    other["constants"][0]["value"] = 60.
    other["observations"][0].update(episode_id="e2")
    other["observations"][0]["target"]["args"]["fuelAmount"] = 49.5
    other["all_refs"] = [{"episode_id": "e2", "call_index": 1}]
    record["metadata"]["evidence_refs"].append({"episode_id": "e2", "call_index": 1, "success": True})
    output, report = compile_records([record], [unit, other], episodes)
    assert output == [record] and report["decisions"]["m"]["status"] == "conflicting_relations"


@pytest.fixture
def fresh_inputs(tmp_path, fuel_case):
    record, unit, episodes = copy.deepcopy(fuel_case)
    calls = episodes["e"]["calls"]
    data, trials, source = (tmp_path / name for name in ("data", "trials", "source"))
    for directory in (data, trials, source):
        directory.mkdir()
    schemas = unit["schemas"]
    schemas["fillFuelTank"]["description"] = "The tank can hold up to 50 gallons."
    task = {"id": "synthetic_train", "extra": {"initial_config": {"VehicleControlAPI": {"fuelLevel": 10.5}}},
            "tools": [{"function": {"name": name, **schema}} for name, schema in schemas.items()]}
    train = data / "train.jsonl"
    train.write_text(json.dumps(task) + "\n")
    payload = {"episode_id": "e", "task_id": task["id"], "tool_calls": calls}
    trial = {"episode_id": "e", "task_id": task["id"], "base_seed": 91,
             "replay_payload": payload, "replay_sha256": canonical_hash(payload)}
    (trials / "trial.json").write_text(json.dumps(trial))
    call = calls[1]
    record["metadata"]["evidence_refs"] = [{
        "episode_id": "e", "call_index": 1, "tool_id": call["tool_id"], "success": call["success"],
        "execution_status": call["execution_status"], "turn_id": call["turn_id"],
        "arguments_sha256": canonical_hash(call["args"]), "output_sha256": canonical_hash(call["output"]),
        "goal_sha256": canonical_hash(call["state"]["user_goal"]),
    }]
    (source / "memories.json").write_text(json.dumps([record]))
    (source / "memories_index.json").write_text(json.dumps({"m": {
        "search_text": record["when_to_use"], "vector": [1.0, 0.0]}}))
    return train, trials, source, tmp_path / "new_bank"


def test_fresh_synthetic_bank_compiles_without_network(fresh_inputs, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("network attempted"))
    manifest = build_bank(*fresh_inputs)
    output = fresh_inputs[-1]
    assert manifest["report"]["compiled_records"] == 1
    assert (output / "memories_index.json").read_bytes() == (fresh_inputs[2] / "memories_index.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_bank(*fresh_inputs)
    with pytest.raises(ValueError, match="uncompiled"):
        build_bank(fresh_inputs[0], fresh_inputs[1], output, output.parent / "again")


def test_fresh_replay_tamper_and_duplicate_fail_before_write(fresh_inputs):
    trial_path = fresh_inputs[1] / "trial.json"
    original = trial_path.read_text()
    trial = json.loads(original)
    trial["replay_payload"]["reward"] = -100
    trial_path.write_text(json.dumps(trial))
    with pytest.raises(ValueError, match="replay hash mismatch"):
        build_bank(*fresh_inputs)
    trial_path.write_text(original)
    (fresh_inputs[1] / "duplicate.json").write_text(original)
    with pytest.raises(ValueError, match="duplicate training identity"):
        build_bank(*fresh_inputs)
    assert not fresh_inputs[-1].exists()


def test_fresh_evidence_tamper_fails_before_write(fresh_inputs):
    path = fresh_inputs[2] / "memories.json"
    records = json.loads(path.read_text())
    records[0]["metadata"]["evidence_refs"][0]["arguments_sha256"] = "tampered"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="source evidence mismatch"):
        build_bank(*fresh_inputs)
    assert not fresh_inputs[-1].exists()
