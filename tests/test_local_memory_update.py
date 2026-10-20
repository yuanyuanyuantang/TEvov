"""Synthetic correctness checks; these counterexamples are NOT research evidence."""
from copy import deepcopy
import json

import pytest

from scripts import check_local_memory_update as audit


def fuel_example(amount=40, response=10):
    schemas = {
        "displayCarStatus": {"name": "displayCarStatus"},
        "fillFuelTank": {"name": "fillFuelTank", "description": "The fuel tank can hold up to 50 gallons.",
                        "parameters": {"properties": {"fuelAmount": {"description": "additional fuel in gallons"}}}},
    }
    rule = {"option_id": "r", "kind": "state_difference", "target_tool": "fillFuelTank", "parameter": "fuelAmount",
            "producer": "displayCarStatus", "source_path": ["fuelLevel"], "selectors": {"option": "fuel"}, "constant": 50,
            "target_schema_sha256": audit.fresh.canonical_hash(schemas["fillFuelTank"]),
            "producer_schema_sha256": audit.fresh.canonical_hash(schemas["displayCarStatus"])}
    trial = {"episode_id": "synthetic", "task_history": [
        {"role": "user", "content": "Fill the fuel tank to full."},
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "displayCarStatus", "arguments": {"option": "fuel"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": json.dumps({"fuelLevel": response})},
        {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "fillFuelTank", "arguments": {"fuelAmount": amount}}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"fuelLevel":50}'}]}
    return rule, schemas, audit.fresh.trace_calls(trial)


def booking_example():
    rule, _, calls = fuel_example()
    schemas = {"book_flight": {"name": "book_flight"}, "cancel_booking": {"name": "cancel_booking"}}
    rule.update(kind="tool_output", producer="book_flight", target_tool="cancel_booking", parameter="booking_id",
                source_path=["booking_id"], selectors={},
                producer_schema_sha256=audit.fresh.canonical_hash(schemas["book_flight"]),
                target_schema_sha256=audit.fresh.canonical_hash(schemas["cancel_booking"]))
    calls[0].update(tool_id="book_flight", args={"destination": "current city"}, output='{"booking_id":"new_42"}')
    calls[1].update(tool_id="cancel_booking", args={"booking_id": "new_42"}, output='{"cancel_status":true}')
    return rule, schemas, calls


def check(example, goal="Fill the fuel tank to full."):
    rule, schemas, calls = example
    return audit.verdict(rule, schemas, calls, goal, 0)


def test_fuel_truth_does_not_use_actor_argument_as_label():
    correct = check(fuel_example())
    deviation = check(fuel_example(amount=39.5))
    assert correct["judgment"] == deviation["judgment"] == "supported"
    assert correct["first_qualified"] and not deviation["first_qualified"]
    assert deviation["behavior"] == "execution_deviation" and deviation["expected"] == 40
    example = fuel_example()
    example[0]["constant"] = 49.5
    bad_rule = check(example)
    assert bad_rule["judgment"] == "refuted" and bad_rule["predicted"] == 39.5


@pytest.mark.parametrize("reason", ["missing", "stale", "unknown_intervening_tool", "ambiguous_branch", "explicit_amount", "schema_drift", "invalid_state"])
def test_unknown_is_not_a_rule_counterexample(reason):
    rule, schemas, calls = fuel_example()
    goal = "Fill the fuel tank to full."
    if reason == "missing": calls = calls[1:]
    if reason == "stale":
        calls.insert(1, {**deepcopy(calls[0]), "tool_id": "startEngine", "call_index": 77})
    if reason == "unknown_intervening_tool":
        calls.insert(1, {**deepcopy(calls[0]), "tool_id": "unknownStateMutator", "call_index": 77})
    if reason == "ambiguous_branch": goal = "If needed, fill the fuel tank to full."
    if reason == "explicit_amount": goal = "Fill the fuel tank to full with 30 gallons."
    if reason == "schema_drift": schemas["fillFuelTank"]["description"] += " changed"
    if reason == "invalid_state": calls[0]["output"] = '{"fuelLevel":true}'
    assert audit.verdict(rule, schemas, calls, goal, 0)["judgment"] == "unknown"


def test_booking_wait_and_value_are_separate_axes_and_errors_do_not_refute():
    example = booking_example()
    goal = "Cancel the reservation I just made."
    good = check(example, goal)
    assert good["first_qualified"]
    example[2][1]["args"]["booking_id"] = "invented"
    example[2][1]["output"] = '{"error":"Booking not found"}'
    bad_actor = check(example, goal)
    assert bad_actor["judgment"] == "supported" and bad_actor["behavior"] == "execution_deviation"
    example[2][1]["args"]["booking_id"] = "new_42"
    example[2][1]["assistant_message"] = example[2][0]["assistant_message"]
    late = check(example, goal)
    assert late["judgment"] == "supported" and late["parameter_matches_witness"] and not late["waited"]
    example[2][1]["assistant_message"] = 5
    example[2][1]["output"] = '{"error":"unauthorized"}'
    assert check(example, goal)["judgment"] == "supported"


def test_multiple_objects_and_explicit_user_id_abstain():
    example = booking_example()
    another = deepcopy(example[2][0]); another["output"] = '{"booking_id":"other_7"}'
    another["call_index"] = 99
    example[2].insert(1, another)
    assert check(example, "Cancel the reservation I just made.")["reason"] == "multiple_current_objects"
    assert check(booking_example(), "Cancel this reservation ID 123456.")["judgment"] == "unknown"


def test_missing_target_stays_in_full_behavior_denominator():
    rule, schemas, calls = booking_example()
    row = audit.verdict(rule, schemas, calls[:1], "Cancel the reservation I just made.", 0)
    summary = audit.summarize([row | {"task_id": "t", "episode_id": "e", "whole_task_reward": 0}])
    assert summary["events"] == 1 and summary["coverage"] == 0 and summary["first_qualified"] == 0


def test_local_update_preserves_ordinary_and_handles_abstention_conflict_replacement():
    rule, _, _ = fuel_example()
    options = {"r": rule, "good": {**rule, "option_id": "good", "constant": 50}}
    ordinary = "Read the tank before refilling."
    suffix = audit.rw.render_attachment([rule])
    record = {"record_id": "old", "when_to_use": "Filling", "tool_tags": ["fillFuelTank"],
              "content": ordinary + "\n" + suffix,
              "metadata": {"selected_options": ["r"], "replacement_text": suffix}}
    saved = deepcopy(record)
    for status in [["unknown"], [], ["supported"], ["supported", "refuted"]]:
        assert not audit.propose_fragment_update(record, options, {"r": status})["changed"]
    suspend = audit.propose_fragment_update(record, options, {"r": ["refuted"]})
    assert suspend["action"] == "suspend" and suspend["content"] == ordinary
    assert suspend["record_id"] != "old"
    replaced = audit.propose_fragment_update(record, options, {"r": ["refuted"], "good": ["supported"]}, {"r": "good"})
    assert replaced["action"] == "replace" and replaced["selected_options"] == ["good"]
    assert record == saved
    with pytest.raises(ValueError, match="invalid local judgment"):
        audit.propose_fragment_update(record, options, {"r": ["reward=0"]})
    record["content"] += "unexpected"
    with pytest.raises(ValueError, match="fragment boundary"):
        audit.propose_fragment_update(record, options, {"r": ["refuted"]})


def test_reward_failure_does_not_punish_supported_rule():
    row = check(fuel_example()) | {"task_id": "t", "episode_id": "e", "whole_task_reward": 0}
    assert audit.summarize([row])["supported_with_whole_task_failure"] == 1
