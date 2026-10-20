"""Bounded relation semantics and actual BFCL local checks (no network)."""
import copy
import importlib.util
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("offline_relations_test", ROOT / "tevo/experiment/offline_relations.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fuel():
    return {"unit_id": "testfuel", "domain": "VehicleControlAPI", "target_tool": "fillFuelTank", "parameter": "fuelAmount",
            "goal": "Fill the tank completely", "condition": "fill_to_full",
            "schemas": {"fillFuelTank": {"parameters": {"properties": {"fuelAmount": {"type": "number"}}}}, "displayCarStatus": {}},
            "prefix": [{"tool": "displayCarStatus", "args": {"option": "fuel"}}], "target_args": {"fuelAmount": 39.5},
            "constants": [{"value": 50., "quote": "hold up to 50 gallons"}],
            "observations": [{"prior": [{"tool_id": "displayCarStatus", "args": {"option": "fuel"}, "output": '{"fuelLevel":10.5}'}],
                              "target": {"args": {"fuelAmount": 39.5}}}]}


RULE = {"kind": "difference", "producer": "displayCarStatus", "path": ["fuelLevel"], "constant": 50.}


def test_contract_only_grammar_and_original_fit():
    unit = fuel()
    assert r.pool(unit, "G1") == r.pool(unit, "G2") == [RULE]
    assert r.pool(unit, "G3", [RULE]) == [RULE]
    assert not r.candidate_valid(unit, {**RULE, "constant": 39.5})
    assert not r.candidate_valid(unit, {**RULE, "call_index": 7})
    assert not r.fits(unit, {"kind": "copy", "producer": "displayCarStatus", "path": ["fuelLevel"]})
    assert not r.same(True, 1)
    assert not r.same(float("nan"), 1)


def test_non_unique_historical_role_abstains():
    unit = fuel()
    unit["observations"][0]["prior"] *= 2
    assert r.pool(unit, "G2") == []


def test_no_candidate_and_multiple_survivors_are_not_success():
    valid = [{"status": "valid"}] * 3
    assert r.selection([], [])["status"] == "no_candidate"
    assert r.selection([RULE], [valid])["status"] == "bounded_checked"
    other = {"kind": "copy", "producer": "other", "path": ["field"]}
    assert r.selection([RULE, other], [valid, valid])["status"] == "unresolved"
    assert r.selection([RULE], [[{"status": "refuted"}] * 3])["status"] == "refuted"
    assert r.selection([RULE], [[{"status": "infrastructure_error"}] * 3])["status"] == "infrastructure_error"


def test_empty_publication_error_is_undefined():
    unit = fuel()
    m = r.metrics([unit], {"testfuel": {"candidate": None, "status": "no_candidate"}}, {"testfuel": []})
    assert m["E"] is None and m["C"] == 0 and m["Q"] == 0 and m["abstention_rate"] == 1


def test_selected_generator_must_pass_its_own_gate():
    domains = ["VehicleControlAPI", "TradingBot", "TravelAPI"]
    base = {"C": .8, "Q": .79, "E": .01, "published_domains": domains}
    summary = {g + v: dict(base) for g in ("G1", "G2", "G3") for v in ("+V", "+V0")}
    summary["G1+V"]["E"] = .02
    summary["G2+V"].update(E=.08, Q=.73)
    summary["G2+V0"]["E"] = .15
    summary["G3+V"].update(E=.09, Q=.72)
    decision = r.decide(summary)
    assert decision["selected_proposer"] == "G1" and not decision["validation_gate"]


def test_scope_excludes_literal_quantities_and_explicit_price():
    assert r.scope("fillFuelTank", "Fill the tank completely")
    assert not r.scope("fillFuelTank", "Fill the tank with 30 gallons")
    assert not r.scope("place_order", "Buy AAPL at $150")
    assert r.scope("place_order", "Buy AAPL at the current market price")
    assert not r.scope("retrieve_invoice", "Retrieve insurance invoice")


def test_real_fuel_checks_are_independent_and_network_free(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("network attempted"))
    check = r.LocalChecks(r.load_executor(ROOT), budget=20, previous=12)
    unit = fuel()
    state = {"variation_id": "smoke", "initial_config": {"VehicleControlAPI": {"fuelLevel": 17.5}}}
    result = check.run(unit, state, RULE)
    assert result["status"] == "valid" and result["argument"] == 32.5
    assert result["explicit_calls"] == result["public_calls"] == 2
    wrong = check.run(unit, state, {"kind": "copy", "producer": "displayCarStatus", "path": ["fuelLevel"]})
    assert wrong["status"] == "refuted"  # 35 gallons is accepted, but not full.
    assert check.attempts == 14


def test_missing_prefix_is_infrastructure_not_counterexample():
    unit = fuel()
    unit["prefix"] = []
    check = r.LocalChecks(r.load_executor(ROOT))
    result = check.run(unit, {"variation_id": "smoke", "initial_config": {"VehicleControlAPI": {}}}, RULE)
    assert result["status"] == "infrastructure_error" and check.attempts == 13


def test_fixed_variations_separate_A_B():
    unit = fuel()
    unit["initial_config"] = {"VehicleControlAPI": {"fuelLevel": 10.5}}
    a = [r.variation(unit, "A", i) for i in range(3)]
    b = [r.variation(unit, "B", i) for i in range(3)]
    assert not {r.digest(v["initial_config"]) for v in a} & {r.digest(v["initial_config"]) for v in b}
    assert unit["initial_config"]["VehicleControlAPI"]["fuelLevel"] == 10.5


def test_incomplete_B_never_scores_zero_error():
    unit = fuel()
    choice = {"testfuel": {"candidate": RULE, "status": "bounded_checked"}}
    result = r.metrics([unit], choice, {"testfuel": [{"status": "infrastructure_error"}] * 3})
    assert result["E"] is None and result["Q"] is None and result["scoring_incomplete"] == ["testfuel"]
