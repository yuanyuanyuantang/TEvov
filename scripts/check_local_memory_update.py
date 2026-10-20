#!/usr/bin/env python3
"""Stage A only: CPU development audit and pure fragment-update prototype.

No actor, environment replay, network, bank writes or automatic later stages.
Oracle adapters are deliberately limited, hand-defined diagnostic predicates;
they do not discover applicability or establish unique causal provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import socket
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rewrite_procedural_memory as rw

fresh = rw.fresh
# Independent adapter fields, NOT inferred from the candidate or actor argument.
IDS = {
    "cancel_booking": ("book_flight", "booking_id", "booking_id", r"cancel", r"flight|trip|reservation|booking"),
    "retrieve_invoice": ("book_flight", "booking_id", "booking_id", r"invoice", r"invoice"),
    "get_order_details": ("place_order", "order_id", "order_id", r"detail", r"order|transaction"),
    "resolve_ticket": ("get_ticket", "ticket_id", "id", r"resolv", r"ticket|\bit\b"),
    "retweet": ("post_tweet", "tweet_id", "id", r"retweet", r"tweet|\bit\b"),
}
FULL = re.compile(r"\bfull\b|\bfill (?:up |out )?(?:my |the )?(?:fuel )?tank", re.I)
CURRENT = re.compile(r"\b(?:just|recent|latest|this|that|its|it|once|secured|confirmed|positioned|shared)\b", re.I)


def family(rule):
    if rule["target_tool"] == "fillFuelTank" and rule["parameter"] == "fuelAmount":
        return "capacity_difference"
    adapter = IDS.get(rule["target_tool"])
    return "returned_id" if adapter and rule["parameter"] == adapter[1] else None


def goal_in_scope(rule, goal):
    """Denominator uses task text, before seeing actor outcomes; ambiguity stays in."""
    if family(rule) == "capacity_difference":
        return bool(re.search(r"fuel|tank", goal, re.I) and FULL.search(goal))
    if family(rule) == "returned_id":
        *_, verb, obj = IDS[rule["target_tool"]]
        return bool(re.search(verb, goal, re.I) and re.search(obj, goal, re.I))
    return False


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verdict(rule, schemas, calls, goal, turn):
    """Compare expression against goal/contract/response witness, then actor separately.

    First target attempt is scored even if later repaired. Missing attempts stay
    failures in behavior denominators. Whole-task reward is never an input here.
    """
    targets = [c for c in calls if c["tool_id"] == rule["target_tool"] and c["turn_id"] == turn]
    target = targets[0] if targets else None
    result = {"judgment": "unknown", "reason": "", "behavior": "unresolved" if target else "not_reached",
              "first_qualified": False, "target_call": target["call_index"] if target else None}

    def unknown(reason):
        return result | {"reason": reason}

    if not goal_in_scope(rule, goal):
        return unknown("outside_declared_goal")
    adapter_producer = "displayCarStatus" if family(rule) == "capacity_difference" else IDS[rule["target_tool"]][0]
    if rule["producer"] != adapter_producer:
        return unknown("producer_outside_independent_adapter")
    if not target:
        return unknown("target_not_reached")
    for name, key in ((rule["target_tool"], "target_schema_sha256"), (rule["producer"], "producer_schema_sha256")):
        if name not in schemas or fresh.canonical_hash(schemas[name]) != rule[key]:
            return unknown("contract_changed_or_missing")
    # Conditional branches require a separate branch witness; actor choosing to
    # execute the target is not itself proof that the condition was satisfied.
    if re.search(r"\bif\b|\bin case\b|\bunless\b", goal, re.I):
        return unknown("conditional_goal_unresolved")
    producers = [c for c in calls if c["tool_id"] == rule["producer"]
                 and c["turn_id"] <= turn and c["assistant_message"] <= target["assistant_message"]
                 and fresh.target_outcome(c) == "local_no_reported_error"]
    if family(rule) == "capacity_difference":
        if re.search(r"\d+(?:\.\d+)?\s*(?:gallons?|liters?)", goal, re.I):
            return unknown("explicit_quantity_goal")
        # Adapter contract and observed path are independent of the candidate.
        caps = re.findall(r"hold up to (\d+(?:\.\d+)?) gallons", schemas["fillFuelTank"].get("description", ""))
        prop = schemas["fillFuelTank"].get("parameters", {}).get("properties", {}).get("fuelAmount", {})
        if len(caps) != 1 or "additional fuel" not in prop.get("description", ""):
            return unknown("capacity_or_addition_contract_unresolved")
        cap = float(caps[0])
        producers = [p for p in producers if p["args"].get("option") == "fuel"
                     and p["response_message"] < target["assistant_message"]]
        if not producers:
            return unknown("no_prior_current_state_read")
        producer = producers[-1]
        # No hidden-state simulation: stale or concurrently modified reads abstain.
        if any(c["tool_id"] not in {"displayCarStatus", "gallon_to_liter", "liter_to_gallon"}
               and c["assistant_message"] >= producer["assistant_message"]
               and c["call_index"] != target["call_index"]
               and c["assistant_message"] <= target["assistant_message"] for c in calls):
            return unknown("state_may_have_changed_after_read")
        value = fresh.parse_output(producer["output"])
        state = value.get("fuelLevel") if isinstance(value, dict) else None
        if not number(state) or not 0 <= state <= cap:
            return unknown("invalid_or_missing_current_fuel")
        expected = cap - state
        witness = {"kind": "contract_capacity_constraint", "capacity": cap, "state": state,
                   "required_final_level": cap, "oracle_field": ["fuelLevel"]}
    else:
        if not CURRENT.search(goal):
            return unknown("current_object_role_not_explicit")
        # Explicit IDs and multiple objects need a resolver this prototype lacks.
        if re.search(r"\b\d{4,}\b|\b(?:id|number)\s*[#:=]?\s*[\w-]*\d", goal, re.I):
            return unknown("explicit_identifier_or_object_selection_unresolved")
        oracle_field = IDS[rule["target_tool"]][2]
        values = []
        for p in producers:
            output = fresh.parse_output(p["output"])
            value = output.get(oracle_field) if isinstance(output, dict) else None
            if isinstance(value, str) and value or number(value):
                values.append((p, value))
        if not values:
            return unknown("no_successful_current_producer")
        distinct = {fresh.canonical_hash(v) for _, v in values}
        if len(distinct) != 1:
            return unknown("multiple_current_objects")
        producer, expected = values[-1]
        witness = {"kind": "explicit_current_role_and_response", "oracle_field": [oracle_field],
                   "object_role": "single_observed_current_object_not_unique_causal_proof"}
    try:
        actual_source = fresh.at_path(fresh.parse_output(producer["output"]), rule["source_path"])
        if rule["kind"] == "state_difference":
            if not number(actual_source) or not number(rule.get("constant")):
                return unknown("invalid_expression_types")
            predicted = rule["constant"] - actual_source
        elif rule["kind"] == "tool_output":
            predicted = actual_source
        else:
            return unknown("unsupported_expression")
    except (KeyError, IndexError, TypeError, ValueError):
        return unknown("candidate_path_not_observed")
    agrees = fresh.equal(predicted, expected)
    waited = producer["response_message"] < target["assistant_message"]
    parameter_matches = fresh.equal(target["args"].get(rule["parameter"]), expected)
    behavior = "followed" if waited and parameter_matches else "execution_deviation"
    return result | {"judgment": "supported" if agrees else "refuted", "reason": "independent_local_constraint",
                     "behavior": behavior, "first_qualified": waited and parameter_matches,
                     "waited": waited, "parameter_matches_witness": parameter_matches,
                     "source_call": producer["call_index"], "source_call_sha256": fresh.canonical_hash(producer),
                     "target_call_sha256": fresh.canonical_hash(target), "witness": witness,
                     "expected": expected, "predicted": predicted,
                     "local_target_outcome": fresh.target_outcome(target)}


def propose_fragment_update(record, options, decisions, replacements=None):
    """Pure proposal: retain/suspend/replace fragments, never edit the live bank.

    Unknown or conflicting evidence does not grant publication. Replacement
    candidates must already have their own all-supported decision list.
    """
    replacements = replacements or {}
    selected = record["metadata"].get("selected_options", [])
    kept, suspended, replaced = [], [], {}
    for identity in selected:
        status = set(decisions.get(identity, []))
        if not status <= {"supported", "refuted", "unknown"}:
            raise ValueError("invalid local judgment")
        if "refuted" in status and "supported" not in status:
            new_id = replacements.get(identity)
            if new_id in options and decisions.get(new_id) and set(decisions[new_id]) == {"supported"}:
                if family(options[identity]) != family(options[new_id]) or any(
                        options[identity][k] != options[new_id][k] for k in ("target_tool", "parameter", "producer")):
                    raise ValueError("replacement changes operation")
                kept.append(new_id)
                replaced[identity] = new_id
            else:
                suspended.append(identity)
        else:
            kept.append(identity)
    if not suspended and not replaced:
        return {"action": "retain", "record_id": record["record_id"], "changed": False}
    if len(kept) != len(set(kept)):
        raise ValueError("duplicate replacement")
    original = record["content"]
    suffix = "\n" + record["metadata"]["replacement_text"]
    if not original.endswith(suffix):
        raise ValueError("fragment boundary mismatch")
    ordinary = original[:-len(suffix)]
    content = ordinary + ("\n" + rw.render_attachment([options[i] for i in kept]) if kept else "")
    new_id = "mem_" + fresh.canonical_hash([record["when_to_use"], content, record["tool_tags"]])[:16]
    return {"action": "replace" if replaced else "suspend", "changed": True,
            "source_record_id": record["record_id"], "record_id": new_id, "content": content,
            "ordinary_preserved": content.startswith(ordinary), "selected_options": kept,
            "suspended": suspended, "replaced": replaced, "publication_status": "proposal_only_not_released"}


def task_goals(task):
    return ["\n".join(m["content"] for m in turn if m["role"] == "user")
            for turn in task["extra"]["question"]]


def summarize(events):
    counts = Counter(e["judgment"] for e in events)
    return {"events": len(events), "tasks": len({e["task_id"] for e in events}),
            "episodes": len({e["episode_id"] for e in events}), "judgments": dict(counts),
            "coverage": (counts["supported"] + counts["refuted"]) / len(events) if events else 0,
            "abstention_rate": counts["unknown"] / len(events) if events else 1,
            "behaviors": dict(Counter(e["behavior"] for e in events)),
            "first_qualified": sum(e["first_qualified"] for e in events),
            "unknown_reasons": dict(Counter(e["reason"] for e in events if e["judgment"] == "unknown")),
            "supported_with_whole_task_failure": sum(e["judgment"] == "supported" and e["whole_task_reward"] == 0 for e in events)}


def audit(bank, train_tasks, train, dev_tasks, development, contexts):
    start = time.monotonic()
    options, _ = rw.compile_options(train_tasks, train)
    selected = {i for r in bank for i in r["metadata"].get("selected_options", [])}
    if not selected <= options.keys():
        raise ValueError("published relation absent from train reconstruction")
    active = {i: options[i] for i in sorted(selected) if family(options[i])}
    if not active or {family(o) for o in active.values()} != {"returned_id", "capacity_difference"}:
        raise ValueError("two-family bank required")
    events = []
    for split, tasks, trials in [("training_reused_not_independent", train_tasks, train),
                                 ("development_previously_evaluated", dev_tasks, development)]:
        for trial in trials:
            task = tasks[trial["task_id"]]
            calls = trial["replay_payload"]["tool_calls"]
            # Strip injected memory from goals: predicates see ORIGINAL task text.
            for turn, goal in enumerate(task_goals(task)):
                for identity, rule in active.items():
                    if rule["target_tool"] not in rw.schemas_for(task) or not goal_in_scope(rule, goal):
                        continue
                    memories = contexts[split].get(trial["task_id"], {}).get("memories", [])
                    exposed = any(identity in m.get("metadata", {}).get("selected_options", []) for m in memories)
                    events.append({"split": split, "task_id": trial["task_id"], "episode_id": trial["episode_id"],
                        "repeat_index": trial.get("repeat_index"), "family": family(rule), "option_id": identity,
                        "turn": turn, "original_goal_sha256": fresh.canonical_hash(goal), "actually_exposed": exposed,
                        "whole_task_reward": trial["reward"],
                        **verdict(rule, rw.schemas_for(task), calls, goal, turn)})
    decisions = {i: [e["judgment"] for e in events if e["option_id"] == i] for i in active}
    proposals = [propose_fragment_update(r, options, decisions) for r in bank if r["metadata"].get("selected_options")]
    local_seconds = time.monotonic() - start
    # Same initial candidates, exactly the same raw logs; no validation actors or
    # oracle requests for either method. Refitting remains training-fit evidence.
    start = time.monotonic()
    refitted, _ = rw.compile_options(train_tasks | dev_tasks, train + development)
    baseline = {i: ("retain" if i in refitted else "unsupported_on_refit") for i in active}
    baseline_seconds = time.monotonic() - start
    all_stats = summarize(events)
    family_stats = {f: summarize([e for e in events if e["family"] == f]) for f in ["returned_id", "capacity_difference"]}
    changed = [p for p in proposals if p["changed"]]
    gates = {"coverage_at_least_50_percent": all_stats["coverage"] >= .5,
             "both_families_nonabstaining": all(s["coverage"] > 0 for s in family_stats.values()),
             "real_refuted_fragment_found": any(e["judgment"] == "refuted" for e in events),
             "concrete_fragment_update": bool(changed),
             "increment_beyond_simple_refit": any(p["changed"] and any(baseline.get(i) == "retain" for i in p.get("suspended", [])) for p in changed)}
    return {"kind": "local_memory_update_stage_a", "research_status": "development_feasibility_not_efficacy",
            "denominator": "one episode x selected in-scope relation x original goal turn; includes missing target and unresolved roles; deduplicates duplicate bank records",
            "unassessed_gaps": "out-of-pool relations and semantic omissions are not certified absent; no automatic applicability discovery",
            "selected_relations": len(selected), "in_scope_relations": len(active),
            "out_of_scope_relation_ids": sorted(selected - active.keys()), "families": family_stats,
            "all": all_stats, "by_split": {s: summarize([e for e in events if e["split"] == s]) for s in contexts},
            "actually_exposed_development": summarize([e for e in events if e["actually_exposed"]]),
            "events": events, "fragment_proposals": proposals, "changed_records": len(changed),
            "simple_refit": {"initial_selected_status": baseline, "new_candidate_count_not_published": len(refitted.keys() - options.keys()),
                "same_raw_evidence": True, "same_initial_candidate_pool_for_decisions": True,
                "local_audit_seconds_including_initial_compile": local_seconds, "refit_seconds": baseline_seconds,
                "reliability_difference": "no independent annotated accuracy estimate; not inferred from agreement",
                "retained_original_candidates": sum(v == "retain" for v in baseline.values())},
            "gates": gates, "advance_to_stage_b": all(gates.values()),
            "cost": {"llm": 0, "embedding": 0, "environment_check_attempts": 0, "environment_tool_calls": 0},
            "limits": ["handwritten lexical role adapters, not independently labeled accuracy", "no causal actor comparison",
                       "supported does not establish unique provenance", "supported plus task failure does not identify unrelated failure cause"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank_dir = ROOT / "outputs/memory_rebuild_20261018/released_bank"
    dev_dir = ROOT / "outputs/bfcl_pass4_rebuild_variant_20261018"
    train_dir = ROOT / "outputs/train400_20261014/trials"
    fixed = [bank_dir / n for n in ["memories.json", "memories_index.json", "manifest.json", "freeze.json"]]
    fixed += [dev_dir / n for n in ["contexts.json", "manifest.json", "regrade.json"]]
    fixed += [fresh.TRAIN, fresh.VAL, Path(__file__).resolve(), Path(rw.__file__), Path(fresh.__file__),
              ROOT / "tests/test_local_memory_update.py",
              ROOT / "outputs/动态记忆更新有限方案_20261018.md"]
    trial_paths = sorted(train_dir.glob("*.json")) + sorted((dev_dir / "trials").glob("*.json"))
    if len(trial_paths) != 1000:
        raise ValueError("expected train400 plus variant development600; no silent partial audit")
    for tree in [bank_dir, dev_dir, train_dir, fresh.TRAIN.parent]:
        if args.output.resolve().is_relative_to(tree.resolve()):
            raise ValueError("output overlaps protected input tree")
    pins = {str(p): fresh.sha256_file(p) for p in fixed + trial_paths}
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    fresh.write_private_json(args.output / "preflight.json", {"input_sha256": pins,
        "stage": "A_only", "denominator_fixed_in_code_before_analysis": True,
        "new_actor_limit_this_stage": 0, "bank_writes_allowed": False}, exclusive=True)
    attempts = []

    def deny(*a, **k):
        attempts.append("blocked")
        raise RuntimeError("stage A forbids network")

    with patch.object(socket.socket, "connect", deny), patch.object(socket.socket, "connect_ex", deny), patch.object(socket, "create_connection", deny):
        tasks, trials, _ = fresh.load_build_inputs(train_dir)
        bank = json.loads((bank_dir / "memories.json").read_text())
        index = json.loads((bank_dir / "memories_index.json").read_text())
        freeze = json.loads((bank_dir / "freeze.json").read_text())
        if freeze["manifest_sha256"] != fresh.sha256_file(bank_dir / "manifest.json") or any(
                fresh.sha256_file(bank_dir / n) != h for n, h in freeze["outputs_sha256"].items()):
            raise ValueError("bank freeze mismatch")
        fresh.validate_frozen_bank_index(bank, index)
        dev_tasks = {t["id"]: t for t in fresh.load_jsonl(fresh.VAL)}
        ctx = json.loads((dev_dir / "contexts.json").read_text())
        from tevo.experiment.suite import validate_cached_tevo_contexts
        validate_cached_tevo_contexts(ctx, bank, list(dev_tasks))
        development, grid = [], set()
        manifest = json.loads((dev_dir / "manifest.json").read_text())
        if fresh.canonical_hash(manifest["fingerprints"]) != manifest["manifest_sha256"]:
            raise ValueError("development manifest self hash mismatch")
        for path in sorted((dev_dir / "trials").glob("*.json")):
            trial = json.loads(path.read_text())
            identity = (trial["task_id"], trial["repeat_index"])
            if trial["status"] != "complete" or identity in grid or trial["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("development identity/status/manifest mismatch")
            if trial["memory_context_sha256"] != fresh.canonical_hash(ctx["tevo"][trial["task_id"]]):
                raise ValueError("development context binding mismatch")
            grid.add(identity)
            trial["replay_payload"] = {"tool_calls": fresh.trace_calls(trial)}
            development.append(trial)
        if grid != {(t, r) for t in dev_tasks for r in range(4)}:
            raise ValueError("development task/repeat grid mismatch")
        report = audit(bank, tasks, trials, dev_tasks, development,
            {"training_reused_not_independent": {}, "development_previously_evaluated": ctx["tevo"]})
    report["cost"]["blocked_network_attempts"] = len(attempts)
    report["inputs_unchanged"] = all(fresh.sha256_file(Path(p)) == h for p, h in pins.items())
    if not report["inputs_unchanged"] or attempts:
        raise RuntimeError("stage A integrity/network gate failed")
    report["next_action"] = "stage_B_allowed_by_plan" if report["advance_to_stage_b"] else "stop_no_new_actors_no_bank_changes"
    fresh.write_private_json(args.output / "report.json", report, exclusive=True)
    print(json.dumps({k: report[k] for k in ["all", "families", "changed_records", "gates", "advance_to_stage_b", "cost", "inputs_unchanged"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
