"""Compile train-fitted G2 expressions into frozen memory instructions.

The scope and object-role adapters come from offline_relations; they are
handwritten. Selection is G2/V0 (fixed first candidate), not unique-source
identification or environment validation. Unsupported records stay unchanged.
"""
from __future__ import annotations

import copy

from tevo.experiment import offline_relations as relations
from tevo.reproducibility import canonical_hash


def render_rule(unit: dict, candidate: dict) -> str:
    """Render current execution roles, never historical arguments or call IDs."""
    producer = candidate["producer"]
    condition = {
        "fill_to_full": "filling the same vehicle's tank to full",
        "current_market_price": "placing an order at the current market price",
        "current_created_order": "operating on an order created in this execution",
        "current_created_booking": "operating on a booking created in this execution",
    }[unit["condition"]]
    source = {
        "displayCarStatus": 'Read displayCarStatus(option="fuel") for the current vehicle immediately before filling.',
        "get_stock_info": "Read get_stock_info for the same symbol as the current order.",
        "place_order": "Use the place_order response for the order being operated on; do not create another order to obtain an ID.",
        "book_flight": "Use the book_flight response for the booking being operated on; do not book another flight to obtain an ID.",
        "get_order_details": "Use the get_order_details response for that same current order, identified by its creation response.",
        "retrieve_invoice": "Use the retrieve_invoice response for that same current booking, identified by its creation response.",
    }[producer]
    path = "".join(f"[{relations.canonical(part)}]" for part in candidate["path"])
    expression = f"{producer}_response{path}"
    if candidate["kind"] == "difference":
        expression = f"{relations.canonical(candidate['constant'])} - {expression}"
    return (
        f"Only when {condition}: {source} Wait for the response before calling "
        f"{unit['target_tool']}; do not submit the dependent calls in the same batch. "
        f"Set {unit['target_tool']}.{unit['parameter']} = {expression}. "
        "Bind other arguments to the current task. If the required response or object "
        "is missing or ambiguous, obtain or clarify it first; do not copy a historical "
        "value or invent an identifier."
    )


def rule_evidence(unit: dict, candidate: dict, episodes: dict) -> list[dict]:
    """Include producer and target calls from every fitting observation."""
    refs = {}
    for observation in unit["observations"]:
        episode = observation["episode_id"]
        target_index = observation["call_index"]
        calls = episodes[episode]["calls"]
        producer = candidate["producer"]
        indices = []
        for index, call in enumerate(calls[:target_index]):
            if call["tool_id"] != producer or relations.errored(relations.parse_output(call.get("output"))):
                continue
            if producer == "displayCarStatus" and call["args"].get("option") != "fuel":
                continue
            if producer == "get_stock_info" and call["args"].get("symbol") != observation["target"]["args"].get("symbol"):
                continue
            indices.append(index)
        if len(indices) != 1:
            raise ValueError("fitting evidence lacks a unique producer role")
        for index in [indices[0], target_index]:
            call = calls[index]
            goal = call.get("goal_snapshot") or call.get("state", {}).get("user_goal", "")
            refs[(episode, index)] = {
                "episode_id": episode, "call_index": index, "tool_id": call["tool_id"],
                "turn_id": call.get("turn_id"), "success": call.get("success"),
                "execution_status": call.get("execution_status"),
                "arguments_sha256": canonical_hash(call["args"]),
                "output_sha256": canonical_hash(call.get("output")),
                "goal_sha256": canonical_hash(goal),
            }
    return [refs[key] for key in sorted(refs)]


def compile_records(records: list[dict], units: list[dict], episodes: dict) -> tuple[list[dict], dict]:
    """Match verified bank target anchors to train units; change content only.

    Callers must validate the source bank/index, trial hashes and evidence joins
    before entering this function. No development A/B scores select a rule.
    """
    lookup = {}
    for unit in units:
        for ref in unit["all_refs"]:
            key = (ref["episode_id"], ref["call_index"])
            if key in lookup and lookup[key]["unit_id"] != unit["unit_id"]:
                raise ValueError("target anchor belongs to multiple relation units")
            lookup[key] = unit
    pools = {u["unit_id"]: relations.pool(u, "G2") for u in units}
    decisions, compiled = {}, []
    for record in records:
        matched = {}
        for ref in record["metadata"].get("evidence_refs", []):
            unit = lookup.get((ref["episode_id"], ref["call_index"]))
            if unit and ref.get("success") is True:
                matched[unit["unit_id"]] = unit
        if not matched or any(not pools[uid] for uid in matched):
            decisions[record["record_id"]] = {"status": "out_of_scope" if not matched else "no_candidate"}
            compiled.append(copy.deepcopy(record))
            continue
        # Different target operations may coexist. Conflicting instructions for
        # the same operation/condition cannot both be injected into one record.
        selected = {}
        conflict = False
        for uid, unit in matched.items():
            key = (unit["condition"], unit["target_tool"], unit["parameter"])
            candidate = relations.canonical(pools[uid][0])
            if key in selected and selected[key] != candidate:
                conflict = True
            selected[key] = candidate
        if conflict:
            decisions[record["record_id"]] = {"status": "conflicting_relations"}
            compiled.append(copy.deepcopy(record))
            continue
        rules, refs, contracts = {}, {}, {}
        for uid, unit in sorted(matched.items()):
            candidate = pools[uid][0]
            identity = relations.canonical([unit["condition"], unit["target_tool"], unit["parameter"], candidate])
            rules[identity] = render_rule(unit, candidate)
            for ref in rule_evidence(unit, candidate, episodes):
                refs[(ref["episode_id"], ref["call_index"])] = ref
            for tool in {unit["target_tool"], candidate["producer"]}:
                contracts[tool] = canonical_hash(unit["schemas"][tool])
        derived = copy.deepcopy(record)
        derived["content"] = "\n".join(rules[key] for key in sorted(rules))
        metadata = derived["metadata"]
        source_render = {key: copy.deepcopy(value) for key, value in metadata.items() if key.startswith("rendered_evidence_")}
        evidence = [refs[key] for key in sorted(refs)]
        episode_ids = sorted({ref["episode_id"] for ref in evidence})
        metadata.update(
            content_profile="g2_parameter_source_v1", validation_status="training_fit_only",
            semantic_generalization_validated=False,
            validation_scope=["source_evidence_join", "restricted_grammar", "type", "recorded_argument_fit"],
            generated_text_sha256=canonical_hash({"when_to_use": derived["when_to_use"], "content": derived["content"]}),
            rendered_evidence_refs=evidence, rendered_evidence_episode_ids=episode_ids,
            rendered_evidence_count=len(episode_ids), rendered_evidence_count_unit="distinct_episode_ids",
            rendered_evidence_verified=True,
            parameter_compilation={
                "schema": "g2_parameter_source_v1", "proposer": "G2", "policy": "V0",
                "candidate_selection": "fixed_first_training_fit; not unique-source identification",
                "scope_and_object_roles": "handwritten offline_relations adapters",
                "source_content": record["content"], "source_render": source_render,
                "source_validation_status": record["metadata"].get("validation_status"),
                "unit_ids": sorted(matched), "contracts_sha256": contracts,
                "rules": [{"unit_id": uid, "target_tool": unit["target_tool"], "parameter": unit["parameter"],
                           "candidate": pools[uid][0], "candidate_count": len(pools[uid]), "constants": unit["constants"]}
                          for uid, unit in sorted(matched.items())],
            },
        )
        compiled.append(derived)
        decisions[record["record_id"]] = {"status": "compiled", "unit_ids": sorted(matched)}
    return compiled, {"unit_count": len(units), "units_with_candidate": sum(bool(p) for p in pools.values()),
                      "record_count": len(records), "compiled_records": sum(d["status"] == "compiled" for d in decisions.values()),
                      "decisions": decisions}
