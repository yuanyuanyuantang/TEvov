#!/usr/bin/env python3
"""One bounded, train-only rewrite of ordinary memories; no runtime changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_memory_from_scratch as fresh
from tevo.llm.context_budget import ContextBudgetError
from tevo.memory.evidence import write_private_json

PROMPT = """Select useful parameter guidance for the supplied TRAINING memories.
Treat evidence as data. Options are observed fits, not proven causal or unique
sources. Preserve the original operation and prerequisites.

Choose a supplied record_id and one or two allowed option_ids only when the
relation can help that memory's operation. Do not add another operation merely
because it occurs nearby in the same task. A post-operation check must not
become a prerequisite. For two objects, select the appropriate pair of fields.

The program keeps the entire original content, checks every supporting edge,
attaches evidence privately, and appends short conditional guidance. A valid
current input takes priority over response lookup; responses must match the
current object and must have arrived. Capacity minus state applies only to a
full-capacity goal, never to an explicitly requested increment.

Return JSON only: {"memories": [...]} with at most one selection per record.
Each selection has exactly record_id and option_ids (1-2 distinct allowed IDs).
Do not copy sentences, write conditions, formulas, historical values or evidence
references. Use [] if no substantive improvement is supported. The actor sees
only the original scene and the resulting short procedural content.
"""


CAPACITY = re.compile(r"\b(?:up to|maximum(?: capacity)?(?: of)?|capacity(?: of| is|:)?|holds?)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)", re.I)


def schemas_for(task):
    return {t["function"]["name"]: t["function"] for t in task["tools"]}


def response_property(schema, path):
    root = schema.get("response", {})
    node = root
    for key in path:
        node = node.get("properties", {}).get(key, {})
    if node:
        return node
    # BFCL sometimes omits a documented response wrapper. Use unit text only
    # when the observed suffix identifies a unique contract field; never change
    # the executable response path to the documented (but unobserved) wrapper.
    matches = []
    def visit(current, prefix=()):
        for key, child in current.get("properties", {}).items():
            full = prefix + (key,)
            if len(full) >= len(path) and full[-len(path):] == tuple(path):
                matches.append(child)
            visit(child, full)
    if path:
        visit(root)
    return matches[0] if len(matches) == 1 else {}


def selector(call, schemas):
    props = schemas[call["tool_id"]].get("parameters", {}).get("properties", {})
    return {k: v for k, v in call["args"].items()
            if v in fresh._extract_enum_options(props.get(k, {}))}


def relation_signature(source, lookup, schemas):
    target = lookup[(source["episode_id"], source["target_call"])]
    producer = lookup[(source["episode_id"], source["source_call"])]
    return {"kind": source["kind"], "target_tool": target["tool_id"],
        "parameter": source["parameter"], "producer": producer["tool_id"],
        "source_path": source["source_path"], "selectors": selector(producer, schemas),
        "target_schema_sha256": fresh.canonical_hash(schemas[target["tool_id"]]),
        "producer_schema_sha256": fresh.canonical_hash(schemas[producer["tool_id"]]),
        **({k: source[k] for k in ("constant", "contract_tool", "contract_quote")}
           if source["kind"] == "state_difference" else {})}


def copy_compatible(producer, target, parameter, path, value, schemas):
    """Mechanical schema gate; compatible descriptions do not prove semantics."""
    source = response_property(schemas[producer["tool_id"]], path)
    dest = schemas[target["tool_id"]].get("parameters", {}).get("properties", {}).get(parameter, {})
    numeric = {"number", "float", "integer", "int"}
    source_type, target_type = source.get("type"), dest.get("type")
    compatible = (source_type == target_type == "string" and isinstance(value, str)
                  or source_type in numeric and target_type in numeric and isinstance(value, (int, float))
                  and not isinstance(value, bool) and math.isfinite(value))
    return (compatible and not source.get("enum") and not dest.get("enum")
            and not fresh._extract_enum_options(source) and not fresh._extract_enum_options(dest))



def matching_returns(target, parameter, available, schemas):
    return [(p, path, value) for p in available
            for path, value in fresh.scalar_fields(fresh.parse_output(p["output"]))
            if fresh.equal(value, target["args"][parameter])
            and copy_compatible(p, target, parameter, path, value, schemas)]


def compile_options(tasks, trials):
    """Finite fits: unique matching return, or adjacent capacity-minus-state."""
    lookup = {(c["episode_id"], c["call_index"]): c for t in trials for c in t["replay_payload"]["tool_calls"]}
    options = {}
    for trial in trials:
        schemas = schemas_for(tasks[trial["task_id"]])
        for target in trial["replay_payload"]["tool_calls"]:
            if fresh.target_outcome(target) != "local_no_reported_error":
                continue
            calls = trial["replay_payload"]["tool_calls"]
            available = [p for p in calls[:target["call_index"]] if p["response_message"] is not None
                         and p["response_message"] < target["assistant_message"]
                         and fresh.target_outcome(p) == "local_no_reported_error"]
            for parameter, actual in target["args"].items():
                if isinstance(actual, bool) or not isinstance(actual, (str, int, float)) or actual == "":
                    continue
                anchor = {"episode_id": trial["episode_id"], "target_call": target["call_index"], "parameter": parameter}
                matches = [({**anchor, "kind": "tool_output", "source_call": p["call_index"], "source_path": path}, value)
                           for p, path, value in matching_returns(target, parameter, available, schemas)]
                differences = []
                for producer in available:
                    for path, value in fresh.scalar_fields(fresh.parse_output(producer["output"])):
                        if producer["call_index"] + 1 != target["call_index"] or not all(
                                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (actual, value)):
                            continue
                        prop = schemas[target["tool_id"]].get("parameters", {}).get("properties", {}).get(parameter, {})
                        description = schemas[target["tool_id"]].get("description", "") + " " + prop.get("description", "")
                        for match in CAPACITY.finditer(description):
                            constant, unit = float(match[1]), match[2].lower().rstrip("s")
                            pattern = r"\b" + re.escape(unit) + r"s?\b"
                            observed_prop = response_property(schemas[producer["tool_id"]], path)
                            if (not re.search(pattern, prop.get("description", ""), re.I)
                                    or not re.search(pattern, observed_prop.get("description", ""), re.I)
                                    or not 0 <= value <= constant or not 0 <= actual <= constant
                                    or not fresh.equal(constant - value, actual)):
                                continue
                            differences.append(({**anchor, "kind": "state_difference", "source_call": producer["call_index"],
                                "source_path": path, "constant": constant, "contract_tool": target["tool_id"], "contract_quote": match[0]}, value))
                # ponytail: abstain on indistinguishable calls/fields; no general object resolver.
                for source, value in (matches if len(matches) == 1 else []) + (differences if len(differences) == 1 else []):
                    fresh.check_source(source, schemas, lookup)
                    signature = relation_signature(source, lookup, schemas)
                    identity = "rel_" + fresh.canonical_hash(signature)[:16]
                    producer = lookup[(source["episode_id"], source["source_call"])]
                    option = options.setdefault(identity, {"option_id": identity, **signature, "supports": []})
                    option["supports"].append({"task_id": trial["task_id"], "source": source, "state_value": value,
                        "producer_arguments": deepcopy(producer["args"]),
                        "producer_contract_field": deepcopy(response_property(schemas[producer["tool_id"]], source["source_path"])),
                        "target_contract_field": deepcopy(schemas[target["tool_id"]]["parameters"]["properties"][parameter]),
                        "matching_source_count": 1})
    options = {k: o for k, o in options.items() if o["kind"] != "state_difference"
               or len({s["state_value"] for s in o["supports"]}) >= 2}
    return options, lookup


def related_options(record, task_id, options, lookup, schemas):
    anchors = {(r["episode_id"], r["call_index"]) for r in record["metadata"]["evidence_refs"]}
    result = {}
    for identity, option in options.items():
        if (fresh.canonical_hash(schemas.get(option["target_tool"])) != option["target_schema_sha256"]
                or fresh.canonical_hash(schemas.get(option["producer"])) != option["producer_schema_sha256"]):
            continue
        if option["kind"] == "state_difference" and re.search(r"\bafter\b", record["when_to_use"], re.I):
            continue  # A post-operation check cannot become a pre-operation calculation.
        anchored = [s for s in option["supports"] if s["task_id"] == task_id and any(
            (s["source"]["episode_id"], s["source"][key]) in anchors for key in ("source_call", "target_call"))]
        if anchored:
            result[identity] = option
    return result


def support_examples(option):
    """Bound excerpts, keeping both numeric extremes for the difference check."""
    supports = option["supports"]
    if option["kind"] == "state_difference":
        ordered = sorted(supports, key=lambda s: (s["state_value"], s["source"]["episode_id"]))
        return [ordered[0], ordered[-1]]
    return supports[:2]


def render_instruction(options, condition):
    # Keep model wording intact: an action sentence is not an if-clause.
    context = "Context: " + condition.strip().rstrip(".") + ".\nParameter guidance: "
    if len(options) == 2 and all(o["kind"] == "tool_output" for o in options) and all(
            options[0][k] == options[1][k] for k in ("producer", "target_tool", "source_path", "selectors")):
        first = options[0]
        fields = " and ".join(o["parameter"] for o in options)
        path = ".".join(first["source_path"])
        return (context + f"obtain each missing {first['target_tool']} parameter ({fields}) via "
                f"{first['producer']} for its corresponding current object. Wait for each response and use its "
                f"{path} field for the matching parameter; preserve object correspondence, not response order.")
    sentences = []
    for option in options:
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in option["selectors"].items())
        producer = option["producer"] + (f"({args})" if args else "")
        path = ".".join(option["source_path"])
        if option["kind"] == "state_difference":
            sentences.append(f"read {producer} for the current object and wait for its response; set "
                f"{option['target_tool']}.{option['parameter']} to {json.dumps(option['constant'])} minus its {path} field")
        else:
            sentences.append(f"if {producer} is needed to obtain {option['target_tool']}.{option['parameter']} "
                f"for the intended object, reuse its matching current response or call it and wait; use its {path} field")
    return context + "; ".join(sentences) + "."


def checked_relations(chosen, record, lookup, tasks, episode_tasks, cited=None):
    """Recheck full actual edges; optional model citations belong to the legacy interface."""
    evidence, sources, constants = {}, [], []
    for option in chosen:
        if cited is not None and not any(all((s["source"]["episode_id"], s["source"][key]) in cited
                       for key in ("source_call", "target_call")) for s in option["supports"]):
            raise ValueError("selected option needs a cited complete producer-target edge")
        values = set()
        for support in option["supports"]:
            source = support["source"]
            schemas = schemas_for(tasks[episode_tasks[source["episode_id"]]])
            _, checked, permitted = fresh.check_source(source, schemas, lookup)
            if relation_signature(source, lookup, schemas) != {k: v for k, v in option.items() if k not in {"option_id", "supports"}}:
                raise ValueError("relation signature drift")
            producer = lookup[(source["episode_id"], source["source_call"])]
            target = lookup[(source["episode_id"], source["target_call"])]
            if (fresh.target_outcome(producer) != "local_no_reported_error"
                    or support["producer_arguments"] != producer["args"] or support["matching_source_count"] != 1):
                raise ValueError("producer/object evidence drift")
            value = fresh.at_path(fresh.parse_output(producer["output"]), source["source_path"])
            if source["kind"] == "tool_output":
                available = [c for (episode, _), c in lookup.items() if episode == source["episode_id"]
                    and c["call_index"] < target["call_index"] and c["response_message"] is not None
                    and c["response_message"] < target["assistant_message"]
                    and fresh.target_outcome(c) == "local_no_reported_error"]
                matches = matching_returns(target, source["parameter"], available, schemas)
                if len(matches) != 1 or matches[0][0]["call_index"] != source["source_call"] or matches[0][1] != source["source_path"]:
                    raise ValueError("response copy is ambiguous or fails scalar/type/enum check")
            if (support["producer_contract_field"] != response_property(schemas[producer["tool_id"]], source["source_path"])
                    or support["target_contract_field"] != schemas[target["tool_id"]]["parameters"]["properties"][source["parameter"]]):
                raise ValueError("contract evidence drift")
            if source["kind"] == "state_difference":
                if producer["call_index"] + 1 != target["call_index"]:
                    raise ValueError("state read is not adjacent")
                values.add(value)
            constants.extend(permitted)
            sources.append(source)
            for ref in checked:
                evidence[(ref["episode_id"], ref["call_index"])] = {**ref, "call_sha256": fresh.canonical_hash(lookup[(ref["episode_id"], ref["call_index"])])}
        if option["kind"] == "state_difference":
            if len(values) < 2:
                raise ValueError("difference needs distinct states")
            if re.search(r"\bafter\b", record["when_to_use"], re.I):
                raise ValueError("post-operation memory cannot become a prerequisite")
    return evidence, sources, constants


def check_rewrite(proposal, record, options, lookup, tasks, episode_tasks, allowed_refs):
    """Check the proposed splice; semantic review is a separate release gate."""
    fields = {"record_id", "option_ids", "replace_span", "condition", "evidence_refs"}
    if not isinstance(proposal, dict) or set(proposal) != fields or proposal["record_id"] != record["record_id"]:
        raise ValueError("proposal fields/identity mismatch")
    selected, span, condition, refs = (proposal[k] for k in ("option_ids", "replace_span", "condition", "evidence_refs"))
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 2 or any(not isinstance(i, str) for i in selected)
            or len(set(selected)) != len(selected) or any(i not in options for i in selected)):
        raise ValueError("one or two distinct in-topic options required")
    if (not isinstance(span, str) or not span.strip() or record["content"].count(span) != 1
            or span != span.strip() or not span.endswith((".", "!", "?"))):
        raise ValueError("replace_span must be a unique complete action sentence")
    start = record["content"].index(span)
    if start and record["content"][:start].rstrip()[-1:] not in ".!?":
        raise ValueError("replace_span starts inside another sentence")
    end = start + len(span)
    if end < len(record["content"]) and not record["content"][end].isspace():
        raise ValueError("replace_span ends inside a token or quoted name")
    if re.search(r"[.!?]\s+[A-Z]", span):
        raise ValueError("replace_span must not remove multiple ordinary sentences")
    if not isinstance(condition, str) or not 1 <= len(condition.strip()) <= 200 or any(c in condition for c in "\r\n") or "{{" in condition:
        raise ValueError("condition must be short, explicit and single-line")
    if not isinstance(refs, list) or not refs:
        raise ValueError("complete supplied evidence is required")
    cited = set()
    for ref in refs:
        fresh.resolve_evidence(ref, lookup)
        key = (ref["episode_id"], ref["call_index"])
        if key not in allowed_refs:
            raise ValueError("reference was not supplied to the proposer")
        cited.add(key)
    chosen = [options[i] for i in selected]
    if len({(o["target_tool"], o["parameter"]) for o in chosen}) != len(chosen):
        raise ValueError("competing expressions for one parameter cannot be combined")
    evidence, sources, constants = checked_relations(chosen, record, lookup, tasks, episode_tasks, cited)
    if any(o["kind"] == "state_difference" for o in chosen) and not re.search(r"\b(full|capacity|completely)\b", condition, re.I):
        raise ValueError("difference needs a stated full-capacity goal")
    replacement = render_instruction(chosen, condition)
    content = record["content"][:start] + replacement + record["content"][start + len(span):]
    if len(content) > 600 or content == record["content"] or "{{" in content or "}}" in content:
        raise ValueError("rendered content too long, unchanged or malformed")
    schemas = {name: schema for task in tasks.values() for name, schema in schemas_for(task).items()}
    # Existing literal screening walks arguments, not returned state values.
    # Include the selected observations so a condition cannot smuggle old fuel/ID values.
    literal_calls = list(lookup.values()) + [{"args": {"observed_value": support["state_value"],
        "producer_arguments": support["producer_arguments"]}} for option in chosen for support in option["supports"]]
    for text in (condition, content):
        fresh.screen_text(text, literal_calls, schemas, constants, include_goal_numbers=True)
    new = deepcopy(record)
    new["content"] = content
    new["record_id"] = "mem_" + fresh.canonical_hash([new["when_to_use"], content, new["tool_tags"]])[:16]
    new["metadata"].update(source_record_id=record["record_id"], source_record_sha256=fresh.canonical_hash(record),
        source_record_metadata=deepcopy(record["metadata"]), conditional_rewrite=True,
        condition_status="model_proposed_not_semantically_proven", unique_source_proven=False,
        selected_options=selected, new_binding_evidence=list(evidence.values()), new_parameter_sources=sources,
        edit_proposal=deepcopy(proposal), replacement_text=replacement, publication_status="pending_training_review")
    return new


def render_attachment(options):
    """Conservative value-availability branches, not an inferred applicability oracle."""
    if len(options) == 2 and all(o["kind"] == "tool_output" for o in options) and all(
            options[0][k] == options[1][k] for k in ("producer", "target_tool", "source_path", "selectors")):
        first = options[0]
        fields = " and ".join(o["parameter"] for o in options)
        return (f"For {first['target_tool']}.{fields}, use valid values already supplied for the respective objects. "
                f"For missing values, query {first['producer']} for each corresponding object, wait, and bind its "
                f"{'.'.join(first['source_path'])} to that object, not response order.")
    sentences = []
    for o in options:
        field = f"{o['target_tool']}.{o['parameter']}"
        path = ".".join(o["source_path"])
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in o["selectors"].items())
        producer = o["producer"] + (f"({args})" if args else "")
        if o["kind"] == "state_difference":
            sentences.append(f"Only to fill to capacity, read current {producer}, wait, then set {field} to "
                f"{json.dumps(o['constant'])} minus {path}; do not use this difference for a requested increment.")
        else:
            sentences.append(f"For {field}, use a valid value already given for the target object; otherwise bind "
                f"{o['producer']}.{path} only from its matching current response and wait if pending. "
                "Do not create a new object merely to obtain this value.")
    return " ".join(sentences)


def check_attachment(proposal, record, options, lookup, tasks, episode_tasks):
    if not isinstance(proposal, dict) or set(proposal) != {"record_id", "option_ids"} or proposal["record_id"] != record["record_id"]:
        raise ValueError("selection fields/identity mismatch")
    selected = proposal["option_ids"]
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 2
            or any(not isinstance(i, str) or i not in options for i in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("one or two distinct in-topic options required")
    chosen = [options[i] for i in selected]
    if len({(o["target_tool"], o["parameter"]) for o in chosen}) != len(chosen):
        raise ValueError("competing expressions for one parameter cannot be combined")
    for o in chosen:
        named = re.search(r"(?<!\w)" + re.escape(o["target_tool"]) + r"(?!\w)", record["when_to_use"] + " " + record["content"])
        if o["target_tool"] not in record["tool_tags"] and not named:
            if o["kind"] != "state_difference" or o["producer"] not in record["tool_tags"]:
                raise ValueError("relation target is outside the original memory operation")
    evidence, sources, constants = checked_relations(chosen, record, lookup, tasks, episode_tasks)
    rule = render_attachment(chosen)
    content = record["content"] + "\n" + rule
    if len(content) > 600 or "{{" in content or "}}" in content:
        raise ValueError("attached content too long or malformed")
    schemas = {name: schema for task in tasks.values() for name, schema in schemas_for(task).items()}
    literal_calls = list(lookup.values()) + [{"args": {"observed_value": s["state_value"],
        "producer_arguments": s["producer_arguments"]}} for o in chosen for s in o["supports"]]
    # Only code-generated identifiers are exempt: a failed old call may have
    # literally passed "booking_id" as its value. No model prose uses this list.
    identifiers = [key for o in chosen for key in [o["parameter"], *o["source_path"]]]
    fresh.screen_text(rule, literal_calls, schemas, constants + identifiers, include_goal_numbers=True)
    new = deepcopy(record)
    new["content"] = content
    new["record_id"] = "mem_" + fresh.canonical_hash([new["when_to_use"], content, new["tool_tags"]])[:16]
    new["metadata"].update(source_record_id=record["record_id"], source_record_sha256=fresh.canonical_hash(record),
        source_record_metadata=deepcopy(record["metadata"]), conditional_rewrite=True,
        condition_status="program_defined_availability_branches_not_semantic_proof", unique_source_proven=False,
        selected_options=selected, new_binding_evidence=list(evidence.values()), new_parameter_sources=sources,
        edit_proposal=deepcopy(proposal), replacement_text=rule, publication_status="pending_training_review",
        original_content_preserved=True, evidence_selection="program_all_rechecked_supports")
    return new


def make_payload(records, task_id, task, trials, options, lookup):
    schemas = schemas_for(task)
    related = {r["record_id"]: related_options(r, task_id, options, lookup, schemas) for r in records}
    offered = {k: o for opts in related.values() for k, o in opts.items()}
    topics = {lookup[(ref["episode_id"], ref["call_index"])]["tool_id"] for r in records for ref in r["metadata"]["evidence_refs"]}
    topics |= {o[k] for o in offered.values() for k in ("producer", "target_tool")}
    views, supplied = [], set()
    for trial in trials:
        if trial["task_id"] != task_id:
            continue
        calls = trial["replay_payload"]["tool_calls"]
        needed = {c["call_index"] for c in calls if c["tool_id"] in topics}
        needed |= {c["call_index"] - 1 for c in calls if c["call_index"] in needed and c["call_index"] > 0}
        projected = []
        for c in calls:
            if c["call_index"] not in needed:
                continue
            supplied.add((c["episode_id"], c["call_index"]))
            projected.append({k: c[k] for k in ("call_index", "tool_id", "args", "goal_snapshot", "assistant_message", "response_message", "execution_status")}
                             | {"output_scalar_fields": list(fresh.scalar_fields(fresh.parse_output(c["output"]))), "local_outcome": fresh.target_outcome(c)})
        views.append({"episode_id": trial["episode_id"], "base_seed": trial["base_seed"], "whole_task_reward": trial["reward"], "calls": projected})
    menu = []
    for identity, option in sorted(offered.items()):
        examples = []
        for s in support_examples(option):
            b = s["source"]
            calls = [lookup[(b["episode_id"], b[k])] for k in ("source_call", "target_call")]
            supplied.update((c["episode_id"], c["call_index"]) for c in calls)
            examples.append({"task_id": s["task_id"], "episode_id": b["episode_id"], "calls": calls})
        menu.append({**{k: v for k, v in option.items() if k not in {"supports", "target_schema_sha256", "producer_schema_sha256"}}, "support_examples": examples})
    payload = {"task_id": task_id, "original_memories": [{k: r[k] for k in ("record_id", "when_to_use", "content")} | {"allowed_option_ids": sorted(related[r["record_id"]])} for r in records],
        "tools": list(schemas.values()), "current_task_trajectories": views, "relation_options": menu,
        "view_note": "operation-local projection; scalar-only outputs in eight-trial views, full supporting calls in relation options; no causal or unique-source certification"}
    return payload, related, supplied


def write_bank(path, records, index):
    path.mkdir(mode=0o700)
    fresh.validate_frozen_bank_index(records, index)
    memories_hash = write_private_json(path / "memories.json", records, exclusive=True)
    # Do not sort the index keys: stable retrieval ties must preserve the source order.
    with os.fdopen(os.open(path / "memories_index.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {"memories.json": memories_hash, "memories_index.json": fresh.sha256_file(path / "memories_index.json")}


def prepare_inputs(source, trials_dir, output):
    if any(output.resolve().is_relative_to(p.resolve()) for p in (source, trials_dir, fresh.TRAIN.parent)):
        raise ValueError("output must be outside input trees")
    tasks, trials, paths = fresh.load_build_inputs(trials_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    freeze = json.loads((source / "freeze.json").read_text())
    if fresh.sha256_file(source / "manifest.json") != freeze["manifest_sha256"] or fresh.sha256_file(source / "verification.json") != freeze["verification_sha256"]:
        raise ValueError("source freeze mismatch")
    for name in ("memories.json", "memories_index.json"):
        if fresh.sha256_file(source / name) != manifest["outputs_sha256"][name] or manifest["outputs_sha256"] != freeze["outputs_sha256"]:
            raise ValueError("source bank identity mismatch")
    original = json.loads((source / "memories.json").read_text())
    original_index = json.loads((source / "memories_index.json").read_text())
    fresh.validate_frozen_bank_index(original, original_index)
    records = [r for r in original if r["type_tag"] == "generic_rule"]
    source_inputs = {str(Path(p).resolve()): h for p, h in manifest["inputs_sha256"].items()}
    for p in [fresh.TRAIN, fresh.VAL, *paths]:
        if source_inputs.get(str(p.resolve())) != fresh.sha256_file(p):
            raise ValueError("training evidence/split differs from source bank")
    options, lookup = compile_options(tasks, trials)
    for r in records:
        if r["record_id"] != "mem_" + fresh.canonical_hash([r["when_to_use"], r["content"], r["tool_tags"]])[:16]:
            raise ValueError("ordinary identity mismatch")
        for ref in r["metadata"]["evidence_refs"]:
            if ref["call_sha256"] != fresh.canonical_hash(lookup[(ref["episode_id"], ref["call_index"])]):
                raise ValueError("ordinary evidence join mismatch")
    pinned = [source / n for n in ("memories.json", "memories_index.json", "manifest.json", "verification.json", "freeze.json")]
    pinned += [fresh.TRAIN, fresh.VAL, Path(__file__), Path(fresh.__file__), *paths]
    pins = {str(p.resolve()): fresh.sha256_file(p) for p in pinned}
    return tasks, trials, records, original_index, options, lookup, pins, manifest


def rewrite(source, trials_dir, output, model=None, *, dry_run=False, cached_run=None):
    tasks, trials, records, original_index, options, lookup, pins, manifest = prepare_inputs(source, trials_dir, output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_private_json(output / "preparation.json", {"inputs_sha256": pins, "prompt_sha256": fresh.canonical_hash(PROMPT), "actor_model_configuration": model,
        "options": len(options), "option_kinds": dict(Counter(o["kind"] for o in options.values())), "records": len(records),
        "budget": "at most one generation per train task, <=50; no acquisition or embedding"}, exclusive=True)
    write_private_json(output / "relation_options.json", options, exclusive=True)
    related = {r["record_id"]: related_options(r, r["source_task"], options, lookup, schemas_for(tasks[r["source_task"]])) for r in records}
    audit = {"kind": "finite_repair_cpu_preparation", "records": len(records), "training_trials": len(trials),
        "options": len(options), "option_kinds": dict(Counter(o["kind"] for o in options.values())),
        "related_records": sum(bool(v) for v in related.values()),
        "related_tasks": len({r["source_task"] for r in records if related[r["record_id"]]}),
        "record_options": {k: sorted(v) for k, v in related.items()},
        "evidence_summary": {k: {"supports": len(o["supports"]),
            "distinct_observed_values": len({json.dumps(v["state_value"], sort_keys=True) for v in o["supports"]}),
            "distinct_producer_arguments": len({fresh.canonical_hash(v["producer_arguments"]) for v in o["supports"]}),
            "source_uniqueness_proven": False} for k, o in options.items()},
        "actor_admission": False, "model_requests": 0, "network_requests": 0, "embedding_requests": 0,
        "inputs_sha256": pins, "prompt_sha256": fresh.canonical_hash(PROMPT)}
    write_private_json(output / "cpu_preparation.json", audit, exclusive=True)
    if dry_run:
        return audit
    if model is None and cached_run is None:
        raise ValueError("generation requires explicit local model configuration")
    cache_pins = {}
    if cached_run is not None:
        if model is not None:
            raise ValueError("cached selection migration takes no model configuration")
        saved = json.loads((cached_run / "manifest.json").read_text())
        freeze = json.loads((cached_run / "freeze.json").read_text())
        if fresh.sha256_file(cached_run / "manifest.json") != freeze["manifest_sha256"]:
            raise ValueError("cached run manifest mismatch")
        for name, digest in saved["inputs_sha256"].items():
            if Path(name).resolve() != Path(__file__).resolve() and fresh.sha256_file(Path(name)) != digest:
                raise ValueError("cached source evidence differs from current evidence")
        if json.loads((cached_run / "relation_options.json").read_text()) != options:
            raise ValueError("cached relation menu differs")
        if json.loads((cached_run / "baseline_bank/memories.json").read_text()) != records:
            raise ValueError("cached ordinary memories differ")
        for name, digest in freeze["outputs_sha256"]["baseline_bank"].items():
            if fresh.sha256_file(cached_run / "baseline_bank" / name) != digest:
                raise ValueError("cached baseline bank mismatch")
        cache_paths = [cached_run / n for n in ("manifest.json", "freeze.json", "relation_options.json")]
        cache_paths += list((cached_run / "inputs").glob("*.json")) + list((cached_run / "responses").glob("*.json"))
        cache_pins = {str(p.resolve()): fresh.sha256_file(p) for p in cache_paths}
        write_private_json(output / "selection_migration.json", {"source_run": str(cached_run.resolve()),
            "source_prompt_sha256": saved["prompt_sha256"], "new_prompt_not_sent": True,
            "retained_model_fields": ["record_id", "option_ids"],
            "ignored_legacy_fields": ["replace_span", "condition", "evidence_refs"],
            "evidence_is_program_owned_not_repaired_model_citations": True, "inputs_sha256": cache_pins}, exclusive=True)
    if not audit["related_tasks"] or audit["related_tasks"] > 50:
        raise ValueError("generation requires 1-50 tasks with checked candidate edges")
    if cached_run is None:
        ledger = fresh.RequestLedger(output / "requests.sqlite", {"max_requests": 102, "max_total_tokens": 1800000, "wall_seconds": 14400, "max_workers": 1})
        settings = fresh.distillation_config(model)
        settings.seed = 0
        write_private_json(output / "generation_settings.json", {"model": settings.model, "temperature": 0,
            "seed": 0, "enable_thinking": False, "cache_enabled": False, "max_tokens": settings.max_tokens,
            "max_retries": settings.max_retries}, exclusive=True)
        generation_model = {**model, "name": settings.model, "base_urls": [settings.base_url],
            "enable_thinking": False, "temperature": 0, "max_tokens": settings.max_tokens,
            "max_retries": settings.max_retries, "context_budget": settings.context_budget}
        client = fresh.DeepSeekLLMClient(settings)
        generation = client._request_with_retry
        def full_headroom(payload):
            if payload["max_tokens"] < 4096:
                raise ContextBudgetError("rewrite requires full output headroom", {"effective_max_tokens": payload["max_tokens"]})
            return generation(payload)
        client._request_with_retry = full_headroom
        active = "provider_identity"
        def tokenization(payload):
            identity = ledger.reserve(active, "tokenize")
            try:
                raw = native_tokenization(payload)
                ledger.finish(identity, None, "success")
                return raw
            except Exception:
                ledger.finish(identity, None, "error")
                raise
        native_tokenization = client._request_tokenization
        client._request_tokenization = tokenization
        identity = ledger.reserve(active, "provider_metadata")
        try:
            write_private_json(output / "providers.json", fresh.inspect_providers(generation_model), exclusive=True)
            ledger.finish(identity, None, "success")
        except Exception:
            ledger.finish(identity, None, "error")
            raise
    by_task = defaultdict(list)
    for r in records:
        by_task[r["source_task"]].append(r)
    changes, decisions = {}, []
    episode_tasks = {t["episode_id"]: t["task_id"] for t in trials}
    for task_id, originals in sorted(by_task.items()):
        active = task_id
        payload, related, supplied = make_payload(originals, task_id, tasks[task_id], trials, options, lookup)
        if not any(related.values()):
            decisions.append({"task_id": task_id, "status": "no_related_option"})
            continue
        write_private_json(output / "inputs" / (task_id + ".json"), payload, exclusive=True)
        if cached_run is not None:
            cached_payload = json.loads((cached_run / "inputs" / (task_id + ".json")).read_text())
            if json.loads(json.dumps(payload)) != cached_payload:
                raise ValueError("cached selection input differs")
            response = json.loads((cached_run / "responses" / (task_id + ".json")).read_text())
            write_private_json(output / "responses" / (task_id + ".json"), response, exclusive=True)
        else:
            identity = ledger.reserve(active, "rewrite_llm")
            try:
                response = client.chat(PROMPT, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), temperature=0, schema_hash=fresh.canonical_hash(PROMPT))
                ledger.finish(identity, response["raw"].get("usage"), "success")
            except Exception as exc:
                audit = getattr(exc, "context_budget_audit", None)
                ledger.finish(identity, None, "not_sent_context_budget" if audit and audit.get("effective_max_tokens", 4096) < 4096 else "error")
                if audit and audit.get("effective_max_tokens", 4096) < 4096:
                    decisions.append({"task_id": task_id, "status": "context_budget_fallback", "audit": audit})
                    continue
                write_private_json(output / "failure.json", {"phase": "rewrite_api", "task_id": task_id,
                    "error_type": type(exc).__name__, "context_budget": audit, "resources": ledger.summary()}, exclusive=True)
                raise RuntimeError("rewrite stopped; no automatic retry") from None
            write_private_json(output / "responses" / (task_id + ".json"), response, exclusive=True)
        try:
            if any(c.get("finish_reason") == "length" for c in response["raw"].get("choices", [])):
                raise ValueError("truncated proposal")
            proposed = json.loads(response["content"])
            if not isinstance(proposed, dict) or set(proposed) != {"memories"} or not isinstance(proposed["memories"], list) or len(proposed["memories"]) > len(originals):
                raise ValueError("proposal envelope mismatch")
            known = {r["record_id"]: r for r in originals}
            ids = [c.get("record_id") for c in proposed["memories"] if isinstance(c, dict)]
            if len(ids) != len(proposed["memories"]) or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids) or not set(ids) <= set(known):
                raise ValueError("unknown or duplicate proposal record")
            for candidate in proposed["memories"]:
                record_id = candidate["record_id"]
                try:
                    raw_candidate = deepcopy(candidate)
                    if cached_run is not None:
                        if set(candidate) not in ({"record_id", "option_ids"}, {"record_id", "option_ids", "replace_span", "condition", "evidence_refs"}):
                            raise ValueError("unknown cached selection fields")
                        candidate = {k: candidate[k] for k in ("record_id", "option_ids")}
                    changes[record_id] = check_attachment(candidate, known[record_id], related[record_id], lookup, tasks, episode_tasks)
                    if cached_run is not None:
                        changes[record_id]["metadata"]["legacy_proposal_private_not_injected"] = raw_candidate
                    decisions.append({"task_id": task_id, "record_id": record_id, "status": "rewritten"})
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    decisions.append({"task_id": task_id, "record_id": record_id, "status": "original_fallback", "reason": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            decisions.append({"task_id": task_id, "status": "proposal_parse_fallback", "reason": str(exc)})
        write_private_json(output / "decisions.json", decisions)
        print(json.dumps({"task_id": task_id, "rewritten_so_far": len(changes)}, ensure_ascii=False), flush=True)
    rewritten = [changes.get(r["record_id"], deepcopy(r)) for r in records]
    mapping = {old["record_id"]: new["record_id"] for old, new in zip(records, rewritten)}
    if len(set(mapping.values())) != len(records):
        raise ValueError("rewritten identities collide; no implicit merge")
    baseline_index = {k: v for k, v in original_index.items() if k in mapping}
    rewritten_index = {mapping[k]: deepcopy(v) for k, v in baseline_index.items()}
    if any(fresh.sha256_file(Path(p)) != h for p, h in {**pins, **cache_pins}.items()):
        raise ValueError("rewrite input drift")
    outputs = {"baseline_bank": write_bank(output / "baseline_bank", records, baseline_index),
               "proposed_bank": write_bank(output / "proposed_bank", rewritten, rewritten_index)}
    review = [{"source_record_id": r["record_id"], "candidate_record_id": changes[r["record_id"]]["record_id"],
        "original_content": r["content"], "proposed_content": changes[r["record_id"]]["content"],
        "condition_policy": changes[r["record_id"]]["metadata"]["condition_status"],
        "selected_options": changes[r["record_id"]]["metadata"]["selected_options"],
        "evidence_refs": changes[r["record_id"]]["metadata"]["new_binding_evidence"],
        "character_difference": len(changes[r["record_id"]]["content"]) - len(r["content"]),
        "decision": "pending", "reason": "", "information_gain": ""} for r in records if r["record_id"] in changes]
    write_private_json(output / "training_review.json", {"status": "pending", "candidates": review}, exclusive=True)
    write_private_json(output / "decisions.json", decisions)
    summary = {"kind": "finite_condition_repair_proposals", "training_trials": len(trials), "records_per_arm": len(records),
        "rewritten_records": len(changes), "fallback_or_unmodified_records": len(records) - len(changes), "old_to_new_ids": mapping,
        "inputs_sha256": pins, "outputs_sha256": outputs, "resources": ledger.summary() if cached_run is None else {"requests": 0, "new_model_requests": 0,
            "prior_model_cost_is_not_zero": True, "source_cost_manifest": str(cached_run / "manifest.json")},
        "prompt_sha256": fresh.canonical_hash(PROMPT),
        "actor_model_configuration": model,
        "generation_settings_sha256": fresh.sha256_file(output / "generation_settings.json") if cached_run is None else None,
        "selection_origin": "new_prompt" if cached_run is None else "historical_option_selection_migration",
        "cached_inputs_sha256": cache_pins, "actor_admission": False, "publication_status": "pending_training_review", "embedding_requests": 0,
        "no_new_acquisition": True, "conditions_semantically_verified": False,
        "natural_retrieval_unchanged": True, "ordinary_source_manual_exclusions": manifest["manual_train_review_exclusions"]}
    digest = write_private_json(output / "manifest.json", summary, exclusive=True)
    write_private_json(output / "freeze.json", {"manifest_sha256": digest, "outputs_sha256": outputs,
        "decisions_sha256": fresh.sha256_file(output / "decisions.json"), "no_online_updates": True}, exclusive=True)
    return summary



def offline_rewrite(source, trials, output, cached_run=None):
    import socket
    from unittest.mock import patch
    def offline(*args, **kwargs):
        raise AssertionError("CPU preparation forbids network requests")
    with patch.object(socket.socket, "connect", offline), patch.object(socket.socket, "connect_ex", offline), patch.object(socket.socket, "sendto", offline):
        return rewrite(source, trials, output, dry_run=cached_run is None, cached_run=cached_run)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source-bank", "trials", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--cached-selection-run", type=Path, help="CPU-only migration of saved option selections; ignores legacy free text")
    parser.add_argument("--dry-run", action="store_true", help="CPU evidence preparation only; socket access refused")
    args = parser.parse_args()
    if args.dry_run or args.cached_selection_run:
        if args.dry_run and args.cached_selection_run:
            parser.error("choose preparation or cached selection migration")
        if args.model_config:
            parser.error("CPU preparation takes no model config")
        summary = offline_rewrite(args.source_bank, args.trials, args.output, args.cached_selection_run)
    else:
        if not args.model_config:
            parser.error("generation requires --model-config; historical replay uses its saved source snapshot")
        summary = rewrite(args.source_bank, args.trials, args.output, json.loads(args.model_config.read_text())["model"])
    print(json.dumps({k: v for k, v in summary.items() if k not in {"inputs_sha256", "old_to_new_ids", "record_options", "evidence_summary"}}, ensure_ascii=False, indent=2))
