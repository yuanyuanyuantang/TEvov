#!/usr/bin/env python3
"""CPU-only correction of positive parameter evidence; ordinary prose is immutable."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import build_memory_from_scratch as fresh
from tevo.memory.evidence import write_private_json


def correct_records(records, index, tasks, trials):
    """Filter individual supports, retaining complete prior metadata separately."""
    fresh.validate_frozen_bank_index(records, index)
    calls = {(c["episode_id"], c["call_index"]): c
             for t in trials for c in t["replay_payload"]["tool_calls"]}
    episode_tasks = {t["episode_id"]: t["task_id"] for t in trials}
    kept, mapping, audit = [], {}, []
    for row in records:
        old_id = row["record_id"]
        if old_id != "mem_" + fresh.canonical_hash([row["when_to_use"], row["content"], row["tool_tags"]])[:16]:
            raise ValueError("source record identity mismatch")
        anchored = set()
        for ref in row["metadata"]["evidence_refs"]:
            key = (ref["episode_id"], ref["call_index"])
            if key not in calls or fresh.canonical_hash(calls[key]) != ref["call_sha256"]:
                raise ValueError("source evidence join mismatch")
            anchored.add(key)
        if row["type_tag"] == "generic_rule":
            kept.append(deepcopy(row))
            mapping[old_id] = old_id
            continue
        if row["type_tag"] != "parameter_template":
            raise ValueError("unsupported source record type")
        accepted, rejected, texts, refs = {}, [], set(), {}
        for episode, bindings in row["metadata"]["parameter_sources_by_episode"].items():
            schemas = {t["function"]["name"]: t["function"] for t in tasks[episode_tasks[episode]]["tools"]}
            for binding in bindings:
                if binding["episode_id"] != episode or binding["kind"] not in {"user_input", "tool_output"}:
                    raise ValueError("binding identity/scope mismatch")
                keys = [(episode, binding["target_call"])]
                if binding["kind"] == "tool_output":
                    keys.append((episode, binding["source_call"]))
                if not set(keys) <= anchored:
                    raise ValueError("binding has unanchored evidence")
                outcome = fresh.target_outcome(calls[keys[0]])
                try:
                    text, evidence, _ = fresh.check_source(binding, schemas, calls)
                except ValueError as exc:
                    # Only the declared admission change may reject old supports.
                    if str(exc) != f"target not positive support: {outcome}" or outcome == "local_no_reported_error":
                        raise
                    rejected.append({"binding": binding, "outcome": outcome})
                    continue
                texts.add(text)
                accepted.setdefault(episode, []).append(binding)
                for ref in evidence:
                    key = (ref["episode_id"], ref["call_index"])
                    refs[key] = {**ref, "call_sha256": fresh.canonical_hash(calls[key])}
        event = {"source_record_id": old_id, "accepted_supports": sum(map(len, accepted.values())),
                 "rejected_supports": rejected, "record_id": None}
        if accepted:
            if len(texts) != 1:
                raise ValueError("one source identity rendered multiple rules")
            new = deepcopy(row)
            new["content"] = texts.pop()
            new_id = "mem_" + fresh.canonical_hash([new["when_to_use"], new["content"], new["tool_tags"]])[:16]
            new["record_id"] = new_id
            counts = Counter(episode_tasks[e] for e, bs in accepted.items() for _ in bs)
            new["source_task"] = sorted(counts)[0]
            new["metadata"].update(source_record_id=old_id,
                source_record_sha256=fresh.canonical_hash(row), source_record_metadata=deepcopy(row["metadata"]),
                parameter_sources_by_episode=deepcopy(accepted), evidence_refs=list(refs.values()),
                candidate_assessments=[{"task_id": t, "accepted_supports": n,
                    "analysis_kind_basis": "trace fit and local no reported error; not task success"}
                    for t, n in sorted(counts.items())],
                positive_support_count=event["accepted_supports"],
                check_level="trace_fit_and_target_no_reported_error",
                applicability_is_instruction_not_runtime_validation=True)
            kept.append(new)
            mapping[old_id] = new_id
            event["record_id"] = new_id
        audit.append(event)
    if len({r["record_id"] for r in kept}) != len(kept):
        raise ValueError("corrected identities collide; no implicit merge")
    # Stable score ties follow original index insertion order, including remapped IDs.
    subset = {mapping[k]: deepcopy(v) for k, v in index.items() if k in mapping}
    fresh.validate_frozen_bank_index(kept, subset)
    return kept, subset, audit, mapping


def correct_bank(source: Path, trials_dir: Path, output: Path):
    for protected in (source, trials_dir, fresh.TRAIN.parent):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("output must be outside input trees")
    tasks, trials, trial_paths = fresh.load_build_inputs(trials_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    freeze = json.loads((source / "freeze.json").read_text())
    if fresh.sha256_file(source / "manifest.json") != freeze["manifest_sha256"] or fresh.sha256_file(source / "verification.json") != freeze["verification_sha256"]:
        raise ValueError("source freeze mismatch")
    hashes = {n: fresh.sha256_file(source / n) for n in ("memories.json", "memories_index.json")}
    if hashes != manifest["outputs_sha256"] or hashes != freeze["outputs_sha256"]:
        raise ValueError("source outputs differ from manifest/freeze")
    source_inputs = {str(Path(p).resolve()): h for p, h in manifest["inputs_sha256"].items()}
    for p in (fresh.TRAIN, fresh.VAL, *trial_paths):
        if source_inputs.get(str(p.resolve())) != fresh.sha256_file(p):
            raise ValueError("source uses different training evidence/split")
    paths = [source / n for n in ("memories.json", "memories_index.json", "manifest.json", "freeze.json", "verification.json")]
    paths += [fresh.TRAIN, fresh.VAL, Path(__file__), Path(fresh.__file__), *trial_paths]
    pins = {str(p.resolve()): fresh.sha256_file(p) for p in paths}
    old, index = [json.loads((source / n).read_text()) for n in ("memories.json", "memories_index.json")]
    kept, subset, audit, mapping = correct_records(old, index, tasks, trials)
    if [r for r in old if r["type_tag"] == "generic_rule"] != [r for r in kept if r["type_tag"] == "generic_rule"]:
        raise ValueError("ordinary memory drift")
    if any(fresh.sha256_file(Path(p)) != h for p, h in pins.items()):
        raise ValueError("input drift during correction")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    outputs = {"memories.json": write_private_json(output / "memories.json", kept, exclusive=True)}
    # The shared JSON writer sorts keys; the index needs its original stable tie order.
    with os.fdopen(os.open(output / "memories_index.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
        f.write("\n")
    outputs["memories_index.json"] = fresh.sha256_file(output / "memories_index.json")
    audit_hash = write_private_json(output / "support_audit.json", {"records": audit, "old_to_new_ids": mapping}, exclusive=True)
    rejected = Counter(b["outcome"] for r in audit for b in r["rejected_supports"])
    result = {"kind": "positive_parameter_evidence_correction", "training_trials": len(trials),
        "source_records": len(old), "records": len(kept), "dropped_parameter_records": sum(r["record_id"] is None for r in audit),
        "memory_type_counts": dict(Counter(r["type_tag"] for r in kept)),
        "accepted_supports": sum(r["accepted_supports"] for r in audit), "rejected_support_outcomes": dict(rejected),
        "ordinary_records_unchanged": True, "search_text_vectors_and_tie_order_unchanged_for_retained_records": True,
        "source_manual_train_review_exclusions": manifest["manual_train_review_exclusions"],
        "source_generation_resources": manifest["source_generation_resources"],
        "prompt_sha256": fresh.canonical_hash(fresh.PROMPT), "inputs_sha256": pins,
        "outputs_sha256": outputs, "support_audit_sha256": audit_hash,
        "model_requests": 0, "embedding_requests": 0, "network_requests": 0,
        "full_semantic_correctness_verified": False, "condition_text_is_not_an_applicability_classifier": True,
        "policy": "Only structured local returns without explicit error support positive rules; unknown abstains. All source metadata retained privately; ordinary failure lessons unchanged."}
    manifest_hash = write_private_json(output / "manifest.json", result, exclusive=True)
    verification = {"passed": True, "manifest_sha256": manifest_hash, "not_a_semantic_certification": True,
        "ordinary_unchanged": True, "index_unchanged_except_ids_and_dropped_records": True, "network_requests": 0}
    verify_hash = write_private_json(output / "verification.json", verification, exclusive=True)
    write_private_json(output / "freeze.json", {"kind": "frozen_parameter_correction_bank", "outputs_sha256": outputs,
        "manifest_sha256": manifest_hash, "verification_sha256": verify_hash, "support_audit_sha256": audit_hash,
        "no_online_updates": True}, exclusive=True)
    return result


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source-bank", "trials", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError("parameter-bank correction is CPU-only")
    socket.socket.connect = socket.socket.connect_ex = no_network
    result = correct_bank(args.source_bank, args.trials, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"inputs_sha256", "source_generation_resources"}}, ensure_ascii=False, indent=2))
