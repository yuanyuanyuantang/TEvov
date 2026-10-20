#!/usr/bin/env python3
"""Publish a drop-only, train-reviewed subset of a generated bank, without requests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_memory_from_scratch as fresh
from tevo.memory.evidence import write_private_json


def select_records(records, tasks, groups, reviewed):
    """Apply literal checks to ordinary prose; keep compiled records unchanged."""
    kept, dropped = [], []
    for record in records:
        identity = record["record_id"]
        if identity in reviewed:
            dropped.append({"record_id": identity, "reason": "train_review_exclusion",
                            "review": reviewed[identity]})
            continue
        if record["type_tag"] == "generic_rule":
            calls = [c for a in record["metadata"]["candidate_assessments"]
                     for t in groups[a["task_id"]]["trajectories"] for c in t["calls"]]
            schemas = {tool["function"]["name"]: tool["function"]
                       for a in record["metadata"]["candidate_assessments"]
                       for tool in tasks[a["task_id"]]["tools"]}
            try:
                fresh.screen_text(record["when_to_use"], calls, schemas, [], include_goal_numbers=True)
                fresh.screen_text(record["content"], calls, schemas, [], include_goal_numbers=True)
            except ValueError as exc:
                dropped.append({"record_id": identity, "reason": str(exc)})
                continue
        kept.append(record)
    return kept, dropped


def clean_bank(source: Path, trials_dir: Path, review: Path, output: Path) -> dict:
    for protected in [source, trials_dir, fresh.TRAIN.parent]:
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("output must be outside the input trees")
    tasks, trials, trial_paths = fresh.load_build_inputs(trials_dir)
    groups = {g["task_id"]: g for g in fresh.group_trials(trials, tasks)}
    manifest = json.loads((source / "manifest.json").read_text())
    record_path, index_path = source / "memories.json", source / "memories_index.json"
    hashes = {p.name: fresh.sha256_file(p) for p in [record_path, index_path]}
    if hashes != manifest["outputs_sha256"]:
        raise ValueError("source bank output hashes differ from manifest")
    records, index = json.loads(record_path.read_text()), json.loads(index_path.read_text())
    fresh.validate_frozen_bank_index(records, index)
    source_inputs = {str(Path(p).resolve()): digest for p, digest in manifest["inputs_sha256"].items()}
    for path in [fresh.TRAIN, fresh.VAL, *trial_paths]:
        if source_inputs.get(str(path.resolve())) != fresh.sha256_file(path):
            raise ValueError("source bank uses different training evidence/split")
    audit = json.loads(review.read_text())
    if audit["bank_sha256"] != hashes:
        raise ValueError("review belongs to a different bank")
    reviewed = {r["record_id"]: r for r in audit["findings"]}
    if len(reviewed) != len(audit["findings"]) or not set(reviewed) <= {r["record_id"] for r in records}:
        raise ValueError("unknown or duplicate reviewed record")
    pinned = [record_path, index_path, source / "manifest.json", source / "distillation_responses.json",
              review, fresh.TRAIN, fresh.VAL, Path(__file__), Path(fresh.__file__), *trial_paths]
    pins = {str(p.resolve()): fresh.sha256_file(p) for p in pinned}
    calls = {(c["episode_id"], c["call_index"]): c for t in trials for c in t["replay_payload"]["tool_calls"]}
    episode_tasks = {t["episode_id"]: t["task_id"] for t in trials}
    for row in records:
        if row["record_id"] != "mem_" + fresh.canonical_hash([row["when_to_use"], row["content"], row["tool_tags"]])[:16]:
            raise ValueError("record identity mismatch")
        for ref in row["metadata"]["evidence_refs"]:
            call = calls[(ref["episode_id"], ref["call_index"])]
            if fresh.canonical_hash(call) != ref["call_sha256"]:
                raise ValueError("evidence join mismatch")
        for episode, bindings in row["metadata"]["parameter_sources_by_episode"].items():
            schemas = {t["function"]["name"]: t["function"] for t in tasks[episode_tasks[episode]]["tools"]}
            for binding in bindings:
                text, _, _ = fresh.check_source(binding, schemas, calls)
                if text not in row["content"]:
                    raise ValueError("parameter renderer/evidence mismatch")
    kept, dropped = select_records(records, tasks, groups, reviewed)
    if not kept or not any(r["type_tag"] == "generic_rule" for r in kept):
        raise ValueError("cleanup removed every ordinary memory")
    subset = {r["record_id"]: index[r["record_id"]] for r in kept}
    fresh.validate_frozen_bank_index(kept, subset)
    if any(fresh.sha256_file(Path(p)) != digest for p, digest in pins.items()):
        raise ValueError("cleanup input drift")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    outputs = {"memories.json": write_private_json(output / "memories.json", kept, exclusive=True),
               "memories_index.json": write_private_json(output / "memories_index.json", subset, exclusive=True)}
    result = {"kind": "train_reviewed_drop_only_bank", "source_bank": str(source.resolve()),
              "source_records": len(records), "records": len(kept), "removed": dropped,
              "memory_type_counts": {tag: sum(r["type_tag"] == tag for r in kept)
                                     for tag in ("generic_rule", "parameter_template")},
              "ordinary_task_coverage": len({a["task_id"] for r in kept if r["type_tag"] == "generic_rule"
                                             for a in r["metadata"]["candidate_assessments"]}),
              "policy": "nested argument literals + whole-goal numeric matches; explicit train-review exclusions; drop only",
              "manual_train_review_exclusions": sorted(reviewed), "contents_and_vectors_unchanged": True,
              "general_temporal_checker": False, "full_semantic_correctness_verified": False,
              "model_requests": 0, "embedding_requests": 0, "network_requests": 0,
              "source_generation_resources": manifest["resources"], "inputs_sha256": pins,
              "outputs_sha256": outputs, "source_prompt_sha256": manifest["prompt_sha256"]}
    manifest_hash = write_private_json(output / "manifest.json", result, exclusive=True)
    verification = {"passed": True, "check_scope": "input hashes, record/reference identity, parameter fits, exact subset, index",
                    "not_a_semantic_certification": True, "manifest_sha256": manifest_hash,
                    "model_requests": 0, "network_requests": 0}
    verify_hash = write_private_json(output / "verification.json", verification, exclusive=True)
    write_private_json(output / "freeze.json", {"kind": "frozen_train_reviewed_bank",
        "outputs_sha256": outputs, "manifest_sha256": manifest_hash, "verification_sha256": verify_hash,
        "no_online_updates": True, "frozen_scope": "fixed artifacts after bounded train-only cleanup; not perfect semantic validation"}, exclusive=True)
    return result


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # A programming error must not turn CPU cleanup into a model request.
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError("cleanup is CPU-only")
    socket.socket.connect = socket.socket.connect_ex = no_network
    result = clean_bank(args.source_bank, args.trials, args.review, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, ensure_ascii=False, indent=2))
