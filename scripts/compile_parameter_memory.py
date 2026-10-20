#!/usr/bin/env python3
"""Build a train-only G2/V0 bank without model, embedding or environment calls."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tevo.memory.evidence import verify_join, write_private_json
from tevo.experiment import offline_relations as relations
from tevo.experiment.suite import validate_frozen_bank_index
from tevo.memory.parameter_compiler import compile_records
from tevo.reproducibility import canonical_hash, sha256_file


def build_bank(train_path: Path, trials_dir: Path, source_dir: Path, output_dir: Path) -> dict:
    inputs = [train_path, source_dir / "memories.json", source_dir / "memories_index.json"]
    files = sorted(trials_dir.glob("*.json"))
    if not files:
        raise ValueError("no training trajectories")
    for protected in [train_path.parent, trials_dir, source_dir]:
        if output_dir.resolve().is_relative_to(protected.resolve()):
            raise ValueError("output must be outside the input trees")
    tasks = {}
    for row in map(json.loads, train_path.read_text().splitlines()):
        if row["id"] in tasks:
            raise ValueError("duplicate training task")
        tasks[row["id"]] = row
    pins = {str(path.resolve()): sha256_file(path) for path in inputs + files}
    trials, episodes, identities = [], {}, set()
    for path in files:
        trial = json.loads(path.read_text())
        episode, payload = trial["episode_id"], trial["replay_payload"]
        if isinstance(trial.get("base_seed"), bool) or not isinstance(trial.get("base_seed"), int):
            raise ValueError("training base_seed must be an integer")
        identity = (trial["task_id"], trial["base_seed"])
        if trial["task_id"] not in tasks or not episode or episode in episodes or identity in identities:
            raise ValueError("unknown task or duplicate training identity")
        if payload.get("episode_id") != episode or payload.get("task_id") != trial["task_id"]:
            raise ValueError("replay identity mismatch")
        if canonical_hash(payload) != trial.get("replay_sha256"):
            raise ValueError("training replay hash mismatch")
        calls = payload["tool_calls"]
        if not isinstance(calls, list):
            raise ValueError("training calls must be a list")
        identities.add(identity)
        episodes[episode] = {"calls": calls, "task_id": trial["task_id"]}
        trials.append((path.name, trial))
    records = json.loads(inputs[1].read_text())
    index_raw = inputs[2].read_bytes()
    validate_frozen_bank_index(records, json.loads(index_raw))
    for record in records:
        if "parameter_compilation" in record.get("metadata", {}):
            raise ValueError("source must contain uncompiled records")
        refs = record["metadata"].get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("source evidence is missing")
        valid, problems, _ = verify_join(refs, episodes, set())
        if not valid:
            raise ValueError(f"source evidence mismatch: {record['record_id']}: {problems}")
    units, excluded = relations.build_units(tasks, trials)
    compiled, report = compile_records(records, units, episodes)
    for record in compiled:
        if "parameter_compilation" in record["metadata"]:
            valid, problems, _ = verify_join(record["metadata"]["rendered_evidence_refs"], episodes, set())
            if not valid:
                raise ValueError(f"compiled evidence mismatch: {problems}")
    validate_frozen_bank_index(compiled, json.loads(index_raw))
    if any(sha256_file(path) != sha for path, sha in pins.items()):
        raise ValueError("input changed during compilation")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    outputs = {"memories.json": write_private_json(output_dir / "memories.json", compiled)}
    with os.fdopen(os.open(output_dir / "memories_index.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(index_raw)
    outputs["memories_index.json"] = sha256_file(output_dir / "memories_index.json")
    manifest = {
        "kind": "g2_parameter_source_bank", "policy": "G2/V0; fixed first training-fit candidate",
        "training_only": True, "development_scores_used_for_candidate_selection": False,
        "proposer_choice": "restricted G2 enumeration toolkit; no independent efficacy claim",
        "source_bank": str(source_dir.resolve()), "inputs_sha256": pins,
        "code_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/compile_parameter_memory.py", "tevo/memory/evidence.py",
            "tevo/memory/parameter_compiler.py", "tevo/experiment/offline_relations.py")},
        "report": report, "excluded_before_generation": excluded,
        "outputs_sha256": outputs, "index_bytes_preserved": True,
        "cost": {"model_requests": 0, "embedding_requests": 0, "environment_attempts": 0},
        "actor_efficacy_evaluated": False,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_private_json(output_dir / "compilation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "benchmark/bfcl/data/multiturn_data_base_train.jsonl")
    parser.add_argument("--trials", type=Path, required=True, help="Explicit execution evidence for the new source bank")
    parser.add_argument("--source-bank", type=Path, required=True, help="Explicit new uncompiled bank; no historical default")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; refuse overwrite")
    args = parser.parse_args()
    os.umask(0o077)
    manifest = build_bank(args.train, args.trials, args.source_bank, args.output)
    print(json.dumps({key: value for key, value in manifest["report"].items() if key != "decisions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
