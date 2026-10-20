#!/usr/bin/env python3
"""Recheck saved full trial histories with pinned BFCL, without model requests."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tevo.experiment.suite import atomic_json, collect_results, fingerprints, load_plan
from tevo.reproducibility import canonical_hash


def regrade(plan: dict, suite_dir: Path, *, fingerprint_builder=fingerprints) -> dict:
    manifest = json.loads((suite_dir / "manifest.json").read_text(encoding="utf-8"))
    before = fingerprint_builder(plan)
    if manifest["fingerprints"] != before:
        raise ValueError("Regrading requires unchanged code/config/data/checker/bank identity")
    if manifest["manifest_sha256"] != canonical_hash(before):
        raise ValueError("Regrading source manifest hash does not match its fingerprints")
    rows, pending = collect_results(plan, suite_dir, manifest["manifest_sha256"])
    sys.path[:0] = [str(ROOT / "benchmark/bfcl"),
                   str(ROOT / "benchmark/bfcl/gorilla/berkeley-function-call-leaderboard")]
    from bfcl_agent import LocalBFCLAgent
    checks = []
    blocked_network_attempts = 0
    def refuse_network(*args, **kwargs):
        nonlocal blocked_network_attempts
        blocked_network_attempts += 1
        raise RuntimeError("Regrading must remain offline")
    model = plan["config"]["model"]
    # The BFCL multi-turn-base scorer executes local simulated environments.
    # Guard the whole loop so an accidental network path fails visibly.
    with ExitStack() as guards:
        guards.enter_context(patch.dict(os.environ, {"TEVO_INJECT_TOOL_PROFILES": "0"}))
        for method in ("connect", "connect_ex", "sendto"):
            guards.enter_context(patch("socket.socket." + method, side_effect=refuse_network))
        for row in rows:
            history = row.get("task_history")
            if not isinstance(history, list) or not history:
                checks.append({"key": row["key"], "task_id": row["task_id"],
                               "status": "missing_full_history", "matches": False})
                continue
            agent = LocalBFCLAgent(index=0, task_ids=[row["task_id"]],
                                   experiment_name="offline_regrade", data_path=plan["dataset_path"],
                                   answer_path=Path(plan["answer_path"]), model_name=model["name"],
                                   model_path=model.get("path", model["name"]), num_trials=1,
                                   tool_scope=row.get("tool_scope", "task"))
            try:
                agent.history[0][0] = history
                agent.request_metrics = row.get("request_metrics", [])
                agent._active_episode_id = row.get("episode_id", "offline")
                score_stdout = io.StringIO()
                with redirect_stdout(score_stdout):
                    reward = agent.get_reward(0, 0)
                error = agent._evaluation_error
                printed = score_stdout.getvalue().encode("utf-8")
                checks.append({"key": row["key"], "task_id": row["task_id"],
                               "experiment": row["experiment"], "stored_reward": row["reward"],
                               "regraded_reward": reward, "evaluation_error": error,
                               "scorer_stdout_sha256": hashlib.sha256(printed).hexdigest(),
                               "scorer_stdout_bytes": len(printed),
                               "matches": error is None and reward == row["reward"]})
            finally:
                try:
                    agent.llm_client.close()
                finally:
                    agent._release_run_state(0, 0)
    unchanged = fingerprint_builder(plan) == before
    return {"schema_version": 1, "model_requests": 0, "network_requests": 0,
            "source_manifest_sha256": manifest["manifest_sha256"],
            "network_guard": "socket connect/connect_ex/sendto refused",
            "blocked_network_attempts": blocked_network_attempts,
            "checked_trials": len(checks), "expected_trials": len(plan["jobs"]),
            "pending_trials": len(pending), "matches": sum(c["matches"] for c in checks),
            "provenance_unchanged": unchanged, "checks_sha256": canonical_hash(checks),
            "all_checked_match": bool(checks) and all(c["matches"] for c in checks) and unchanged,
            "complete_matrix_regraded": not pending and bool(checks) and all(c["matches"] for c in checks) and unchanged,
            "checks": checks,
            "scope": "Saved model outputs regraded with the same pinned checker; not independent model replication."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new --output path")
    report = regrade(load_plan(args.config), args.suite_dir)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))
    return 0 if report["all_checked_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
