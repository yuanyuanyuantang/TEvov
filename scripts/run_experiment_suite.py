#!/usr/bin/env python3
"""Validate or execute an auditable frozen-bank BFCL experiment matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tevo.experiment.suite import load_plan, resolve_path, run_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="+", help="Explicit experiment-name subset")
    parser.add_argument("--output-dir", help="New isolated output directory")
    parser.add_argument("--run", action="store_true", help="Contact configured services and run the selected matrix")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without network calls (default)")
    parser.add_argument("--resume", action="store_true", help="Require unchanged provenance for completed trials")
    args = parser.parse_args()
    if args.run and args.dry_run:
        parser.error("--run and --dry-run are mutually exclusive")
    plan = load_plan(args.config, args.only)
    if not args.run:
        print(json.dumps({"mode": "offline_validation", "tasks": len(plan["task_ids"]),
                          "seeds": plan["config"]["seeds"], "expected_trials": len(plan["jobs"]),
                          "sampling_seed_mode": plan["config"].get("sampling_seed_mode", "task_derived"),
                          "repeats_per_seed": plan["config"].get("repeats_per_seed", 1),
                          "experiments": [arm["name"] for arm in plan["experiments"]],
                          "split": plan["split"], "bank_paths": plan["bank_paths"],
                          "budgets": plan["config"]["budgets"], "network_requests": 0}, ensure_ascii=False, indent=2))
        return 0
    output = resolve_path(args.output_dir or plan["config"]["output_dir"])
    summary = run_suite(plan, output, args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["paper_result_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
