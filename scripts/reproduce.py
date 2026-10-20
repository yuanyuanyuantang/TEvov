#!/usr/bin/env python3
"""One command for pinned BFCL bootstrap, provenance, tests and optional suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Offline Gorilla Git source for bootstrap")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Experiment suite JSON")
    parser.add_argument("--run", action="store_true", help="Run model experiments after checks")
    args = parser.parse_args()
    if args.run and args.config is None:
        parser.error("--run requires --config")
    output = args.output_dir or ROOT / "outputs" / "reproduction" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    commands = [[sys.executable, "scripts/bootstrap_bfcl.py"]]
    if args.source_dir:
        commands[0].extend(["--source-dir", str(args.source_dir.resolve())])
    commands.extend([
        [sys.executable, "scripts/capture_environment.py", "--output", str(output / "environment.json"),
         "--artifact", "third_party/bfcl.lock.json", "--artifact", "requirements-reproduce.lock.txt",
         "--artifact", "benchmark/bfcl/data/multiturn_data_base_train.jsonl",
         "--artifact", "benchmark/bfcl/data/multiturn_data_base_val.jsonl"],
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        [sys.executable, "-m", "pip", "check"],
    ])
    if args.config:
        commands.append([sys.executable, "scripts/run_experiment_suite.py", "--config", str(args.config.resolve()),
                         "--run" if args.run else "--dry-run"])
    results = []
    for i, command in enumerate(commands):
        log = output / f"step_{i + 1}.log"
        print(f"Step {i + 1}/{len(commands)}: {Path(command[1]).name}; log={log}", flush=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        results.append({"step": i + 1, "command": command, "exit_code": result.returncode, "log": log.name})
        (output / "checks.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if result.returncode:
            print(f"Check failed; see {log}", file=sys.stderr)
            return result.returncode
    print(f"Completed {len(results)} checks; evidence={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
