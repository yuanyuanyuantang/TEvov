"""Task-level paired analysis with strict experiment completeness checks."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def load_trials(paths: list[Path], *, task_ids: set[str], trials_per_task: int) -> dict[str, list[dict]]:
    """Validate historical or new trials; never silently drop missing tasks."""
    if not task_ids or trials_per_task <= 0:
        raise ValueError("task set and trial count must be nonempty")
    by_task: dict[str, list[dict]] = {task_id: [] for task_id in sorted(task_ids)}
    keys: set[tuple] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row.get("task_id")
                if task_id not in by_task:
                    raise ValueError(f"{path}:{line_number}: unexpected task {task_id}")
                reward = row.get("reward")
                if isinstance(reward, bool) or reward not in (0, 1):
                    raise ValueError(f"{path}:{line_number}: BFCL reward must be binary")
                if row.get("error") and reward != 0:
                    raise ValueError("error trial must be retained as failure")
                # Historical chunks reuse run_id=0; the experiment name is the
                # declared chunk identity. Source filenames cannot turn a copied
                # duplicate into a new trial.
                trial_id = row.get("trial_id", row.get("run_id"))
                if trial_id is None or not row.get("experiment_name"):
                    raise ValueError(f"{path}:{line_number}: missing trial provenance")
                key = (task_id, row["experiment_name"], trial_id)
                if key in keys:
                    raise ValueError(f"duplicate trial: {key}")
                keys.add(key)
                by_task[task_id].append(row)
    incomplete = {t: len(rows) for t, rows in by_task.items() if len(rows) != trials_per_task}
    if incomplete:
        raise ValueError(f"incomplete trial matrix; expected {trials_per_task}: {incomplete}")
    return by_task


def bootstrap_mean_ci(values: list[float], *, seed: int = 0, samples: int = 10000) -> list[float]:
    if not values or samples < 100:
        raise ValueError("bootstrap requires observations and at least 100 samples")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return [means[int(.025 * samples)], means[min(int(.975 * samples), samples - 1)]]


def sign_test(values: list[float]) -> dict[str, Any]:
    wins, losses = sum(v > 1e-9 for v in values), sum(v < -1e-9 for v in values)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return {"wins": wins, "losses": losses, "ties": len(values) - n, "p_value": p,
            "null": "equal probability of a task-level win and loss; not zero mean difference"}


def summarize(by_task: dict[str, list[dict]], *, seed: int = 0) -> dict[str, Any]:
    counts = {len(rows) for rows in by_task.values()}
    if not by_task or len(counts) != 1 or 0 in counts:
        raise ValueError("summary requires a complete equal-trial matrix")
    task_means = [statistics.mean(r["reward"] for r in rows) for rows in by_task.values()]
    task_best = [max(r["reward"] for r in rows) for rows in by_task.values()]
    rows = [r for group in by_task.values() for r in group]
    return {"tasks": len(by_task), "trials": len(rows), "trials_per_task": counts.pop(),
            "successes": sum(r["reward"] for r in rows), "trial_mean": statistics.mean(task_means),
            "trial_mean_task_bootstrap_ci95": bootstrap_mean_ci(task_means, seed=seed),
            "best_of_observed": statistics.mean(task_best),
            "best_of_observed_task_bootstrap_ci95": bootstrap_mean_ci(task_best, seed=seed),
            "error_trials": sum(bool(r.get("error")) for r in rows),
            "sampling_recorded_trials": sum(bool(r.get("sampling")) for r in rows),
            "cost_recorded_trials": sum(bool(r.get("request_metrics")) for r in rows)}


def paired_comparison(left: dict[str, list[dict]], right: dict[str, list[dict]], *, seed: int = 0) -> dict[str, Any]:
    if not left or set(left) != set(right):
        raise ValueError("paired comparison requires identical nonempty task sets")
    if any(len(left[t]) != len(right[t]) or not left[t] for t in left):
        raise ValueError("paired comparison requires equal trials per task")
    result = {}
    for label, reducer in (("trial_mean", statistics.mean), ("best_of_observed", max)):
        diffs = [reducer(r["reward"] for r in left[t]) - reducer(r["reward"] for r in right[t])
                 for t in sorted(left)]
        result[label] = {"difference": statistics.mean(diffs),
                         "task_bootstrap_ci95": bootstrap_mean_ci(diffs, seed=seed),
                         "sign_test": sign_test(diffs)}
    result["unit"] = "task; repeated trials are not independent samples"
    result["interpretation"] = "a confidence interval crossing zero does not establish equivalence"
    return result
