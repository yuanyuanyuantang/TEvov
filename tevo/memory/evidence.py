"""Shared evidence joins and private JSON output for parameter compilation.

Extracted unchanged from the retired migration tool; no evolution/registry imports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tevo.reproducibility import canonical_hash, sha256_file

def write_private_json(path: Path, value: object, *, exclusive: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if exclusive:
        # O_EXCL: refuse to replace an existing file (fail closed).
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
    else:
        path.write_text(data, encoding="utf-8")
    os.chmod(path, 0o600)
    return sha256_file(path)


def call_goal(call: dict) -> str:
    goal = call.get("goal_snapshot")
    state = call.get("state") if isinstance(call.get("state"), dict) else {}
    if not isinstance(goal, str) or not goal.strip():
        goal = state.get("user_goal", call.get("user_goal", ""))
    return goal if isinstance(goal, str) else ""


def verify_join(evidence_refs: list, episodes: dict[str, dict], conflicts: set[str]) -> tuple[bool, list[str], list[str]]:
    """Verify the record's own evidence anchors against the original trials."""
    problems: list[str] = []
    needed: list[str] = []
    for ref in evidence_refs:
        if not isinstance(ref, dict):
            problems.append("invalid_ref")
            continue
        episode, index = ref.get("episode_id"), ref.get("call_index")
        if not isinstance(episode, str) or isinstance(index, bool) or not isinstance(index, int):
            problems.append("invalid_ref")
            continue
        if episode in conflicts:
            problems.append(f"duplicate_episode_conflict:{episode}")
            continue
        entry = episodes.get(episode)
        if entry is None:
            problems.append(f"unknown_episode:{episode}")
            continue
        calls = entry["calls"]
        if index < 0 or index >= len(calls) or not isinstance(calls[index], dict):
            problems.append(f"call_index_out_of_range:{episode}:{index}")
            continue
        call = calls[index]
        if episode not in needed:
            needed.append(episode)
        checks = {
            "tool_id": str(call.get("tool_id", call.get("name", ""))) == str(ref.get("tool_id")),
            "arguments_sha256": canonical_hash(call.get("args", call.get("arguments", {}))) == ref.get("arguments_sha256"),
            "output_sha256": canonical_hash(call.get("output", call.get("result", call.get("error", "")))) == ref.get("output_sha256"),
            "success": isinstance(call.get("success"), bool) and call.get("success") == ref.get("success"),
            "execution_status": str(call.get("execution_status")) == str(ref.get("execution_status")),
            "turn_id": str(call.get("turn_id")) == str(ref.get("turn_id")),
            "goal_sha256": canonical_hash(call_goal(call)) == ref.get("goal_sha256"),
        }
        for name, ok in checks.items():
            if not ok:
                problems.append(f"mismatch:{name}:{episode}:{index}")
    return (not problems), problems, needed

