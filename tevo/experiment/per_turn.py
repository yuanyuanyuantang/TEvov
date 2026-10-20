"""Per-user-turn retrieval helpers (pure CPU; no I/O, no requests).

The query rule is frozen by the doc-05 diagnostic on the frozen v6 bank: for a
task with user turns U_0..U_{T-1} (verbatim user-role contents of the frozen
converted val dataset, in order; every entry's role is asserted to be
``user``), the query at turn t is

    Q_t = "\\n".join(U_0..U_t)

i.e. exactly the user-visible goal and history at turn t. Tool outputs,
injected memory and reference calls never enter the query.  These helpers are
the single implementation used by the experiment suite (per-turn context
preparation and cache validation) and its tests; the diagnostic script
``scripts/diagnose_per_turn_recall.py`` keeps its own frozen copy because its
bytes are pinned in doc 05 and must not change.
"""

from __future__ import annotations

import ast


def parse_user_content(entry) -> str:
    """BFCL val question entries are dict-repr strings; accept dicts/strings too."""
    if isinstance(entry, dict):
        role, content = entry.get("role"), entry.get("content")
    elif isinstance(entry, str):
        parsed = None
        if entry.strip().startswith("{"):
            try:
                parsed = ast.literal_eval(entry)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, dict):
            role, content = parsed.get("role"), parsed.get("content")
        else:
            role, content = "user", entry
    else:
        raise ValueError(f"unsupported user turn entry type: {type(entry).__name__}")
    if role != "user":
        raise ValueError(f"question turn is not a user message: role={role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("user message content must be a non-empty string")
    return content


def extract_user_turns(question_turns) -> list[str]:
    """User turns U_0..U_{T-1} from an extra.question list."""
    turns = []
    for item in question_turns or []:
        entry = item[0] if isinstance(item, (list, tuple)) and item else item
        turns.append(parse_user_content(entry))
    if not turns:
        raise ValueError("task exposes no user turns")
    return turns


def cumulative_queries(turns: list[str]) -> list[str]:
    """Q_t = newline-joined U_0..U_t (index t corresponds to turn t)."""
    queries, parts = [], []
    for turn in turns:
        parts.append(turn)
        queries.append("\n".join(parts))
    return queries


def partition_turn_queries(turns_by_task: dict, task_ids: list[str]) -> tuple[list[tuple], list[tuple]]:
    """(task, turn_index, query) items split into first-turn and later-turn batches.

    Batch 1 keeps the pilot composition (all first turns in one batch) so the
    stored query vectors reproduce byte-for-byte (doc 05 §2); batch 2 carries
    the later-turn queries, which have no stored reference.
    """
    first, later = [], []
    for task in task_ids:
        for turn_index, query in enumerate(cumulative_queries(turns_by_task[task])):
            (first if turn_index == 0 else later).append((task, turn_index, query))
    return first, later


def validate_turn_identity(first_message_content: str, turns: list[str], task: str) -> None:
    """The dataset exposes Q_0 twice; both copies must be the same verbatim text."""
    if turns[0] != first_message_content:
        raise ValueError(
            f"{task}: extra.question turn 0 differs from messages[0] content; "
            "refusing an ambiguous first-turn query"
        )