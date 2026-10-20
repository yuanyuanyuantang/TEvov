"""TEvo 状态 schema — s_t = {g, h_t, o_t, c_t, e_t, b_t}。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


def build_constraint_snapshot(
    available_params: dict[str, Any],
    missing_params: list[str],
) -> dict[str, Any]:
    satisfied = sorted(str(key) for key in available_params.keys())
    unsatisfied = sorted({str(param) for param in missing_params if str(param) not in available_params})
    return {
        "satisfied_params": satisfied,
        "unsatisfied_params": unsatisfied,
        "n_satisfied": len(satisfied),
        "n_unsatisfied": len(unsatisfied),
    }


def build_error_snapshot(
    latest_error: str | None,
    *,
    current_phase: str = "main",
    failure_history: list[str] | None = None,
    last_failed_tool: str = "",
) -> dict[str, Any]:
    failures = list(failure_history or [])
    latest = (latest_error or "").strip()
    return {
        "has_error": bool(latest),
        "latest_error": latest,
        "recovery_phase": current_phase == "recover" or bool(latest),
        "consecutive_failures": len(failures),
        "last_failed_tool": last_failed_tool,
    }


def build_budget_snapshot(
    budget: float,
    *,
    tracked: bool = False,
    floor: float = 1.0,
) -> dict[str, Any]:
    remaining = float(budget)
    effective_tracked = bool(tracked)
    return {
        "tracked": effective_tracked,
        "remaining_budget": remaining,
        "budget_floor": float(floor),
        "low_budget": effective_tracked and remaining <= float(floor),
        "exhausted": effective_tracked and remaining <= 0.0,
    }


def build_state_dump(
    *,
    task_id: str,
    user_goal: str,
    current_phase: str,
    executed_tools: list[str],
    observations: list[str],
    constraints: dict[str, Any],
    error: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    latest_observation = str(observations[-1]) if observations else ""
    return {
        "task_id": task_id,
        "goal": user_goal,
        "phase": current_phase,
        "history": {
            "executed_tools": list(executed_tools),
            "n_executed_tools": len(executed_tools),
        },
        "observations": {
            "latest_observation": latest_observation,
            "n_observations": len(observations),
            "latest_bindings": {},
        },
        "constraints": constraints,
        "error": error,
        "budget": budget,
    }


def summarize_state_dump(state_dump: dict[str, Any]) -> str:
    history = state_dump.get("history", {}) if isinstance(state_dump.get("history"), dict) else {}
    observations = state_dump.get("observations", {}) if isinstance(state_dump.get("observations"), dict) else {}
    constraints = state_dump.get("constraints", {}) if isinstance(state_dump.get("constraints"), dict) else {}
    error = state_dump.get("error", {}) if isinstance(state_dump.get("error"), dict) else {}
    budget = state_dump.get("budget", {}) if isinstance(state_dump.get("budget"), dict) else {}
    executed_tools = history.get("executed_tools", [])
    if not isinstance(executed_tools, list):
        executed_tools = []
    satisfied = constraints.get("satisfied_params", [])
    if not isinstance(satisfied, list):
        satisfied = []
    unsatisfied = constraints.get("unsatisfied_params", [])
    if not isinstance(unsatisfied, list):
        unsatisfied = []
    return (
        f"task_id={state_dump.get('task_id', '')} "
        f"goal={state_dump.get('goal', '')} "
        f"phase={state_dump.get('phase', 'main')} "
        f"executed_tools={' -> '.join(str(item) for item in executed_tools)} "
        f"latest_error={error.get('latest_error') or 'none'} "
        f"recovery_phase={error.get('recovery_phase', False)} "
        f"budget_remaining={budget.get('remaining_budget', 0.0)} "
        f"budget_low={budget.get('low_budget', False)} "
        f"satisfied_params={','.join(str(item) for item in satisfied[:5])} "
        f"unsatisfied_params={','.join(str(item) for item in unsatisfied)} "
        f"observation={observations.get('latest_observation', '')}"
    )


def coerce_state_dump(state_payload: dict[str, Any]) -> dict[str, Any]:
    """从 TEvoState.to_dict() 或较旧的 state payload 中统一恢复结构化 state_dump。"""
    if isinstance(state_payload.get("state_dump"), dict):
        return dict(state_payload["state_dump"])

    observations = state_payload.get("observations", [])
    if not isinstance(observations, list):
        observations = []
    executed_tools = state_payload.get("executed_tools", [])
    if not isinstance(executed_tools, list):
        executed_tools = []
    available_params = state_payload.get("available_params", {})
    if not isinstance(available_params, dict):
        available_params = {}
    missing_params = state_payload.get("missing_params", [])
    if not isinstance(missing_params, list):
        missing_params = []
    failure_history = state_payload.get("failure_history", [])
    if not isinstance(failure_history, list):
        failure_history = []

    constraints = state_payload.get("constraint_snapshot") or state_payload.get("constraints")
    if not isinstance(constraints, dict):
        constraints = build_constraint_snapshot(available_params, missing_params)

    error = state_payload.get("error_snapshot") or state_payload.get("error_context")
    if not isinstance(error, dict):
        error = build_error_snapshot(
            state_payload.get("latest_error"),
            current_phase=str(state_payload.get("current_phase", "main")),
            failure_history=failure_history,
            last_failed_tool=str(state_payload.get("last_failed_tool", "")),
        )

    budget = state_payload.get("budget_snapshot") or state_payload.get("budget_context")
    if not isinstance(budget, dict):
        raw_budget = state_payload.get("budget", 0.0)
        try:
            budget_value = float(raw_budget or 0.0)
        except (TypeError, ValueError):
            budget_value = 0.0
        budget = build_budget_snapshot(budget_value, tracked=False)

    return build_state_dump(
        task_id=str(state_payload.get("task_id", "")),
        user_goal=str(state_payload.get("user_goal", "")),
        current_phase=str(state_payload.get("current_phase", "main")),
        executed_tools=[str(item) for item in executed_tools],
        observations=[str(item) for item in observations],
        constraints=constraints,
        error=error,
        budget=budget,
    )


def coerce_state_summary(state_payload: dict[str, Any]) -> str:
    """优先从 state_dump 派生统一摘要；兼容较旧 payload。"""
    existing = state_payload.get("state_summary")
    if existing and isinstance(state_payload.get("state_dump"), dict):
        return str(existing)
    return summarize_state_dump(coerce_state_dump(state_payload))


def summarize_memory_state_pattern(state_dump: dict[str, Any]) -> str:
    """为记忆学习/检索生成更稳定、可跨任务复用的状态键。

    只保留跨任务稳定的泛化特征（phase, intent, domain, recovery, error），
    去掉 step 级别的组合爆炸字段（recent_tools, satisfied/unsatisfied params,
    last_failed_tool, budget），这些会导致同一个任务的不同 step 产生完全不同的 pattern，
    使记忆无法跨任务复用。
    """
    if not state_dump:
        return ""
    goal = str(state_dump.get("goal", ""))
    history = state_dump.get("history", {}) if isinstance(state_dump.get("history"), dict) else {}
    recent_tools = history.get("executed_tools", [])
    if not isinstance(recent_tools, list):
        recent_tools = []
    error = state_dump.get("error", {}) if isinstance(state_dump.get("error"), dict) else {}
    parts = [
        f"phase={str(state_dump.get('phase', 'main'))}",
        "intent=" + _infer_goal_intent(goal),
        "domain=" + _infer_goal_domain(goal, [str(tool) for tool in recent_tools]),
        f"recovery_phase={str(bool(error.get('recovery_phase', False))).lower()}",
        "error_family=" + _infer_error_family(str(error.get("latest_error", ""))),
        f"has_error={str(bool(error.get('has_error', False))).lower()}",
    ]
    return " | ".join(parts)


_COMMAND_MARKERS = {
    "cat",
    "cd",
    "cp",
    "diff",
    "du",
    "find",
    "grep",
    "ls",
    "mkdir",
    "mv",
    "pwd",
    "rm",
    "rmdir",
    "sort",
    "tail",
    "touch",
    "wc",
}


def _goal_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9_]+", text.lower()) if token}


def _marker_in_text(marker: str, *, text: str, tokens: set[str]) -> bool:
    marker = marker.lower().strip()
    if not marker:
        return False
    if any(ch.isspace() for ch in marker):
        return marker in text
    if marker in _COMMAND_MARKERS or len(marker) <= 3:
        return marker in tokens
    pattern = r"\b" + re.escape(marker) + r"(?:s|es|d|ed|ing)?\b"
    return re.search(pattern, text) is not None


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    tokens = _goal_tokens(text)
    return any(_marker_in_text(marker, text=text, tokens=tokens) for marker in markers)


def _infer_goal_intent(goal: str) -> str:
    text = goal.lower()
    intent_markers = [
        ("move", ("move", "relocate", "rename")),
        ("copy", ("copy", "duplicate")),
        ("create", ("create", "make", "mkdir", "touch", "add", "write")),
        ("delete", ("delete", "remove", "rm", "cancel")),
        ("search", ("search", "lookup", "find")),
        ("read", ("read", "view", "show", "display", "cat", "list")),
        ("compare", ("compare", "diff")),
        ("book", ("book", "reserve")),
        ("send", ("send", "message", "post", "tweet", "retweet")),
        ("trade", ("stock", "trade", "order", "watchlist")),
        (
            "calculate",
            (
                "calculate",
                "compute",
                "convert",
                "conversion",
                "estimate",
                "determine",
                "determining",
                "how many",
                "how much",
                "liter",
                "liters",
                "litre",
                "litres",
                "gallon",
                "gallons",
                "distance",
                "zipcode",
                "exchange",
            ),
        ),
        ("vehicle", ("engine", "fuel", "gasoline", "tank", "ignition", "door", "brake", "cruise", "tire", "car", "vehicle")),
    ]
    for label, markers in intent_markers:
        if _contains_any_marker(text, markers):
            return label
    tokens = [token for token in re.split(r"[^a-z0-9]+", text) if len(token) > 2]
    return tokens[0] if tokens else "general"


def _infer_goal_domain(goal: str, recent_tool_ids: list[str]) -> str:
    text = " ".join([goal.lower(), " ".join(recent_tool_ids).lower()])
    domain_markers = [
        ("filesystem", ("file", "directory", "folder", "document", "pdf", "txt", "csv", "mkdir", "touch", "cat", "grep", "mv", "cp", "ls")),
        ("travel", ("flight", "booking", "airport", "travel", "hotel", "card", "budget")),
        ("finance", ("stock", "trade", "order", "watchlist", "market", "portfolio", "account")),
        ("vehicle", ("engine", "fuel", "gasoline", "tank", "ignition", "door", "brake", "cruise", "tire", "car", "vehicle")),
        ("social", ("tweet", "message", "contact", "user", "follow", "post")),
        ("math", ("calculate", "compute", "convert", "conversion", "absolute", "distance", "zipcode", "exchange", "liter", "liters", "litre", "litres", "gallon", "gallons", "mile", "miles", "kilometer", "kilometers")),
        ("support", ("ticket", "invoice", "customer", "support")),
    ]
    for label, markers in domain_markers:
        if _contains_any_marker(text, markers):
            return label
    return "general"


def _infer_error_family(error_text: str) -> str:
    text = error_text.lower()
    if not text:
        return "none"
    families = [
        ("not_found", ("not found", "no such", "missing", "booking not found")),
        ("invalid_argument", ("invalid", "unexpected keyword", "argument", "required", "format", "type")),
        ("already_exists", ("exists", "already")),
        ("permission", ("permission", "access denied", "unauthorized", "token expired")),
        ("insufficient_resource", ("insufficient", "balance", "capacity", "above the tank")),
        ("path_error", ("path", "directory", "folder")),
        ("execution_error", ("error", "failed", "cannot")),
    ]
    for label, markers in families:
        if any(marker in text for marker in markers):
            return label
    return "other"


def coerce_memory_state_pattern(state_payload: dict[str, Any]) -> str:
    existing = state_payload.get("memory_state_pattern")
    if existing and isinstance(state_payload.get("state_dump"), dict):
        return str(existing)
    return summarize_memory_state_pattern(coerce_state_dump(state_payload))


@dataclass(slots=True)
class TEvoState:
    """
    s_t = {g, h_t, o_t, c_t, e_t, b_t}

    g             — user_goal
    h_t           — executed_tools (工具调用历史)
    o_t           — observations (中间输出)
    c_t           — constraints (satisfied/unsatisfied 参数约束快照)
    e_t           — latest_error (错误信号)
    b_t           — budget (剩余预算)
    """

    task_id: str
    user_goal: str  # g

    executed_tools: list[str] = field(default_factory=list)  # h_t
    observations: list[str] = field(default_factory=list)    # o_t
    available_params: dict[str, Any] = field(default_factory=dict)
    missing_params: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)  # c_t

    latest_error: str | None = None  # e_t
    failure_history: list[str] = field(default_factory=list)
    error_context: dict[str, Any] = field(default_factory=dict)  # e_t

    budget: float = 0.0  # b_t
    budget_context: dict[str, Any] = field(default_factory=dict)  # b_t

    current_phase: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        user_goal: str,
        executed_tools: list[str] | None = None,
        observations: list[str] | None = None,
        available_params: dict[str, Any] | None = None,
        missing_params: list[str] | None = None,
        latest_error: str | None = None,
        failure_history: list[str] | None = None,
        budget: float = 0.0,
        current_phase: str = "main",
        metadata: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        error_context: dict[str, Any] | None = None,
        budget_context: dict[str, Any] | None = None,
        last_failed_tool: str = "",
        budget_tracked: bool | None = None,
        budget_floor: float = 1.0,
    ) -> "TEvoState":
        normalized_executed = list(executed_tools or [])
        normalized_observations = list(observations or [])
        normalized_params = dict(available_params or {})
        normalized_missing = list(missing_params or [])
        normalized_failures = list(failure_history or [])
        normalized_metadata = dict(metadata or {})

        snapshot_constraints = constraints or build_constraint_snapshot(
            normalized_params,
            normalized_missing,
        )
        snapshot_error = error_context or build_error_snapshot(
            latest_error,
            current_phase=current_phase,
            failure_history=normalized_failures,
            last_failed_tool=last_failed_tool,
        )
        tracked = budget_tracked if budget_tracked is not None else budget > 0
        snapshot_budget = budget_context or build_budget_snapshot(
            budget,
            tracked=tracked,
            floor=budget_floor,
        )

        return cls(
            task_id=task_id,
            user_goal=user_goal,
            executed_tools=normalized_executed,
            observations=normalized_observations,
            available_params=normalized_params,
            missing_params=normalized_missing,
            constraints=snapshot_constraints,
            latest_error=latest_error,
            failure_history=normalized_failures,
            error_context=snapshot_error,
            budget=budget,
            budget_context=snapshot_budget,
            current_phase=current_phase,
            metadata=normalized_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraints"] = self.constraint_snapshot
        payload["constraint_snapshot"] = self.constraint_snapshot
        payload["error_context"] = self.error_snapshot
        payload["error_snapshot"] = self.error_snapshot
        payload["budget_context"] = self.budget_snapshot
        payload["budget_snapshot"] = self.budget_snapshot
        payload["state_dump"] = self.state_dump
        # Keep state_summary in serialized payloads so downstream evolution code
        # receives the same state signal regardless of whether it gets a TEvoState
        # instance or a plain dict snapshot.
        payload["state_summary"] = self.state_summary
        payload["memory_state_pattern"] = self.memory_state_pattern
        return payload

    @property
    def constraint_snapshot(self) -> dict[str, Any]:
        if self.constraints:
            satisfied = sorted(str(item) for item in self.constraints.get("satisfied_params", []))
            unsatisfied = sorted(str(item) for item in self.constraints.get("unsatisfied_params", []))
            return {
                "satisfied_params": satisfied,
                "unsatisfied_params": unsatisfied,
                "n_satisfied": len(satisfied),
                "n_unsatisfied": len(unsatisfied),
            }
        return build_constraint_snapshot(self.available_params, self.missing_params)

    @property
    def error_snapshot(self) -> dict[str, Any]:
        if self.error_context:
            latest = str(self.error_context.get("latest_error", self.latest_error or "")).strip()
            last_failed_tool = str(self.error_context.get("last_failed_tool", ""))
            consecutive_failures = int(self.error_context.get("consecutive_failures", len(self.failure_history or [])) or 0)
            recovery_phase = bool(self.error_context.get("recovery_phase", self.current_phase == "recover" or bool(latest)))
            return {
                "has_error": bool(latest),
                "latest_error": latest,
                "recovery_phase": recovery_phase,
                "consecutive_failures": consecutive_failures,
                "last_failed_tool": last_failed_tool,
            }
        return build_error_snapshot(
            self.latest_error,
            current_phase=self.current_phase,
            failure_history=self.failure_history,
        )

    @property
    def budget_snapshot(self) -> dict[str, Any]:
        if self.budget_context:
            tracked = bool(self.budget_context.get("tracked", False))
            floor = float(self.budget_context.get("budget_floor", 1.0) or 1.0)
            remaining = float(self.budget_context.get("remaining_budget", self.budget) or 0.0)
            return {
                "tracked": tracked,
                "remaining_budget": remaining,
                "budget_floor": floor,
                "low_budget": tracked and remaining <= floor,
                "exhausted": tracked and remaining <= 0.0,
            }
        return build_budget_snapshot(self.budget, tracked=False)

    @property
    def state_dump(self) -> dict[str, Any]:
        state_dump = build_state_dump(
            task_id=self.task_id,
            user_goal=self.user_goal,
            current_phase=self.current_phase,
            executed_tools=self.executed_tools,
            observations=self.observations,
            constraints=self.constraint_snapshot,
            error=self.error_snapshot,
            budget=self.budget_snapshot,
        )
        state_dump["observations"]["latest_bindings"] = dict(self.recent_output_bindings)
        state_dump["history"]["recent_steps"] = list(self.recent_steps)
        return state_dump

    @property
    def state_summary(self) -> str:
        return summarize_state_dump(self.state_dump)

    @property
    def memory_state_pattern(self) -> str:
        return summarize_memory_state_pattern(self.state_dump)

    @property
    def recent_output_bindings(self) -> dict[str, Any]:
        bindings = self.metadata.get("recent_output_bindings", {})
        return dict(bindings) if isinstance(bindings, dict) else {}

    @property
    def recent_output_payload(self) -> Any:
        return self.metadata.get("recent_output_payload")

    @property
    def recent_steps(self) -> list[dict[str, Any]]:
        steps = self.metadata.get("recent_steps", [])
        return list(steps) if isinstance(steps, list) else []
