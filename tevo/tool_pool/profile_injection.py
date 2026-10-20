"""工具画像注入 — 把演化后的 ToolSpec 字段渲染进 LLM 可见的工具 description。

与 Procedural Memory 的注入（服务端渲染整段 markdown，贴到首条 user message）不同，
工具画像必须贴在每个工具自己的 description 上，受 OpenAI tool schema 结构约束，
因此渲染放在 agent 侧调用这里的纯函数：不依赖服务/LLM，便于单测与字段级消融。

对应 proposal C2（双库协同演化）：Tool Profile Pool 的演化字段首次进入 LLM 的因果路径。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .tool_schema import ToolSpec

# 可注入的演化字段（desc_u 是静态字段，永不注入；统计字段只用于过滤/分析）
ALL_PROFILE_FIELDS: tuple[str, ...] = (
    "when_to_use",
    "preconditions",
    "failure_cues",
    "recovery_hints",
    "expected_outcomes",
    "input_requirements",
)

# 默认注入字段：input_requirements 的基线内容（"参数名: required (type)"）只是把同一个
# tool schema 里已可见的 parameters 再说一遍，贴进 description 是重复 token；演化新增的
# 条目（"required (inferred from successful executions)"）另有价值，需要时用
# TEVO_INJECT_TOOL_PROFILE_FIELDS=input_requirements,... 显式打开。
#
# expected_outcomes 同样默认不注入：2026-09-29 的内容审计（outputs/audits/profile_content_*）
# 显示它在 70/81 个工具里出现、其中 44 个工具只有它一个字段，渲染出来是
# "Expected: Returns object fields: result:integer" 这类返回值 schema 复述；被判 generic
# 的 54 个工具里 51 个含该字段，而有用的信息集中在 failure_cues / when_to_use /
# recovery_hints。需要时用 TEVO_INJECT_TOOL_PROFILE_FIELDS 显式加回。
_EXCLUDED_BY_DEFAULT: tuple[str, ...] = ("input_requirements", "expected_outcomes")
DEFAULT_INJECT_FIELDS: tuple[str, ...] = tuple(
    field for field in ALL_PROFILE_FIELDS if field not in _EXCLUDED_BY_DEFAULT
)

DEFAULT_MAX_CHARS = 400  # 每个工具注入段落的字符上限（防止 129 工具池把 prompt 撑爆）
_LIST_ITEM_LIMIT = 3  # 每个列表字段最多渲染几条
_LIST_ITEM_CHARS = 90  # 单条目的字符上限

_LABELS = {
    "when_to_use": "When to use",
    "preconditions": "Preconditions",
    "failure_cues": "Watch out",
    "recovery_hints": "If it fails",
    "expected_outcomes": "Expected",
    "input_requirements": "Inputs",
}

_PREFIX = "[Evolved]"


def _clip(text: Any, limit: int) -> str:
    """压平空白并截断到 limit 字符（超出以 … 结尾）。"""
    flat = " ".join(str(text or "").split())
    if limit > 0 and len(flat) > limit:
        return flat[: max(limit - 1, 1)].rstrip() + "…"
    return flat


def _render_recovery_hint(hint: Any) -> str:
    """RecoveryHint（dataclass 或 dict）→ 'failure_mode→alternative_tool'。"""
    if isinstance(hint, dict):
        failure_mode = hint.get("failure_mode")
        alternative = hint.get("alternative_tool")
        status = hint.get("validation_status", "legacy_cooccurrence")
    else:
        failure_mode = getattr(hint, "failure_mode", None)
        alternative = getattr(hint, "alternative_tool", None)
        status = getattr(hint, "validation_status", "legacy_cooccurrence")
    failure_mode = _clip(failure_mode, _LIST_ITEM_CHARS)
    alternative = _clip(alternative, _LIST_ITEM_CHARS)
    if failure_mode and alternative:
        text = f"{failure_mode}→{alternative}"
        return f"[unverified candidate] {text}" if status == "candidate" else text
    return failure_mode or alternative


def _render_field(field_name: str, value: Any, item_chars: int) -> str:
    """把单个演化字段渲染成 'Label: 内容'；空值返回空串。"""
    label = _LABELS.get(field_name, field_name)
    if value is None:
        return ""
    if isinstance(value, str):
        text = _clip(value, item_chars * 2)
    elif isinstance(value, dict):
        items = [
            f"{_clip(k, 40)}: {_clip(v, item_chars)}"
            for k, v in value.items()
            if _clip(v, item_chars)
        ][:_LIST_ITEM_LIMIT]
        text = "; ".join(items)
    elif isinstance(value, Iterable):
        rendered: list[str] = []
        for item in value:
            if field_name == "recovery_hints":
                item_text = _render_recovery_hint(item)
            else:
                item_text = _clip(item, item_chars)
            if item_text:
                rendered.append(item_text)
            if len(rendered) >= _LIST_ITEM_LIMIT:
                break
        text = "; ".join(rendered)
    else:
        text = _clip(value, item_chars)
    return f"{label}: {text}" if text else ""


def _profile_value(profile: dict[str, Any], name: str) -> Any:
    value = profile.get(name)
    if name != "expected_outcomes":
        return value
    if not isinstance(value, list):
        return []
    # Registry records supply the immutable interface needed to authenticate
    # contract quotes. Partial legacy endpoint payloads may still render their
    # ordinary prose, but cannot authenticate a reserved contract quote.
    identity_valid = all(isinstance(profile.get(key), str) and bool(profile[key])
                         for key in ("tool_id", "tool_name"))
    contract_valid = (identity_valid and isinstance(profile.get("description"), str)
                      and isinstance(profile.get("parameter_schema"), dict)
                      and isinstance(profile.get("metadata"), dict))
    return ToolSpec(
        tool_id=profile.get("tool_id", "") if identity_valid else "",
        tool_name=profile.get("tool_name", "") if identity_valid else "",
        description=profile.get("description", "") if contract_valid else "",
        parameter_schema=profile.get("parameter_schema", {}) if contract_valid else {},
        metadata=profile["metadata"] if contract_valid else {},
        expected_outcomes=value,
    ).safe_expected_outcomes()


def profile_has_content(
    profile: dict[str, Any] | None,
    fields: Iterable[str] = ALL_PROFILE_FIELDS,
) -> bool:
    """画像在指定字段上是否有可注入内容。"""
    if not isinstance(profile, dict):
        return False
    return any(_render_field(name, _profile_value(profile, name), _LIST_ITEM_CHARS) for name in fields)


def format_profile_notes(
    profile: dict[str, Any] | None,
    *,
    fields: Iterable[str] = ALL_PROFILE_FIELDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    item_chars: int = _LIST_ITEM_CHARS,
) -> str:
    """把一个工具的演化画像渲染成可贴进 description 的紧凑段落。

    字段顺序固定（when_to_use → preconditions → failure_cues → recovery_hints →
    expected_outcomes → input_requirements），总长受 max_chars 限制。
    无内容时返回空串（调用方据此跳过该工具）。
    """
    if not isinstance(profile, dict) or max_chars <= 0:
        return ""
    parts = [
        rendered
        for rendered in (
            _render_field(name, _profile_value(profile, name), item_chars) for name in fields
        )
        if rendered
    ]
    if not parts:
        return ""
    return _clip(f"{_PREFIX} " + " ".join(parts), max_chars)


def _tool_name(tool_entry: Any) -> str:
    """取工具名，兼容 {"function": {"name": ...}} 与扁平 {"name": ...}。"""
    if not isinstance(tool_entry, dict):
        return ""
    function = tool_entry.get("function")
    if isinstance(function, dict):
        return str(function.get("name", "") or "").strip()
    return str(tool_entry.get("name", "") or "").strip()


def apply_tool_profiles(
    tool_schema: list[dict[str, Any]] | None,
    profiles: dict[str, dict[str, Any]] | None,
    *,
    fields: Iterable[str] = ALL_PROFILE_FIELDS,
    max_chars: int = DEFAULT_MAX_CHARS,
    item_chars: int = _LIST_ITEM_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    """把画像段落追加到可见工具 schema 的 description 上。

    返回 (新 schema, 实际注入的工具数)。输入 schema 不被修改（深拷贝），
    因此调用方（如 all-scope 的 global_tool_schema）可跨任务复用而不累积污染。
    """
    if not tool_schema or not profiles:
        return tool_schema or [], 0

    injected = 0
    enriched: list[dict[str, Any]] = []
    for entry in tool_schema:
        name = _tool_name(entry)
        profile = profiles.get(name) if name else None
        if isinstance(profile, dict) and profile.get("tool_id") not in (None, name):
            profile = {**profile, "metadata": {}}
        notes = format_profile_notes(
            profile, fields=fields, max_chars=max_chars, item_chars=item_chars
        )
        if not notes:
            enriched.append(entry)
            continue
        new_entry = deepcopy(entry)
        target = new_entry.get("function") if isinstance(new_entry.get("function"), dict) else new_entry
        base = str(target.get("description", "") or "").strip()
        target["description"] = f"{base}\n{notes}".strip() if base else notes
        enriched.append(new_entry)
        injected += 1
    return enriched, injected
