"""工具画像注入（profile_injection）+ /tool-profiles 载荷的单元测试。

不依赖服务/LLM：纯函数渲染与 schema 注入可直接单测；
服务侧 _tool_profile_payload 只测纯函数部分（不启动 FastAPI）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tevo.tool_pool.profile_injection import (
    ALL_PROFILE_FIELDS,
    DEFAULT_INJECT_FIELDS,
    DEFAULT_MAX_CHARS,
    apply_tool_profiles,
    format_profile_notes,
    profile_has_content,
)
from tevo.tool_pool.tool_schema import RecoveryHint, ToolSpec


def _openai_entry(name: str, description: str = "Base description.") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _full_profile() -> dict:
    return {
        "when_to_use": "当用户询问航班价格时。",
        "preconditions": ["需要先获取 user_id", "需要已登录"],
        "failure_cues": ["HTTP 504", "空结果"],
        "recovery_hints": [
            RecoveryHint(failure_mode="超时", error_signal="HTTP 504", alternative_tool="search_flights_v2"),
        ],
        "expected_outcomes": ["返回价格列表"],
        "input_requirements": {"user_id": "required (string)"},
    }


# ---------------------------------------------------------------------------
# format_profile_notes / profile_has_content
# ---------------------------------------------------------------------------

def test_empty_profile_renders_nothing():
    assert format_profile_notes(None) == ""
    assert format_profile_notes({}) == ""
    assert format_profile_notes({"description": "static only"}) == ""
    assert not profile_has_content({})
    assert not profile_has_content(None)


def test_profile_renders_all_fields_with_labels():
    notes = format_profile_notes(_full_profile())
    assert notes.startswith("[Evolved] ")
    for label in ("When to use:", "Preconditions:", "Watch out:", "If it fails:", "Expected:", "Inputs:"):
        assert label in notes
    # recovery hint 渲染成 failure_mode→alternative_tool
    assert "超时→search_flights_v2" in notes


def test_field_subset_selection():
    notes = format_profile_notes(_full_profile(), fields=("failure_cues",))
    assert "Watch out:" in notes
    assert "When to use:" not in notes
    assert "Preconditions:" not in notes


def test_list_items_and_total_length_caps():
    profile = {
        "when_to_use": "x" * 500,
        "preconditions": [f"precondition number {i} " + "y" * 120 for i in range(10)],
    }
    notes = format_profile_notes(profile, max_chars=200)
    assert len(notes) <= 200
    assert notes.endswith("…")
    # 每个列表字段最多 3 条
    many_items = format_profile_notes({"preconditions": [f"p{i}" for i in range(10)]}, max_chars=1000)
    assert many_items == "[Evolved] Preconditions: p0; p1; p2"


def test_dict_field_rendering():
    notes = format_profile_notes({"input_requirements": {"user_id": "required (string)"}})
    assert notes == "[Evolved] Inputs: user_id: required (string)"


def test_default_inject_fields_exclude_schema_restating_fields():
    """默认注入不含 input_requirements / expected_outcomes（两者都与 schema 信息重复，
    见 outputs/audits/profile_content_* 的内容审计）。"""
    assert set(DEFAULT_INJECT_FIELDS) == set(ALL_PROFILE_FIELDS) - {"input_requirements", "expected_outcomes"}
    only_expected = {"expected_outcomes": ["Returns object fields: result:integer"]}
    assert not profile_has_content(only_expected, fields=DEFAULT_INJECT_FIELDS)
    assert format_profile_notes(only_expected, fields=DEFAULT_INJECT_FIELDS) == ""


def test_profile_has_content_respects_fields():
    profile = _full_profile()
    assert profile_has_content(profile)
    assert profile_has_content(profile, fields=("failure_cues",))
    assert not profile_has_content(profile, fields=("nonexistent_field",))


# ---------------------------------------------------------------------------
# apply_tool_profiles
# ---------------------------------------------------------------------------

def test_apply_injects_only_matching_tools_and_counts():
    schema = [_openai_entry("tool_a"), _openai_entry("tool_b"), _openai_entry("tool_c")]
    profiles = {"tool_a": _full_profile(), "tool_c": {"when_to_use": "只在 C 用"}}
    enriched, injected = apply_tool_profiles(schema, profiles)

    assert injected == 2
    assert enriched[0]["function"]["description"].startswith("Base description.")
    assert "[Evolved]" in enriched[0]["function"]["description"]
    assert enriched[1]["function"]["description"] == "Base description."
    assert "[Evolved]" in enriched[2]["function"]["description"]


def test_apply_does_not_mutate_input_schema():
    schema = [_openai_entry("tool_a")]
    before = repr(schema)
    apply_tool_profiles(schema, {"tool_a": _full_profile()})
    assert repr(schema) == before, "base schema 必须保持干净（all-scope 跨任务复用）"


def test_apply_handles_flat_schema_shape():
    schema = [{"name": "tool_a", "description": "Flat."}]
    enriched, injected = apply_tool_profiles(schema, {"tool_a": {"when_to_use": "flat 用"}})
    assert injected == 1
    assert enriched[0]["description"].startswith("Flat.")
    assert "[Evolved]" in enriched[0]["description"]


def test_apply_no_profiles_returns_input_unchanged():
    schema = [_openai_entry("tool_a")]
    enriched, injected = apply_tool_profiles(schema, None)
    assert injected == 0
    assert enriched == schema
    enriched, injected = apply_tool_profiles(schema, {})
    assert injected == 0
    assert enriched == schema


def test_apply_respects_field_subset_and_caps():
    schema = [_openai_entry("tool_a")]
    enriched, injected = apply_tool_profiles(
        schema,
        {"tool_a": _full_profile()},
        fields=("when_to_use",),
        max_chars=DEFAULT_MAX_CHARS,
    )
    assert injected == 1
    desc = enriched[0]["function"]["description"]
    assert "When to use:" in desc
    assert "Watch out:" not in desc
