"""参数绑定辅助工具。

目标：
- 从 tool 输出中递归提取可复用的标量字段；
- 基于工具 schema / 参数名，对 available_params 做轻量绑定。
"""

from __future__ import annotations

import json
import ast
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tevo.tool_pool.tool_schema import ToolSpec
    from tevo.state.tevo_state import TEvoState

_SCALAR_TYPES = (str, int, float, bool, type(None))
_TOKEN_ALIASES = {
    "identifier": "id",
    "identifiers": "ids",
    "uuid": "id",
    "uid": "user_id",
}
_LOW_SIGNAL_TOKENS = {
    "name",
    "type",
    "content",
    "result",
    "value",
    "values",
    "data",
    "field",
    "fields",
    "item",
    "items",
    "current",
    "new",
    "old",
}
_FILESYSTEM_ENTITY_TYPES = {"file", "directory"}
_PLACEHOLDER_TOKENS = {
    "none",
    "null",
    "nil",
    "n/a",
    "na",
    "unknown",
    "undefined",
    "not set",
    "unset",
    "missing",
    "tbd",
}


def parse_output_payload(content: Any) -> Any | None:
    """尝试把工具输出解析成结构化对象。"""
    if isinstance(content, (Mapping, list)):
        return content
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def tool_output_is_error(content: Any) -> bool:
    """Recognize execution failures without interpreting ordinary returned text.

    BFCL returns JSON or Python-repr dictionaries. Only a root-level error or
    success=False signals failure; file bodies and query status values are data.
    """
    payload = parse_output_payload(content)
    if payload is None and isinstance(content, str):
        try:
            payload = ast.literal_eval(content.strip())
        except (SyntaxError, ValueError, TypeError):
            pass
    if isinstance(payload, Mapping):
        return bool(payload.get("error")) or payload.get("success") is False
    if payload is not None:
        return False
    text = str(content or "").strip().lower()
    return text.startswith(("[error]", "error during execution", "traceback (most recent call last)"))


def is_semantically_placeholder_value(value: Any) -> bool:
    """Return true when a scalar looks like a placeholder rather than a real argument value."""
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().strip("\"'").lower()
        if not normalized:
            return True
        return normalized in _PLACEHOLDER_TOKENS
    return False


def detect_payload_error(content: Any) -> tuple[bool, str]:
    """检测工具输出是否显式表示执行失败。

    返回:
        (has_error, error_message)
    """
    payload = parse_output_payload(content)
    if isinstance(payload, Mapping) and "error" in payload:
        error_value = payload.get("error")
        if error_value not in ("", None, False, 0, {}, []):
            return True, str(error_value)
    if isinstance(content, str):
        text = content.strip()
        if text.lower().startswith("error"):
            return True, text
    return False, ""


def extract_output_bindings(content: Any) -> dict[str, Any]:
    """从 JSON 字符串递归提取可复用字段。

    例：
    {"ticket": {"id": "T1"}} -> {"ticket.id": "T1", "ticket_id": "T1", "id": "T1"}
    """
    payload = parse_output_payload(content)
    if payload is None:
        return {}

    extracted: dict[str, Any] = {}
    _walk_payload(payload, extracted, path=())
    return extracted


def extract_goal_aligned_bindings(user_goal: str, context: Any) -> dict[str, Any]:
    """从 goal 和当前环境上下文中提炼更直接的参数别名。

    主要用于文件系统类任务：
    - 让 goal 中出现的目录/文件名变成可直接绑定的 param alias；
    - 为 goal 中出现但环境里尚不存在的目标目录（如待创建 temp）补出 dir_name/destination alias。
    """
    goal = str(user_goal or "").strip()
    if not goal:
        return {}

    aliases: dict[str, Any] = {}
    known_entity_details = _collect_named_entity_details(context)
    known_entities = [(detail["name"], detail["type"]) for detail in known_entity_details]
    known_names = {name.lower() for name, _ in known_entities}
    aliases.update(_aliases_from_path_hints(goal, known_entity_details))

    for name, entity_type in known_entities:
        for before, after in _find_goal_mentions(goal, name):
            aliases.update(_aliases_for_goal_mention(name, entity_type, before=before, after=after))

    quoted_mentions = _extract_quoted_literal_mentions(goal)
    for literal, before, after in quoted_mentions:
        if literal.lower() in known_names:
            continue
        aliases.update(_aliases_for_quoted_literal(literal, before=before, after=after))

    aliases.update(
        _aliases_from_comparison_literals(
            goal,
            [literal for literal, _, _ in quoted_mentions],
        )
    )

    # Extract source/destination from "move/copy X to Y" patterns
    aliases.update(_aliases_from_transfer_patterns(goal))

    # Extract unquoted file/dir mentions (e.g. bare "final_report.pdf" or "temp")
    aliases.update(_aliases_from_unquoted_file_mentions(goal, known_names))

    return aliases


def select_tool_param_bindings(tool: "ToolSpec | None", available_params: dict[str, Any]) -> dict[str, Any]:
    """为指定工具从 available_params 中挑选最可能合适的参数。"""
    if tool is None:
        return dict(available_params)

    schema = tool.parameter_schema if isinstance(tool.parameter_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return {}

    selected: dict[str, Any] = {}
    for param_name in list(tool.required_parameters) + [name for name in properties if name not in tool.required_parameters]:
        property_schema = properties.get(param_name, {})
        matched_key = resolve_binding_key(
            param_name,
            available_params,
            property_schema=property_schema,
            requirement_text=str(tool.input_requirements.get(param_name, "")),
        )
        if matched_key is not None and _schema_value_satisfies(available_params[matched_key], property_schema):
            selected[param_name] = available_params[matched_key]
    if any(param_name not in selected for param_name in tool.required_parameters):
        return {}
    return selected


def build_tool_param_context(
    tool: "ToolSpec | None",
    available_params: dict[str, Any],
    *,
    recent_output_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """为当前工具构造更强的参数绑定上下文。

    规则：
    - 继承已有 available_params；
    - 若最近一步结构化输出里能较高置信度匹配到当前工具参数，则为该参数补一层直接别名；
    - 不覆盖已有的精确参数键。
    """
    context = dict(available_params)
    if tool is None or not recent_output_bindings:
        return context

    schema = tool.parameter_schema if isinstance(tool.parameter_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    ordered_params = list(tool.required_parameters) + [name for name in properties if name not in tool.required_parameters]
    for param_name in ordered_params:
        if param_name in context:
            continue
        matched_key = resolve_binding_key(
            param_name,
            dict(recent_output_bindings),
            property_schema=properties.get(param_name, {}),
            requirement_text=str(tool.input_requirements.get(param_name, "")),
        )
        if matched_key is not None:
            context[param_name] = recent_output_bindings[matched_key]
    return context


def build_param_fit_context(
    tool: "ToolSpec | None",
    available_params: dict[str, Any],
    *,
    recent_output_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造供排序期 param_fit 使用的上下文，尽量与执行期保持一致。"""
    context = build_tool_param_context(
        tool,
        available_params,
        recent_output_bindings=recent_output_bindings,
    )
    return assemble_structured_params(tool, context)


def assemble_structured_params(
    tool: "ToolSpec | None",
    available_params: dict[str, Any],
) -> dict[str, Any]:
    """根据 schema 从扁平上下文中恢复对象/数组参数。"""
    context = dict(available_params)
    if tool is None:
        return context

    schema = tool.parameter_schema if isinstance(tool.parameter_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return context

    ordered_params = list(tool.required_parameters) + [name for name in properties if name not in tool.required_parameters]
    for param_name in ordered_params:
        if param_name in context:
            continue
        property_schema = properties.get(param_name, {})
        assembled = _assemble_value_for_schema(
            param_path=param_name,
            schema=property_schema,
            available_params=context,
            requirement_text=str(tool.input_requirements.get(param_name, "")),
        )
        if assembled is not None:
            context[param_name] = assembled
    return context


def resolve_param_source_value(source_label: str, state: "TEvoState") -> Any:
    """根据 parameter_template 的 source_label 从当前 state 中解引用真实值。"""
    if not source_label:
        return None
    namespace, _, remainder = source_label.partition(".")
    source_key = remainder if remainder else source_label

    named_sources: dict[str, Any] = {
        "available_params": state.available_params,
        "recent_output_bindings": state.recent_output_bindings,
    }
    metadata = getattr(state, "metadata", {})
    if isinstance(metadata, dict):
        named_sources["initial_output_bindings"] = metadata.get("initial_output_bindings", {})
    state_dump = state.state_dump
    if isinstance(state_dump, dict):
        observations = state_dump.get("observations", {})
        if isinstance(observations, dict):
            named_sources["latest_bindings"] = observations.get("latest_bindings", {})

    _missing = object()

    def _resolve_nested_path(value: Any, key: str) -> Any:
        current = value
        for part in key.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    return _missing
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return _missing
                current = current[idx]
                continue
            return _missing
        return current

    def _resolve_from(value: Any, key: str) -> Any:
        nested = _resolve_nested_path(value, key)
        if nested is not _missing:
            return nested
        flat_candidates = _flatten_binding_candidates(value)
        if key in flat_candidates:
            return flat_candidates[key]
        matched = resolve_binding_key(key, flat_candidates)
        if matched is not None:
            return flat_candidates[matched]
        return None

    if namespace in named_sources and remainder:
        return _resolve_from(named_sources[namespace], source_key)

    flat_candidates: dict[str, Any] = {}
    for value in named_sources.values():
        nested = _resolve_nested_path(value, source_key)
        if nested is not _missing:
            return nested
        for key, scalar in _flatten_binding_candidates(value).items():
            if key not in flat_candidates:
                flat_candidates[key] = scalar

    if source_key in flat_candidates:
        return flat_candidates[source_key]
    matched = resolve_binding_key(source_key, flat_candidates)
    if matched is not None:
        return flat_candidates[matched]
    return None


def _flatten_binding_candidates(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_binding_candidates(child, child_prefix))
        return flattened
    if isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            flattened.update(_flatten_binding_candidates(child, child_prefix))
        return flattened
    if prefix:
        flattened[prefix] = value
    return flattened


def apply_parameter_template_context(
    tool: "ToolSpec | None",
    state: "TEvoState",
    available_params: dict[str, Any],
    template: Any,
) -> dict[str, Any]:
    """将 parameter_template 转换成当前 state 可用的参数上下文。

    注：遗留路径（供 /select 旧选工具链使用）；统一记忆重构后
    ParameterTemplateRecord 已不存在，template 需自带 param_sources 属性。
    """
    context = dict(available_params)
    if tool is None or template is None:
        return context

    schema = tool.parameter_schema if isinstance(tool.parameter_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    for param_name in list(tool.required_parameters) + [name for name in properties if name not in tool.required_parameters]:
        if param_name in context:
            continue
        source_label = template.param_sources.get(param_name, "")
        resolved = resolve_param_source_value(source_label, state)
        if resolved is not None:
            context[param_name] = resolved
    return context


def _assemble_value_for_schema(
    *,
    param_path: str,
    schema: dict[str, Any],
    available_params: dict[str, Any],
    requirement_text: str,
) -> Any:
    expected_type = str(schema.get("type", "")).lower().strip()
    if not expected_type:
        matched = resolve_binding_key(param_path, available_params, property_schema=schema, requirement_text=requirement_text)
        if matched is None:
            matched = resolve_binding_key(param_path.split(".")[-1], available_params, property_schema=schema, requirement_text=requirement_text)
        return available_params.get(matched) if matched is not None else None

    if expected_type == "object":
        existing = available_params.get(param_path)
        if isinstance(existing, Mapping):
            return dict(existing)
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        assembled: dict[str, Any] = {}
        for child_name, child_schema in properties.items():
            child_path = f"{param_path}.{child_name}" if param_path else child_name
            child_value = _assemble_value_for_schema(
                param_path=child_path,
                schema=child_schema if isinstance(child_schema, dict) else {},
                available_params=available_params,
                requirement_text=requirement_text,
            )
            if child_value is not None:
                assembled[child_name] = child_value
        if required and not all(name in assembled for name in required):
            return None
        return assembled or None

    if expected_type == "array":
        existing = available_params.get(param_path)
        if isinstance(existing, list):
            return existing
        item_schema = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        flat_params = _flatten_binding_candidates(available_params)
        prefix = f"{param_path}." if param_path else ""
        indices = sorted({
            int(rest.split(".", 1)[0])
            for key in flat_params
            if key.startswith(prefix)
            for rest in [key[len(prefix):]]
            if rest and rest.split(".", 1)[0].isdigit()
        })
        if indices and indices != list(range(indices[-1] + 1)):
            # 稀疏索引若被压缩会改变位置语义；保守起见，不在此场景下组装数组。
            return None
        items: list[Any] = []
        for idx in indices:
            item_path = f"{param_path}.{idx}" if param_path else str(idx)
            item_value = _assemble_value_for_schema(
                param_path=item_path,
                schema=item_schema,
                available_params=available_params,
                requirement_text=requirement_text,
            )
            if item_value is None:
                # 即便原始索引连续，只要中间某个 item 无法组装，后续 append
                # 仍会导致元素前移，破坏位置语义；保守起见，整体放弃该数组参数。
                return None
            items.append(item_value)
        return items or None

    flat_params = _flatten_binding_candidates(available_params)
    if param_path in flat_params:
        return flat_params[param_path]
    parent_prefix = param_path.rsplit(".", 1)[0] + "." if "." in param_path else ""
    local_params = {
        key: value for key, value in flat_params.items()
        if parent_prefix and key.startswith(parent_prefix)
    }
    if local_params:
        return local_params.get(param_path)
    search_space = local_params or flat_params
    matched = resolve_binding_key(param_path, search_space, property_schema=schema, requirement_text=requirement_text)
    if matched is None and not local_params:
        matched = resolve_binding_key(
            param_path.split(".")[-1],
            search_space,
            property_schema=schema,
            requirement_text=requirement_text,
        )
    source_space = search_space if matched in search_space else flat_params
    return source_space.get(matched) if matched is not None else None


def compute_param_fit(tool: ToolSpec, available_params: dict[str, Any]) -> float:
    """计算 proposal 中近似的 ParamFit。

    近似实现：
    - required/preconditions-like 参数覆盖率
    - input_requirements 覆盖率
    最终取两者平均；若没有 input_requirements，则退化为 required 覆盖率。
    """
    return compute_param_fit_with_constraints(tool, available_params, constraints_snapshot=None)


def compute_param_fit_with_constraints(
    tool: "ToolSpec",
    available_params: dict[str, Any],
    constraints_snapshot: dict[str, Any] | None,
) -> float:
    schema = tool.parameter_schema if isinstance(tool.parameter_schema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    satisfied_names = {
        str(name): True
        for name in (constraints_snapshot or {}).get("satisfied_params", [])
    }
    unsatisfied_names = {
        str(name)
        for name in (constraints_snapshot or {}).get("unsatisfied_params", [])
    }

    required = list(tool.required_parameters)
    required_covered = 0
    for param_name in required:
        if _is_param_satisfied(
            param_name,
            available_params,
            satisfied_names,
            unsatisfied_names,
            property_schema=properties.get(param_name, {}),
            requirement_text=str(tool.input_requirements.get(param_name, "")),
        ):
            required_covered += 1
    required_ratio = 1.0 if not required else required_covered / len(required)

    if not tool.input_requirements:
        return round(required_ratio, 6)

    input_covered = 0
    for param_name, requirement_text in tool.input_requirements.items():
        if _is_param_satisfied(
            param_name,
            available_params,
            satisfied_names,
            unsatisfied_names,
            property_schema=properties.get(param_name, {}),
            requirement_text=str(requirement_text),
        ):
            input_covered += 1
    input_ratio = input_covered / max(len(tool.input_requirements), 1)
    return round((required_ratio + input_ratio) / 2.0, 6)


def _is_param_satisfied(
    param_name: str,
    available_params: dict[str, Any],
    satisfied_names: dict[str, bool],
    unsatisfied_names: set[str],
    *,
    property_schema: dict[str, Any],
    requirement_text: str,
) -> bool:
    matched_key = resolve_binding_key(
        param_name,
        available_params,
        property_schema=property_schema,
        requirement_text=requirement_text,
    )
    if matched_key is not None and _schema_value_satisfies(available_params.get(matched_key), property_schema):
        return True
    if resolve_binding_key(param_name, satisfied_names):
        return True
    if param_name in unsatisfied_names:
        return False
    return False


def _schema_value_satisfies(value: Any, schema: dict[str, Any] | None) -> bool:
    if value is None:
        return False
    if not isinstance(schema, dict) or not schema:
        return not is_semantically_placeholder_value(value)

    expected_type = str(schema.get("type", "")).lower().strip()
    if expected_type == "object":
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        for child_name in required:
            if child_name not in value:
                return False
            child_schema = properties.get(child_name, {}) if isinstance(properties.get(child_name), dict) else {}
            if not _schema_value_satisfies(value.get(child_name), child_schema):
                return False
        return True

    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        return all(_schema_value_satisfies(item, item_schema) for item in value)

    if expected_type == "string":
        if not isinstance(value, str):
            return False
        normalized = value.strip()
        if not normalized or is_semantically_placeholder_value(normalized):
            return False
        enum_values = _extract_semantic_enum_values(schema)
        if enum_values and normalized.lower() not in {item.lower() for item in enum_values}:
            return False
        return True
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    enum_values = _extract_semantic_enum_values(schema)
    if enum_values and isinstance(value, str):
        return value.strip().lower() in {item.lower() for item in enum_values}
    return True


def _extract_semantic_enum_values(schema: dict[str, Any]) -> list[str]:
    if not isinstance(schema, dict):
        return []
    values: list[str] = []
    enum_payload = schema.get("enum")
    if isinstance(enum_payload, list):
        values.extend(str(item).strip() for item in enum_payload if str(item).strip())
    description = str(schema.get("description", "") or "")
    if description:
        for pattern in (
            r"\[Enum\]\s*:\s*\[(.*?)\]",
            r"(?i)\benum\b\s*:\s*\[(.*?)\]",
        ):
            match = re.search(pattern, description)
            if not match:
                continue
            raw_items = [part.strip().strip("\"'") for part in match.group(1).split(",")]
            values.extend(item for item in raw_items if item)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def resolve_binding_key(
    param_name: str,
    available_params: dict[str, Any],
    *,
    property_schema: dict[str, Any] | None = None,
    requirement_text: str = "",
) -> str | None:
    """返回最适合绑定到 param_name 的 available_params 键。"""
    return _best_binding_key(
        param_name,
        available_params,
        property_schema=property_schema or {},
        requirement_text=requirement_text,
    )


def _walk_payload(value: Any, extracted: dict[str, Any], path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            _walk_payload(child, extracted, path + (key,))
        return

    if isinstance(value, list):
        if path:
            dotted = ".".join(path)
            underscored = "_".join(path)
            leaf = path[-1]
            extracted.setdefault(dotted, value)
            extracted.setdefault(underscored, value)
            extracted.setdefault(leaf, value)
            if value and all(not isinstance(item, (Mapping, list)) for item in value[:5]) and leaf.endswith("identifier"):
                prefix = leaf[: -len("identifier")].rstrip("_")
                if prefix:
                    extracted.setdefault(f"{prefix}_id", value)
        for idx, item in enumerate(value[:5]):
            if isinstance(item, (Mapping, list)):
                _walk_payload(item, extracted, path + (str(idx),))
        return

    if not isinstance(value, _SCALAR_TYPES) or not path:
        return

    dotted = ".".join(path)
    underscored = "_".join(path)
    leaf = path[-1]

    extracted.setdefault(dotted, value)
    extracted.setdefault(underscored, value)
    extracted.setdefault(leaf, value)

    # 常见嵌套对象别名：ticket.id -> ticket_id
    if len(path) >= 2:
        extracted.setdefault(f"{path[-2]}_{leaf}", value)
        if leaf == "id":
            extracted.setdefault(f"{path[-2]}_identifier", value)
        if leaf == "identifier":
            extracted.setdefault(f"{path[-2]}_id", value)
        if leaf == "ids":
            extracted.setdefault(f"{path[-2]}_ids", value)

    if leaf.endswith("identifier"):
        prefix = leaf[: -len("identifier")].rstrip("_")
        if prefix:
            extracted.setdefault(f"{prefix}_id", value)


def _best_binding_key(
    param_name: str,
    available_params: dict[str, Any],
    *,
    property_schema: dict[str, Any],
    requirement_text: str,
) -> str | None:
    best_key: str | None = None
    best_score = -10_000
    target_norm = _normalize_key(param_name)
    target_tokens = _tokenize(param_name)
    requirement_tokens = _tokenize(requirement_text)

    for source_key, source_value in available_params.items():
        score = _binding_score(
            target_norm,
            target_tokens,
            source_key,
            source_value,
            property_schema,
            requirement_tokens,
        )
        if score > best_score:
            best_score = score
            best_key = source_key

    return best_key if best_score >= 10 else None


def _binding_score(
    target_norm: str,
    target_tokens: set[str],
    source_key: str,
    source_value: Any,
    property_schema: dict[str, Any],
    requirement_tokens: set[str],
) -> int:
    source_norm = _normalize_key(source_key)
    exact_like_match = False
    if source_norm == target_norm:
        score = 100
        exact_like_match = True
    else:
        score = 0

    leaf = source_key.split(".")[-1]
    leaf_norm = _normalize_key(leaf)
    if leaf_norm == target_norm:
        score = max(score, 90)
        exact_like_match = True

    underscored_norm = _normalize_key(source_key.replace(".", "_"))
    if underscored_norm == target_norm:
        score = max(score, 85)
        exact_like_match = True

    if source_norm.startswith(target_norm) and len(source_norm) > len(target_norm):
        score = max(score, 75)

    source_tokens = _tokenize(source_key)
    overlap_tokens = target_tokens & source_tokens
    informative_overlap = overlap_tokens - _LOW_SIGNAL_TOKENS
    if informative_overlap:
        score = max(score, 10 * len(informative_overlap))

    req_overlap = len(requirement_tokens & source_tokens)
    if req_overlap:
        score += 15 * req_overlap

    if not exact_like_match and not informative_overlap and req_overlap == 0 and not source_norm.startswith(target_norm):
        return -10_000

    score += _type_compatibility_bonus(property_schema, source_value)
    score += _specificity_bonus(target_tokens, source_key)

    return score


def _type_compatibility_bonus(property_schema: dict[str, Any], source_value: Any) -> int:
    expected_type = str(property_schema.get("type", "")).lower().strip()
    if not expected_type:
        return 0

    if expected_type == "string":
        return 20 if isinstance(source_value, str) else -80
    if expected_type == "integer":
        return 35 if isinstance(source_value, int) and not isinstance(source_value, bool) else -120
    if expected_type == "number":
        return 35 if isinstance(source_value, (int, float)) and not isinstance(source_value, bool) else -120
    if expected_type == "boolean":
        return 35 if isinstance(source_value, bool) else -120
    if expected_type == "array":
        return 20 if isinstance(source_value, list) else -120
    if expected_type == "object":
        return 20 if isinstance(source_value, Mapping) else -120
    return 0


def _specificity_bonus(target_tokens: set[str], source_key: str) -> int:
    source_tokens = _tokenize(source_key)
    bonus = 0

    # For specific *_id parameters, prefer contextual keys like ticket.id over plain id.
    if "id" in target_tokens and len(target_tokens) > 1:
        if source_key.split(".")[-1].lower() == "id" and len(source_tokens - {"id"}) == 0:
            bonus -= 30
        if len((target_tokens - {"id"}) & (source_tokens - {"id"})) > 0:
            bonus += 20

    if "." in source_key:
        bonus += 5
    return bonus


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _tokenize(value: str) -> set[str]:
    pieces = re.split(r"[^a-zA-Z0-9]+", value.lower().replace(".", "_"))
    tokens = {piece for piece in pieces if piece}
    combined = value.lower().replace(".", "_")
    if "_" in combined:
        tokens.update(piece for piece in combined.split("_") if piece)
    normalized_tokens = set(tokens)
    for token in list(tokens):
        alias = _TOKEN_ALIASES.get(token)
        if alias:
            normalized_tokens.update(piece for piece in alias.split("_") if piece)
    return normalized_tokens


def _collect_named_entities(context: Any) -> list[tuple[str, str]]:
    entities: dict[str, str] = {}

    def _walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            node_type = str(value.get("type", "")).lower().strip()
            if node_type in _FILESYSTEM_ENTITY_TYPES and path:
                name = str(path[-1]).strip()
                if name and name not in {"root", "contents"}:
                    entities.setdefault(name, node_type)

            contents = value.get("contents")
            if isinstance(contents, Mapping):
                for child_name, child in contents.items():
                    if not isinstance(child_name, str):
                        continue
                    _walk(child, path + (child_name,))
                return

            for key, child in value.items():
                if key == "contents" or not isinstance(key, str):
                    continue
                if isinstance(child, (Mapping, list)):
                    _walk(child, path + (key,))
            return

        if isinstance(value, list):
            for idx, item in enumerate(value[:20]):
                _walk(item, path + (str(idx),))

    _walk(context, ())
    return list(entities.items())


def _collect_named_entity_details(context: Any) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def _walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            node_type = str(value.get("type", "")).lower().strip()
            if node_type in _FILESYSTEM_ENTITY_TYPES and path:
                name = str(path[-1]).strip()
                if name and name not in {"root", "contents"}:
                    identity = (name, node_type, path)
                    if identity not in seen:
                        seen.add(identity)
                        details.append(
                            {
                                "name": name,
                                "type": node_type,
                                "path": path,
                                "parent_name": str(path[-2]).strip() if len(path) >= 2 else "",
                            }
                        )

            contents = value.get("contents")
            if isinstance(contents, Mapping):
                for child_name, child in contents.items():
                    if not isinstance(child_name, str):
                        continue
                    _walk(child, path + (child_name,))
                return

            for key, child in value.items():
                if key == "contents" or not isinstance(key, str):
                    continue
                if isinstance(child, (Mapping, list)):
                    _walk(child, path + (key,))
            return

        if isinstance(value, list):
            for idx, item in enumerate(value[:20]):
                _walk(item, path + (str(idx),))

    _walk(context, ())
    return details


def _extract_quoted_literals(goal: str) -> list[str]:
    return [literal for literal, _, _ in _extract_quoted_literal_mentions(goal)]


def _extract_quoted_literal_mentions(goal: str) -> list[tuple[str, str, str]]:
    mentions: list[tuple[str, str, str]] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9])'([^'\n]{1,160})'(?![A-Za-z0-9])|\"([^\"\n]{1,160})\""
    )
    for match in pattern.finditer(goal):
        literal = (match.group(1) or match.group(2) or "").strip()
        if not literal:
            continue
        start, end = match.span()
        before = goal[max(0, start - 40):start]
        after = goal[end:min(len(goal), end + 40)]
        mention = (literal, before, after)
        if mention not in mentions:
            mentions.append(mention)
    return mentions


def _aliases_from_path_hints(goal: str, details: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not goal:
        return aliases

    mentioned_files: list[dict[str, Any]] = []
    mentioned_destination_dirs: list[dict[str, Any]] = []
    for detail in details:
        name = str(detail.get("name", "")).strip()
        entity_type = str(detail.get("type", "")).strip()
        if not name or entity_type not in _FILESYSTEM_ENTITY_TYPES:
            continue
        mentions = _find_goal_mentions(goal, name)
        if not mentions:
            continue
        if entity_type == "file":
            mentioned_files.append(detail)
        elif entity_type == "directory":
            if any(_goal_mention_looks_like_destination(before, after) for before, after in mentions):
                mentioned_destination_dirs.append(detail)

    unique_file_names = {str(detail.get("name", "")).strip() for detail in mentioned_files if str(detail.get("name", "")).strip()}
    if len(unique_file_names) == 1 and mentioned_files:
        file_detail = mentioned_files[0]
        file_name = str(file_detail.get("name", "")).strip()
        parent_name = str(file_detail.get("parent_name", "")).strip()
        if file_name:
            aliases.setdefault("file_name", file_name)
            if _goal_requires_file_source(goal):
                aliases.setdefault("source", file_name)
        if parent_name and _goal_likely_needs_parent_directory(goal):
            aliases.setdefault("folder", parent_name)

    unique_destination_names = {
        str(detail.get("name", "")).strip()
        for detail in mentioned_destination_dirs
        if str(detail.get("name", "")).strip()
    }
    if len(unique_destination_names) == 1 and mentioned_destination_dirs:
        aliases.setdefault("destination", str(mentioned_destination_dirs[0].get("name", "")).strip())

    return aliases


def _find_goal_mentions(goal: str, term: str) -> list[tuple[str, str]]:
    goal_lower = goal.lower()
    term_lower = term.lower()
    variants = {term_lower}
    if term_lower.endswith("s") and len(term_lower) > 3:
        variants.add(term_lower[:-1])
    elif len(term_lower) > 3:
        variants.add(term_lower + "s")
    mentions: list[tuple[str, str]] = []
    for variant in sorted(variants, key=len, reverse=True):
        start = 0
        while True:
            idx = goal_lower.find(variant, start)
            if idx < 0:
                break
            end = idx + len(variant)
            prev_char = goal_lower[idx - 1] if idx > 0 else " "
            next_char = goal_lower[end] if end < len(goal_lower) else " "
            if not prev_char.isalnum() and not next_char.isalnum():
                mention = (goal_lower[max(0, idx - 32):idx], goal_lower[end:min(len(goal_lower), end + 32)])
                if mention not in mentions:
                    mentions.append(mention)
            start = idx + len(variant)
    return mentions


def _aliases_for_goal_mention(
    name: str,
    entity_type: str | None,
    *,
    before: str,
    after: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    literal = str(name).strip()
    if not literal:
        return aliases

    before_tail = _normalize_goal_context(before[-20:])
    after_head = _normalize_goal_context(after[:24])
    looks_like_file = bool(entity_type == "file" or "." in literal)
    looks_like_directory = bool(
        entity_type == "directory"
        or re.match(r"^(directory|folder|path)\b", after_head)
        or re.search(r"\b(within|inside|under|directory|folder)$", before_tail)
    )

    if looks_like_file:
        aliases.setdefault(f"file_name_{literal}", literal)
        if re.search(r"\b(move|copy|read|open|sort|remove|delete|find)$", before_tail):
            aliases.setdefault(f"source_{literal}", literal)

    if looks_like_directory:
        aliases.setdefault(f"folder_{literal}", literal)
        if re.search(r"\b(to|into|inside)( the)?$", before_tail):
            aliases.setdefault(f"destination_{literal}", literal)
        if re.search(r"\b(to|into|create|make|mkdir)( the)?$", before_tail):
            aliases.setdefault(f"dir_name_{literal}", literal)
        if _goal_mention_looks_like_navigation_target(before, after):
            aliases.setdefault("folder", literal)

    if not looks_like_file and not looks_like_directory and re.search(r"\b(to|into|create|make)( the)?$", before_tail):
        aliases.setdefault(f"destination_{literal}", literal)
        aliases.setdefault(f"dir_name_{literal}", literal)

    return aliases


def _aliases_for_quoted_literal(
    literal: str,
    *,
    before: str,
    after: str,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    value = str(literal).strip()
    if not value:
        return aliases

    before_tail = _normalize_goal_context(before[-32:])
    after_head = _normalize_goal_context(after[:32])
    file_context = bool(
        re.search(r"\b(file|document|report|log|notes?|summary|proposal)(\s+(named|called|titled))?$", before_tail)
        or re.match(r"^(file|document|report|log|notes?|summary|proposal)\b", after_head)
    )
    directory_context = bool(
        re.search(r"\b(directory|folder|path)$", before_tail)
        or re.match(r"^(directory|folder|path)\b", after_head)
    )
    looks_like_file = bool(
        "." in value
        or file_context
    )
    file_name_signal = bool(
        re.search(r"\b(file|document|report|log)\s+(named|called|titled)$", before_tail)
        or re.search(r"\b(labeling|label|name|named|call|called|title|titled)( it| the file)? as$", before_tail)
        or re.search(r"\bcall it$", before_tail)
    )
    dir_name_signal = bool(
        re.search(r"\b(directory|folder|path)\s+(named|called)$", before_tail)
        or re.search(r"\b(name|named|call|called)( it| the directory| the folder)? as$", before_tail)
        or re.search(r"\bcall it$", before_tail)
    )
    pattern_signal = bool(
        re.search(r"\b(term|pattern|keyword|query|match|matches|matching|contains|contain)$", before_tail)
        or re.search(r"\b(features|feature|search|grep|extract)\b", before_tail)
    )

    if looks_like_file:
        aliases.setdefault(f"file_name_{value}", value)
        # Always create direct file_name binding for quoted file literals,
        # not just when preceded by "file named/called/titled"
        aliases.setdefault("file_name", value)
        if file_name_signal:
            aliases.setdefault("file_name", value)

    if pattern_signal and not looks_like_file:
        aliases.setdefault("pattern", value)
        aliases.setdefault(f"pattern_{value}", value)

    if (dir_name_signal or directory_context) and not looks_like_file:
        aliases.setdefault("dir_name", value)
        aliases.setdefault(f"dir_name_{value}", value)
    if directory_context and not looks_like_file:
        aliases.setdefault(f"destination_{value}", value)

    return aliases


def _aliases_from_transfer_patterns(goal: str) -> dict[str, str]:
    """Extract source/destination from move/copy/transfer patterns.

    Handles: "move/copy/transfer X to Y", where X and Y can be quoted
    strings or file/dir references.
    """
    aliases: dict[str, str] = {}
    normalized = _normalize_goal_context(goal)

    # "move/copy/transfer/send X to Y"
    m = re.search(
        r"\b(move|copy|transfer|send|rename)\s+(.+?)\s+to\s+(.+?)(?:\s+(?:in|into|within|from|at|and|for|the|a|an)\b|$)",
        normalized,
    )
    if not m:
        return aliases

    source_val = _extract_source_from_original_goal(goal)
    dest_val = _extract_dest_from_original_goal(goal)

    if source_val:
        aliases.setdefault("source", source_val)
        aliases.setdefault(f"source_{source_val.replace('.', '_')}", source_val)
    if dest_val:
        aliases.setdefault("destination", dest_val)
        aliases.setdefault(f"destination_{dest_val.replace('.', '_')}", dest_val)

    return aliases


def _extract_source_from_original_goal(goal: str) -> str | None:
    """Extract the source file/path from the original (non-normalized) goal."""
    # "move/copy/transfer 'X' ... to 'Y'" - extract X
    m = re.search(
        r"\b[Mm]ove(?:s)?\s+(['\"])(.+?)\1\s+.*?\bto\b", goal,
    ) or re.search(
        r"\b(?:copy|transfer|send|rename)\s+(['\"])(.+?)\1\s+.*?\bto\b", goal,
    )
    if m:
        return m.group(2).strip()

    # Fallback: first quoted string after move/copy
    m = re.search(r"\b[Mm]ove(?:s)?\s+(?:the\s+)?(?:file\s+)?(['\"])(.+?)\1", goal)
    if m:
        return m.group(2).strip()

    return None


def _extract_dest_from_original_goal(goal: str) -> str | None:
    """Extract the destination from the original (non-normalized) goal."""
    # "... to 'Y'" or "... to Y directory/folder"
    m = re.search(r"\bto\s+(['\"])(.+?)\1", goal)
    if m:
        return m.group(2).strip()

    # "... to X directory/folder"
    m = re.search(r"\bto\s+(['\"]?)(\S+?)\1\s+(?:directory|folder|dir)\b", goal)
    if m:
        return m.group(2).strip()

    return None


def _aliases_from_comparison_literals(goal: str, quoted_literals: list[str]) -> dict[str, str]:
    normalized_goal = _normalize_goal_context(goal)
    if not re.search(r"\b(compare|comparison|diff|difference|between|versus|vs)\b", normalized_goal):
        return {}

    file_literals: list[str] = []
    for literal in quoted_literals:
        value = str(literal).strip()
        if not value or "." not in value or value in file_literals:
            continue
        file_literals.append(value)
    if len(file_literals) < 2:
        return {}
    return {
        "file_name1": file_literals[0],
        "file_name2": file_literals[1],
    }


def _normalize_goal_context(fragment: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", fragment.lower()).strip()


def _goal_requires_file_source(goal: str) -> bool:
    normalized = _normalize_goal_context(goal)
    return bool(re.search(r"\b(find|locat|move|copy|read|open|sort|remove|delete|display|show)\b", normalized))


def _goal_likely_needs_parent_directory(goal: str) -> bool:
    normalized = _normalize_goal_context(goal)
    return bool(
        re.search(r"\b(same directory|same folder|upon locating|after locating)\b", normalized)
        or re.search(r"\b(find|locat)\b", normalized)
    )


def _aliases_from_unquoted_file_mentions(
    goal: str,
    known_names: set[str],
) -> dict[str, str]:
    """从 goal 中提取未加引号的文件/目录名提及。

    处理 BFCL 风格的 goal：Create a new directory called temp ... Move final_report.pdf into temp.
    其中 final_report.pdf 和 temp 没有引号包裹。
    """
    aliases: dict[str, str] = {}
    # Use original goal for file pattern matching to preserve dots/underscores
    file_pattern = re.compile(r"\b([a-zA-Z0-9_][a-zA-Z0-9_.-]*\.[a-zA-Z0-9]{2,6})\b")
    seen_files: set[str] = set()
    for match in file_pattern.finditer(goal):
        value = match.group(1)
        if value.lower() in known_names:
            continue
        if value.lower() in {"e.g", "i.e"}:
            continue
        # Avoid subsumed matches (e.g. "report.pdf" within "final_report.pdf")
        if any(value in other and value != other for other in seen_files):
            continue
        seen_files = {other for other in seen_files if other not in value or other == value}
        seen_files.add(value)
        aliases["file_name"] = value
        aliases[f"file_name_{value}"] = value

    # Match directory-like tokens after "directory"/"folder" keywords
    goal_lower = goal.lower()
    dir_pattern = re.compile(
        r"\b(directory|folder)\s+(?:called|named)\s+([a-zA-Z0-9_][a-zA-Z0-9_]*)"
    )
    for match in dir_pattern.finditer(goal_lower):
        value = match.group(2)
        if value.lower() in known_names:
            continue
        aliases.setdefault("dir_name", value)
        aliases.setdefault(f"dir_name_{value}", value)

    # Match "into/in/inside the X directory/folder" patterns
    bare_dir_pattern = re.compile(
        r"\b(into|in|inside|under)\s+(?:the\s+)?([a-zA-Z0-9_][a-zA-Z0-9_]*)\s+(?:directory|folder)\b"
    )
    for match in bare_dir_pattern.finditer(goal_lower):
        value = match.group(2)
        if value.lower() in known_names:
            continue
        if value in {"current", "same", "target", "source", "root", "parent"}:
            continue
        aliases.setdefault("destination", value)
        aliases.setdefault(f"destination_{value}", value)

    return aliases


def _goal_mention_looks_like_destination(before: str, after: str) -> bool:
    before_tail = _normalize_goal_context(before[-24:])
    after_head = _normalize_goal_context(after[:24])
    if re.search(r"\b(to|into|inside)( the)?$", before_tail):
        return True
    return bool(
        re.search(r"\b(move|copy|put|place|transfer)\b", before_tail)
        and re.match(r"^(directory|folder|path)\b", after_head)
    )


def _goal_mention_looks_like_navigation_target(before: str, after: str) -> bool:
    before_tail = _normalize_goal_context(before[-24:])
    after_head = _normalize_goal_context(after[:32])
    if re.search(r"\b(go into|enter|navigate to|change directory to|switch to)\s*$", before_tail):
        return True
    if re.search(r"\b(within|inside|under)\s*$", before_tail):
        return bool(re.match(r"^(directory|folder|path)\b", after_head))
    if re.search(r"\b(same directory|same folder)\s*(of|as)?\s*$", before_tail):
        return bool(re.match(r"^(directory|folder|path)\b", after_head))
    return False
