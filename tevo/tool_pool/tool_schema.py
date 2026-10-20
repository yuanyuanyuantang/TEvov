"""TEvo 工具 schema — 1 静态字段 + 6 可进化字段。"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from tevo.reproducibility import canonical_hash


@dataclass(slots=True)
class RecoveryHint:
    """Observational repair candidate; legacy co-occurrence is not causal evidence."""

    failure_mode: str       # 失败模式，如 "超时"、"参数缺失"
    error_signal: str       # 可观测的错误信号，如 "HTTP 504"、"空结果"
    alternative_tool: str   # 推荐的替代工具
    confidence: float = 0.5  # 基于历史 co-occurrence 频率
    validation_status: str = "legacy_cooccurrence"


@dataclass(slots=True)
class ToolSpec:
    """Tool Profile: desc_u (static) + 6 evolvable fields."""

    # Identity
    tool_id: str
    tool_name: str

    # Static field (desc_u) — never overwritten by evolution
    description: str

    # Evolvable fields (z_u \ {desc_u})
    when_to_use: str = ""                        # when_u — 适用条件
    preconditions: list[str] = field(default_factory=list)   # pre_u — 前置条件
    input_requirements: dict[str, str] = field(default_factory=dict)  # in_u — {参数名: 来源描述}
    expected_outcomes: list[str] = field(default_factory=list)   # out_u — 预期输出
    failure_cues: list[str] = field(default_factory=list)        # fail_u — 失败信号
    recovery_hints: list[RecoveryHint] = field(default_factory=list)  # recover_u — 结构化恢复提示

    # 统计字段（自动维护）
    success_count: int = 0
    fail_count: int = 0
    last_updated: float = 0.0
    deprecated: bool = False

    # Metadata (not part of z_u)
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def safe_expected_outcomes(self, limit: int | None = None) -> list[str]:
        initialization = self.metadata.get("contract_initialization", {}) if isinstance(self.metadata, dict) else {}
        bound_quotes = set()
        if (isinstance(initialization, dict) and initialization.get("policy") == "contract_quotes"
            and initialization.get("contract_sha256") == canonical_hash({
                "tool_id": self.tool_id, "description": self.description,
                "parameters": self.parameter_schema, "response_schema": _response_schema(self),
            })):
            declared_hashes = initialization.get("expected_outcome_quotes_sha256", [])
            bound_quotes = set(value for value in declared_hashes if isinstance(value, str)) & {
                canonical_hash(item) for item in _contract_expected_outcomes(_response_schema(self))
            } if isinstance(declared_hashes, list) else set()
        outcomes = []
        for item in self.expected_outcomes:
            text = str(item).strip()
            if not text:
                continue
            # This namespace denotes a verified contract quote, including
            # scalar/array JSON that the legacy literal detector does not catch.
            if text.startswith("Declared response "):
                if canonical_hash(text) not in bound_quotes:
                    continue
            elif _looks_like_literal_expected_outcome(item):
                continue
            outcomes.append(text)
        if limit is None:
            return outcomes
        return outcomes[: max(int(limit), 0)]

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI-compatible tool schema。"""
        parameter_schema = deepcopy(self.parameter_schema) if isinstance(self.parameter_schema, dict) else {}
        if not parameter_schema:
            parameter_schema = {"type": "object", "properties": {}}
        if not isinstance(parameter_schema.get("properties"), dict):
            parameter_schema["properties"] = {}
        if "type" not in parameter_schema:
            parameter_schema["type"] = "object"
        description_parts = [self.description.strip()]
        if self.when_to_use:
            description_parts.append(f"When to use: {self.when_to_use.strip()}")
        if self.preconditions:
            description_parts.append("Preconditions: " + "; ".join(str(item).strip() for item in self.preconditions[:4] if str(item).strip()))
        safe_outcomes = self.safe_expected_outcomes(3)
        if safe_outcomes:
            description_parts.append("Expected outcomes: " + "; ".join(safe_outcomes))
        description = " ".join(part for part in description_parts if part).strip() or self.tool_name
        return {
            "type": "function",
            "function": {
                "name": self.tool_id,
                "description": description,
                "parameters": parameter_schema,
            },
        }

    @property
    def required_parameters(self) -> list[str]:
        properties = self.parameter_schema.get("properties", {})
        required = self.parameter_schema.get("required", [])
        if isinstance(required, list) and required:
            return [str(item) for item in required]
        if isinstance(properties, dict):
            return [
                str(key)
                for key, value in properties.items()
                if isinstance(value, dict) and value.get("required")
            ]
        return []

    def search_text(self) -> str:
        """序列化文本，用于 embedding。包含所有可进化字段。"""
        recovery_text = ", ".join(
            f"{h.failure_mode}→{h.alternative_tool}" for h in self.recovery_hints
        )
        input_req_text = ", ".join(self.input_requirements.keys()) if self.input_requirements else ""
        outcome_text = ", ".join(self.safe_expected_outcomes())
        return "\n".join(
            [
                f"tool_id: {self.tool_id}",
                f"tool_name: {self.tool_name}",
                f"description: {self.description}",
                f"when_to_use: {self.when_to_use}",
                f"preconditions: {', '.join(self.preconditions)}",
                f"input_requirements: {input_req_text}",
                f"expected_outcomes: {outcome_text}",
                f"failure_cues: {', '.join(self.failure_cues)}",
                f"recovery_hints: {recovery_text}",
            ]
        )

    @property
    def total_calls(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.5
        return self.success_count / self.total_calls

    def record_outcome(self, success: bool) -> None:
        """记录一次调用结果并自动维护统计字段。"""
        import time
        if success:
            self.success_count += 1
            self.metadata["consecutive_failures"] = 0
        else:
            self.fail_count += 1
            self.metadata["consecutive_failures"] = int(self.metadata.get("consecutive_failures", 0) or 0) + 1
        self.last_updated = time.time()

    def check_health(
        self,
        *,
        min_calls: int = 10,
        fail_rate_threshold: float = 0.8,
        consecutive_fails: int = 0,
    ) -> bool:
        """检查工具健康度。连续失败率高 → deprecated。

        返回 True 表示工具仍健康，False 表示应标记为 deprecated。
        """
        import os

        # In BFCL/iBFCL a failed tool execution usually means the model chose a
        # bad argument/order, not that the tool itself is unavailable. Hiding
        # core tools after such failures hurts memory-context retrieval, so keep
        # deprecation opt-in for experiments that explicitly need it.
        if os.environ.get("TEVO_ENABLE_TOOL_DEPRECATION", "").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            self.deprecated = False
            return True
        if consecutive_fails >= 5:
            self.deprecated = True
            return False
        if self.total_calls < min_calls:
            return not self.deprecated
        if self.fail_count / self.total_calls >= fail_rate_threshold:
            self.deprecated = True
            return False
        # 如果之前 deprecated 但现在表现改善，可以恢复
        if self.deprecated and self.success_rate > 0.5:
            self.deprecated = False
        return not self.deprecated

    @classmethod
    def from_openai_function(cls, function: dict[str, Any]) -> "ToolSpec":
        """从 OpenAI function schema 创建 ToolSpec。

        自动从参数 schema 推导 input_requirements 和 preconditions：
        - input_requirements: 所有参数按 required/optional 标记
        - preconditions: 从 ID 类必需参数推导（如 user_id → "需要先获取 user_id"）
        """
        name = str(function.get("name", "")).strip()
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters", {}) if isinstance(function.get("parameters"), dict) else {}
        response_schema = function.get("response", {}) if isinstance(function.get("response"), dict) else {}

        # 从参数 schema 推导 input_requirements 和 preconditions
        input_requirements: dict[str, str] = {}
        preconditions: list[str] = []
        properties = parameters.get("properties", {}) if isinstance(parameters.get("properties"), dict) else {}
        required_params = set(parameters.get("required", [])) if isinstance(parameters.get("required"), list) else set()
        for param_name, param_def in properties.items():
            if isinstance(param_def, dict):
                param_type = param_def.get("type", "unknown")
                if param_name in required_params:
                    input_requirements[param_name] = f"required ({param_type})"
                    # ID 类必需参数 → 推导前置条件
                    if _is_resource_id_param(param_name):
                        preconditions.append(f"需要已获取 {param_name}")
                else:
                    input_requirements[param_name] = f"optional ({param_type})"

        return cls(
            tool_id=name,
            tool_name=name,
            description=description,
            parameter_schema=parameters,
            input_requirements=input_requirements,
            preconditions=preconditions,
            metadata={
                "source": "openai-function-schema",
                "response_schema": response_schema,
            },
        )

    def init_from_contract(self) -> "ToolSpec":
        """Quote the declared interface without generating execution claims.

        This intentionally describes the supplied schema, which can differ
        from an implementation. No output examples, prerequisite calls,
        failure modes or aliases are inferred. Repeated calls are identical.
        """
        parameters = deepcopy(self.parameter_schema)
        response = deepcopy(_response_schema(self))
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        if (not isinstance(required, list) or any(not isinstance(name, str) for name in required)
            or not isinstance(properties, dict)):
            raise ValueError("Contract initialization requires declared required names and properties")
        def quote(value):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.when_to_use = (
            "Declared schema documentation only; not an execution guarantee. "
            "All required fields jointly, as declared: " + quote(required) + ".\n"
            "Description quoted verbatim:\n" + self.description + "\n"
            "Declared parameter schema (including optional/default/enum declarations): " + quote(parameters)
        )
        self.preconditions = ["Declared required-presence only: " + quote(name) for name in required]
        self.input_requirements = {
            name: "Declared " + ("required" if name in required else "optional")
            + " parameter schema; not an execution guarantee: " + quote(value)
            for name, value in properties.items()
        }
        self.expected_outcomes = _contract_expected_outcomes(response)
        self.failure_cues, self.recovery_hints = [], []
        self.metadata.pop("alias_of", None)
        self.metadata.pop("aliases", None)
        contract = {"tool_id": self.tool_id, "description": self.description,
                    "parameters": parameters, "response_schema": response}
        contract_hash = canonical_hash(contract)
        sources = {
            "when_to_use": {"description": self.description, "parameters": parameters},
            "preconditions": {"required": required}, "input_requirements": parameters,
            "expected_outcomes": response,
            **{name: {"no_claim": name, "contract_sha256": contract_hash}
               for name in ("failure_cues", "recovery_hints", "alias")},
        }
        values = {name: getattr(self, name) for name in sources if name != "alias"}
        values["alias"] = {"alias_of": None, "aliases": []}
        self.metadata["contract_initialization"] = {
            "policy": "contract_quotes", "scope": "declared_schema_not_execution_guarantee",
            "stage": "static_initialization_only",
            "execution_semantics_verified": False, "schema_vendor_consistency_verified": False,
            "contract_sha256": contract_hash, "response_schema_available": bool(response),
            "field_contracts": {name: {"source_sha256": canonical_hash(source),
                                      "scope": "declared_only" if name not in {"failure_cues", "recovery_hints", "alias"} else "no_claim"}
                                for name, source in sources.items()},
            "generated_fields_sha256": {name: canonical_hash(value) for name, value in values.items()},
            "expected_outcome_quotes_sha256": [canonical_hash(item) for item in self.expected_outcomes],
        }
        return self

    def init_from_llm(
        self,
        llm_client: Any,
        *,
        temperature: float = 0.2,
    ) -> "ToolSpec":
        """使用 LLM 初始化 when_to_use 和 expected_outcomes 字段。

        Args:
            llm_client: 实现了 chat(system_prompt, user_prompt) -> {"content": str} 的客户端。
            temperature: 生成温度。

        Returns:
            更新后的 ToolSpec（原地修改）。
        """
        prompt = _build_llm_init_prompt(self)
        result = llm_client.chat(
            system_prompt="You are an API documentation assistant.",
            user_prompt=prompt,
            temperature=temperature,
            schema_hash="tool-init-v3",
            sampling_profile="distillation",
        )
        when_to_use, expected_outcomes = _parse_llm_init_response(result["content"])
        when_to_use = _normalize_when_to_use(when_to_use)
        expected_outcomes = _normalize_expected_outcomes(expected_outcomes)
        issues = _profile_quality_issues(self, when_to_use, expected_outcomes) if when_to_use and expected_outcomes else ["missing required profile fields"]
        if (not when_to_use or not expected_outcomes or issues) and llm_client is not None:
            try:
                repaired_when_to_use, repaired_expected_outcomes = _repair_llm_init_response(
                    self,
                    llm_client,
                    raw_content=str(result.get("content", "")),
                    issues=issues,
                )
                when_to_use = _normalize_when_to_use(repaired_when_to_use or when_to_use)
                expected_outcomes = _normalize_expected_outcomes(repaired_expected_outcomes or expected_outcomes)
                issues = _profile_quality_issues(self, when_to_use, expected_outcomes) if when_to_use and expected_outcomes else ["missing required profile fields"]
            except Exception:
                pass
        if not when_to_use or not expected_outcomes or issues:
            fallback_when_to_use = _build_deterministic_when_to_use(self)
            fallback_expected_outcomes = _build_deterministic_expected_outcomes(self, fallback=expected_outcomes)
            when_to_use, expected_outcomes, issues = normalize_profile_fields(
                self,
                when_to_use=fallback_when_to_use,
                expected_outcomes=fallback_expected_outcomes,
            )
        final_when_to_use = self.when_to_use or when_to_use
        final_expected_outcomes = list(self.expected_outcomes) or expected_outcomes
        if not final_when_to_use or not final_expected_outcomes:
            raise ValueError("LLM init response missing required profile fields")
        if not self.when_to_use:
            self.when_to_use = final_when_to_use
        if not self.expected_outcomes:
            self.expected_outcomes = final_expected_outcomes
        return self


# ---- 辅助函数 ----

_RESOURCE_ID_SUFFIXES = {"_id", "_key", "_hash"}
_PROFILE_HALLUCINATION_PHRASES = (
    "already active",
    "already engaged",
    "already inactive",
    "system readiness",
    "readiness status",
)


def _is_resource_id_param(param_name: str) -> bool:
    """判断参数名是否为资源 ID 类参数（用于推导 preconditions）。"""
    lower = param_name.lower()
    # 精确匹配常见 ID 类参数
    _id_words = {"id", "user_id", "folder_id", "file_id", "doc_id",
                  "ticket_id", "order_id", "api_key", "token", "email"}
    if lower in _id_words:
        return True
    # 后缀匹配：xxx_id, xxx_key, xxx_hash
    for suffix in _RESOURCE_ID_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            return True
    return False


def _tool_description_core(description: str) -> str:
    text = str(description or "").strip()
    marker = "Tool description:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text.rstrip(".")


def _looks_like_literal_expected_outcome(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if text.startswith(("{", "[", "(")):
        return True
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        pass
    if re.search(r'["\'][A-Za-z0-9_ -]+["\']\s*:', text):
        return True
    compact = re.sub(r"\s+", "", text)
    if compact and sum(ch.isdigit() or ch in "{}[],:._-\"" for ch in compact) / len(compact) > 0.45:
        return True
    return False


def _response_schema(tool: ToolSpec) -> dict[str, Any]:
    metadata = tool.metadata if isinstance(tool.metadata, dict) else {}
    schema = metadata.get("response_schema", {})
    return schema if isinstance(schema, dict) else {}


def _contract_expected_outcomes(response: dict[str, Any]) -> list[str]:
    return [
        "Declared response documentation only; not a guarantee for every execution branch.",
        "Declared response schema quote: " + json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if response else "Declared response schema is unavailable; actual outputs and execution branches are unknown.",
    ]


def _format_or_list(items: list[str]) -> str:
    unique = [str(item).strip() for item in items if str(item).strip()]
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} or {unique[1]}"
    return f"{', '.join(unique[:-1])}, or {unique[-1]}"


def _extract_enum_options(param_def: dict[str, Any]) -> list[str]:
    explicit = param_def.get("enum", [])
    if isinstance(explicit, list) and explicit:
        return [str(item).strip() for item in explicit if str(item).strip()]
    description = str(param_def.get("description", "") or "")
    match = re.search(r"\[Enum\]\s*:\s*(\[[^\]]+\])", description, flags=re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1)
    candidates = [raw, raw.replace("'", '"')]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    return []


def _required_enum_map(tool: ToolSpec) -> dict[str, list[str]]:
    required = set(tool.parameter_schema.get("required", [])) if isinstance(tool.parameter_schema.get("required"), list) else set()
    properties = tool.parameter_schema.get("properties", {}) if isinstance(tool.parameter_schema.get("properties"), dict) else {}
    enum_map: dict[str, list[str]] = {}
    for param_name in sorted(required):
        param_def = properties.get(param_name, {})
        if not isinstance(param_def, dict):
            continue
        options = _extract_enum_options(param_def)
        if options:
            enum_map[str(param_name)] = options
    return enum_map


def _response_fields(tool: ToolSpec, *, max_fields: int = 4) -> list[tuple[str, str]]:
    schema = _response_schema(tool)
    properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    fields: list[tuple[str, str]] = []
    for field_name, field_def in properties.items():
        if len(fields) >= max_fields:
            break
        if not isinstance(field_def, dict):
            fields.append((str(field_name), ""))
            continue
        fields.append((str(field_name), str(field_def.get("description", "") or "").strip()))
    return fields


def _expected_field_coverage(expected_outcomes: list[str], response_fields: list[tuple[str, str]]) -> int:
    lowered_outcomes = [str(item).lower() for item in expected_outcomes]
    covered = 0
    for field_name, _ in response_fields:
        if any(field_name.lower() in item for item in lowered_outcomes):
            covered += 1
    return covered


def _missing_enum_coverage(tool: ToolSpec, when_to_use: str) -> dict[str, list[str]]:
    lowered = str(when_to_use or "").lower()
    missing: dict[str, list[str]] = {}
    for param_name, options in _required_enum_map(tool).items():
        missing_options = [option for option in options if option.lower() not in lowered]
        if missing_options:
            missing[param_name] = missing_options
    return missing


def _unsupported_profile_phrases(tool: ToolSpec, when_to_use: str, expected_outcomes: list[str]) -> list[str]:
    source_text = " ".join(
        [
            str(tool.description or ""),
            json.dumps(tool.parameter_schema, ensure_ascii=False, sort_keys=True),
            json.dumps(_response_schema(tool), ensure_ascii=False, sort_keys=True),
        ]
    ).lower()
    target_text = " ".join([str(when_to_use or "")] + [str(item) for item in expected_outcomes]).lower()
    issues: list[str] = []
    for phrase in _PROFILE_HALLUCINATION_PHRASES:
        if phrase in target_text and phrase not in source_text:
            issues.append(f"unsupported phrase: {phrase}")
    return issues


def _profile_quality_issues(tool: ToolSpec, when_to_use: str, expected_outcomes: list[str]) -> list[str]:
    issues: list[str] = []
    missing_enum = _missing_enum_coverage(tool, when_to_use)
    for param_name, options in missing_enum.items():
        issues.append(f"missing enum coverage for {param_name}: {', '.join(options)}")
    response_fields = _response_fields(tool)
    if response_fields:
        required_coverage = min(len(response_fields), 4)
        actual_coverage = _expected_field_coverage(expected_outcomes, response_fields)
        if actual_coverage < required_coverage:
            missing_names = [name for name, _ in response_fields if not any(name.lower() in item.lower() for item in expected_outcomes)]
            issues.append(f"missing response field coverage: {', '.join(missing_names)}")
    issues.extend(_unsupported_profile_phrases(tool, when_to_use, expected_outcomes))
    return issues


def _build_deterministic_when_to_use(tool: ToolSpec) -> str:
    core = _tool_description_core(tool.description) or tool.tool_name
    parts = [f"Use when this tool matches the task: {core}."]
    enum_phrases = []
    for param_name, options in _required_enum_map(tool).items():
        enum_phrases.append(f"{param_name} set to {_format_or_list(options)}")
    if enum_phrases:
        parts.append(f"Requires {'; '.join(enum_phrases)}.")
    else:
        required_parts = []
        properties = tool.parameter_schema.get("properties", {}) if isinstance(tool.parameter_schema.get("properties"), dict) else {}
        for param_name in tool.required_parameters:
            param_def = properties.get(param_name, {})
            param_type = str(param_def.get('type', 'any')) if isinstance(param_def, dict) else "any"
            required_parts.append(f"{param_name} ({param_type})")
        if required_parts:
            parts.append(f"Requires {_format_or_list(required_parts)}.")
    return " ".join(parts).strip()


def _build_deterministic_expected_outcomes(tool: ToolSpec, *, fallback: list[str] | None = None) -> list[str]:
    response_fields = _response_fields(tool)
    if response_fields:
        outcomes = []
        for field_name, description in response_fields:
            if description:
                short_desc = description.split(".")[0].strip()
                outcomes.append(f"Returns {field_name} ({short_desc})")
            else:
                outcomes.append(f"Returns {field_name}")
        return outcomes
    normalized_fallback = _normalize_expected_outcomes(fallback or [])
    if normalized_fallback:
        return normalized_fallback
    core = _tool_description_core(tool.description) or tool.tool_name
    return [f"Returns the result of: {core}"]


def normalize_profile_fields(
    tool: ToolSpec,
    *,
    when_to_use: str | None = None,
    expected_outcomes: list[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    normalized_when = _normalize_when_to_use(tool.when_to_use if when_to_use is None else when_to_use)
    normalized_outcomes = _normalize_expected_outcomes(tool.expected_outcomes if expected_outcomes is None else expected_outcomes)

    if not normalized_when or _missing_enum_coverage(tool, normalized_when) or _unsupported_profile_phrases(tool, normalized_when, normalized_outcomes):
        normalized_when = _build_deterministic_when_to_use(tool)
    response_fields = _response_fields(tool)
    if not normalized_outcomes or (response_fields and _expected_field_coverage(normalized_outcomes, response_fields) < min(len(response_fields), 4)):
        normalized_outcomes = _build_deterministic_expected_outcomes(tool, fallback=normalized_outcomes)

    issues = _profile_quality_issues(tool, normalized_when, normalized_outcomes)
    return normalized_when, normalized_outcomes, issues


def normalize_tool_profile_in_place(tool: ToolSpec) -> list[str]:
    when_to_use, expected_outcomes, issues = normalize_profile_fields(
        tool,
        when_to_use=tool.when_to_use,
        expected_outcomes=tool.expected_outcomes,
    )
    tool.when_to_use = when_to_use
    tool.expected_outcomes = expected_outcomes
    return issues


# ---- LLM 初始化 when_to_use 和 expected_outcomes ----

def _build_llm_init_prompt(tool: "ToolSpec") -> str:
    """构建 LLM 初始化提示词。"""
    param_desc = ""
    props = tool.parameter_schema.get("properties", {})
    required = set(tool.parameter_schema.get("required", []))
    if props:
        param_lines = []
        for name, defn in props.items():
            typ = defn.get("type", "any") if isinstance(defn, dict) else "any"
            tag = "required" if name in required else "optional"
            enum_options = _extract_enum_options(defn) if isinstance(defn, dict) else []
            if enum_options:
                param_lines.append(f"  - {name} ({tag}, {typ}; enum={enum_options})")
            else:
                param_lines.append(f"  - {name} ({tag}, {typ})")
        param_desc = "\n".join(param_lines)
    else:
        param_desc = "  (none)"
    response_fields = _response_fields(tool)
    if response_fields:
        response_desc = "\n".join(
            f"  - {field_name}: {description or 'response field'}"
            for field_name, description in response_fields
        )
    else:
        response_desc = "  (unknown)"

    return f"""\
Generate initialization guidance for the following API tool.

Tool: {tool.tool_name}
Description: {tool.description}
Parameters:
{param_desc}
Response fields:
{response_desc}

Output JSON with exactly these fields:
{{
  "when_to_use": "Briefly answer: what task stage is this tool for? what input completeness is needed? when should it NOT be used? Under 60 words.",
  "expected_outcomes": ["List 2-4 typical outputs or intermediate results this tool provides."]
}}

Return raw JSON only. Do not include markdown fences, analysis, or <think> tags.
Keep when_to_use under 60 words. expected_outcomes should have 2-4 items.
Mandatory rules:
- Cover every required enum option explicitly when the tool has a required enum parameter.
- Prefer response-field-grounded outcomes; mention concrete response fields instead of generic system status text.
- Do not invent unsupported state checks unless they are directly grounded in the schema or description."""


def _build_llm_init_repair_prompt(tool: "ToolSpec", raw_content: str, issues: list[str]) -> str:
    return json.dumps(
        {
            "tool_id": tool.tool_id,
            "tool_name": tool.tool_name,
            "description": tool.description,
            "required_enum_options": _required_enum_map(tool),
            "response_fields": [field_name for field_name, _ in _response_fields(tool)],
            "required_output_schema": {
                "when_to_use": "string",
                "expected_outcomes": ["string"],
            },
            "instruction": (
                "Convert the previous model answer into strict JSON with exactly the keys "
                "'when_to_use' and 'expected_outcomes'. Do not include analysis or thinking tags. "
                "Fix the concrete quality issues listed below instead of rewriting freely."
            ),
            "quality_issues": issues,
            "fallback_profile": {
                "when_to_use": _build_deterministic_when_to_use(tool),
                "expected_outcomes": _build_deterministic_expected_outcomes(tool),
            },
            "raw_model_answer": raw_content,
        },
        ensure_ascii=False,
    )


def _strip_reasoning_blocks(text: str) -> str:
    cleaned = re.sub(
        r"<(?:think|thinking)\b[^>]*>.*?</(?:think|thinking)>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"</?(?:think|thinking)\b[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _iter_json_segments(text: str) -> list[str]:
    segments: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escaped = False

    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            if not stack:
                start = idx
            stack.append("}" if char == "{" else "]")
            continue

        if char in "}]":
            if not stack or char != stack[-1]:
                stack.clear()
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                segments.append(text[start : idx + 1])
                start = None

    return segments


def _normalize_when_to_use(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _normalize_expected_outcomes(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    elif isinstance(value, str):
        raw_items = [segment.strip(" -*\t") for segment in value.splitlines()]
    else:
        raw_items = []
    seen: set[str] = set()
    outcomes: list[str] = []
    for item in raw_items:
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        outcomes.append(item)
    return outcomes


def _try_load_init_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    repaired = re.sub(r'\\(?![ntrf"\\/b])', r"\\\\", text)
    if repaired != text:
        candidates.append(repaired)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _init_payload_priority(payload: dict[str, Any]) -> tuple[int, int, int]:
    when_to_use = _normalize_when_to_use(payload.get("when_to_use", ""))
    expected_outcomes = _normalize_expected_outcomes(payload.get("expected_outcomes", []))
    expected_keys = {"when_to_use", "expected_outcomes"}
    return (
        sum(key in payload for key in expected_keys),
        int(bool(when_to_use)) + int(bool(expected_outcomes)),
        len(expected_outcomes),
    )


def _parse_llm_init_response(content: str) -> tuple[str, list[str]]:
    """解析 LLM 返回的 when_to_use 和 expected_outcomes。"""
    text = content.strip()
    candidates: list[dict[str, Any]] = []
    seen_segments: set[str] = set()
    candidate_texts = [text]

    stripped = _strip_reasoning_blocks(text)
    if stripped and stripped != text:
        candidate_texts.append(stripped)

    for candidate_text in candidate_texts:
        direct = _try_load_init_payload(candidate_text)
        if direct is not None:
            candidates.append(direct)

        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?\s*```", candidate_text, re.DOTALL | re.IGNORECASE):
            fence_text = match.group(1).strip()
            direct_fence = _try_load_init_payload(fence_text)
            if direct_fence is not None:
                candidates.append(direct_fence)
            for segment in _iter_json_segments(fence_text):
                if segment in seen_segments:
                    continue
                seen_segments.add(segment)
                payload = _try_load_init_payload(segment)
                if payload is not None:
                    candidates.append(payload)

        for segment in _iter_json_segments(candidate_text):
            if segment in seen_segments:
                continue
            seen_segments.add(segment)
            payload = _try_load_init_payload(segment)
            if payload is not None:
                candidates.append(payload)

    if not candidates:
        return "", []

    payload = max(candidates, key=_init_payload_priority)
    return (
        _normalize_when_to_use(payload.get("when_to_use", "")),
        _normalize_expected_outcomes(payload.get("expected_outcomes", [])),
    )


def _repair_llm_init_response(
    tool: "ToolSpec",
    llm_client: Any,
    *,
    raw_content: str,
    issues: list[str],
) -> tuple[str, list[str]]:
    repair_prompt = _build_llm_init_repair_prompt(tool, raw_content, issues)
    repaired = llm_client.chat(
        system_prompt="You are a strict JSON formatter for API tool profiles.",
        user_prompt=repair_prompt,
        temperature=0.0,
        schema_hash="tool-init-repair-v2",
        sampling_profile="judge",
    )
    return _parse_llm_init_response(str(repaired.get("content", "")))


def init_tool_profiles_from_llm(
    registry: "ToolRegistry",
    llm_client: Any,
    *,
    tool_ids: set[str] | None = None,
    temperature: float = 0.2,
    skip_existing: bool = True,
    max_workers: int = 1,
) -> dict[str, bool]:
    """批量使用 LLM 初始化 registry 中工具的 when_to_use 和 expected_outcomes。

    Args:
        registry: 工具注册表。
        llm_client: 实现了 chat(system, user) -> {"content": str} 的客户端。
        tool_ids: 仅初始化指定工具，None 表示全部。
        temperature: 生成温度。
        skip_existing: 仅当 when_to_use 和 expected_outcomes 都已存在时跳过。

    Returns:
        {tool_id: success} 字典，记录每个工具的初始化结果。
    """
    target_ids = tool_ids or {t.tool_id for t in registry.all()}
    results: dict[str, bool] = {}

    def _clone_llm_client(base_client: Any) -> Any:
        client_config = getattr(base_client, "config", None)
        client_type = getattr(base_client, "__class__", None)
        if client_config is not None and client_type is not None:
            try:
                return client_type(client_config)
            except Exception:
                return base_client
        return base_client

    def _init_one(tool_id: str) -> tuple[str, bool]:
        try:
            tool = registry.get(tool_id)
            if skip_existing and tool.when_to_use and tool.expected_outcomes:
                return tool_id, True
            worker_client = _clone_llm_client(llm_client)
            tool.init_from_llm(worker_client, temperature=temperature)
            return tool_id, bool(tool.when_to_use and tool.expected_outcomes)
        except Exception:  # noqa: BLE001
            return tool_id, False

    ordered_ids = sorted(target_ids)
    worker_count = max(1, min(int(max_workers or 1), len(ordered_ids) or 1))

    if worker_count <= 1:
        for tool_id in ordered_ids:
            key, success = _init_one(tool_id)
            results[key] = success
        return results

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_init_one, tool_id): tool_id for tool_id in ordered_ids}
        for future in as_completed(futures):
            key, success = future.result()
            results[key] = success

    return results
