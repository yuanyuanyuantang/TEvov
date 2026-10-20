"""Procedural Memory Bank — 统一记忆记录。

设计原则（2026-09 重构）：
- 记忆内容用自然语言（when_to_use 一行 + content markdown 正文），不再按类型
  维护 7 套字段 schema —— 小模型（Qwen3-8B）写入/读取自然语言都更可靠。
- 保留一层"薄元数据信封"（id / type_tag / tool_tags / confidence / 计数器），
  支撑归因（credited/blamed）、去重、生命周期管理与按 tag 的分析/配额。
- 原 7 类记忆降级为 type_tag 标签，仅用于分析与检索过滤，不再决定字段结构。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# 7 个类型 tag（原 7 类结构化记忆降级而来，仅用于分析/过滤/配额）
MEMORY_TAGS: tuple[str, ...] = (
    "success_trajectory",     # 某类任务验证有效的工具序列/路径
    "failure_repair",         # 具体报错 → 修复步骤
    "parameter_template",     # 参数值来源（哪个工具输出字段/任务文本字段）
    "tool_switching",         # 什么条件下从工具 A 换到工具 B
    "state_tool_preference",  # 某状态/阶段下优先用哪个工具
    "tool_distractor",        # 易混工具对 + 区分要点（负面提醒）
    "generic_rule",           # 其余通用经验（默认兜底）
)

DEFAULT_TAG = "generic_rule"

# 记忆来源标签
SOURCE_LLM_EXTRACT = "llm_extract"
SOURCE_COMPARATIVE = "comparative"
SOURCE_SUMMARIZE_MERGE = "summarize_merge"
SOURCE_MANUAL = "manual"

# confidence 初值（代码定，LLM 不自评 —— 8B 自评噪声大）：
# 成功任务提取 0.6，失败任务提取（修复类经验）0.5；归因证据驱动后续增减。
CONFIDENCE_FROM_SUCCESS = 0.6
CONFIDENCE_FROM_FAILURE = 0.5
CONFIDENCE_INCREMENT = 0.05
CONFIDENCE_PENALTY = 0.1
STALE_THRESHOLD = 0.2


def normalize_tag(value: Any) -> str:
    """把任意 tag 输入规范化到 7 个合法 tag，非法值落到 generic_rule。"""
    raw = str(value or "").strip()
    text = raw.lower()
    if text in MEMORY_TAGS:
        return text
    # 兼容旧类名（如 "FailureRepairRecord" → "failure_repair"）
    camel = raw.replace("Record", "").replace("record", "")
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", camel).lower().strip("_")
    if snake in MEMORY_TAGS:
        return snake
    return DEFAULT_TAG


@dataclass(slots=True)
class MemoryRecord:
    """统一程序性记忆记录 = 薄信封 + 自然语言内容。"""

    # ── 信封（结构化；归因/去重/生命周期依赖）──
    record_id: str
    type_tag: str = DEFAULT_TAG            # MEMORY_TAGS 之一
    tool_tags: list[str] = field(default_factory=list)  # 提及的工具 id，归因匹配挂钩
    confidence: float = CONFIDENCE_FROM_SUCCESS         # [0,1]，credited/blamed 增减

    # ── 内容（自然语言，主体）──
    when_to_use: str = ""                  # 泛化场景句（≤240 字符一行），检索主信号
    content: str = ""                      # markdown 正文（≤80 词）：具体、祈使、自包含

    # ── 统计与溯源（代码维护，LLM 不生成）──
    usage_count: int = 0                   # 被检索展示次数（exposure）
    success_count: int = 0                 # 归因命中且任务成功次数
    source: str = ""                       # llm_extract | comparative | summarize_merge | manual
    source_task: str = ""                  # 产生该记忆的 task_id
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stale: bool = False                    # confidence < STALE_THRESHOLD 时置位，检索排除
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type_tag = normalize_tag(self.type_tag)
        self.tool_tags = [
            str(t).strip() for t in (self.tool_tags or []) if str(t).strip()
        ]

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------

    @property
    def search_text(self) -> str:
        """embedding / 去重用的检索文本 = 仅 when_to_use 场景句（ReMe 对齐）。

        content 是策略正文，不进入检索键：键保持短场景句，查询（用户目标）
        与键在"意图空间"匹配，而不是被正文样板或任务实体稀释。
        """
        return self.when_to_use.strip()

    def to_markdown(self) -> str:
        """注入 prompt 用的 markdown 渲染（含 [tag | tools] 标头）。"""
        header = self.type_tag
        if self.tool_tags:
            header += f" | tools: {', '.join(self.tool_tags[:6])}"
        lines = [f"[{header}]"]
        if self.when_to_use:
            lines.append(f"When to use: {self.when_to_use.strip()}")
        if self.content:
            lines.append(self.content.strip())
        return "\n".join(lines)
