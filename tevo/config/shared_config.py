"""实验配置加载与覆盖逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LoggingConfig:
    """日志配置。"""

    level: str = "INFO"
    run_name: str = "mctpo-demo"


@dataclass(slots=True)
class MemoryEmbeddingConfig:
    """记忆检索使用的 embedding 配置。"""

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "MCTPO_MEMORY_EMBEDDING_API_KEY"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    ca_bundle: str = ""
    insecure_skip_verify: bool = False
    batch_size: int = 32
    dimensions: int = 0
    encoding_format: str = "float"

    def resolved_api_key(self) -> str:
        """优先返回显式 key，否则回退到环境变量。"""

        if self.api_key:
            return self.api_key
        primary = os.getenv(self.api_key_env, "")
        if primary:
            return primary
        for fallback_env in [
            "EMBEDDING_API_KEY",
            "MCTPO_MEMORY_EMBEDDING_API_KEY",
            "LLM_API_KEY",
            "MCTPO_LLM_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
        ]:
            fallback = os.getenv(fallback_env, "")
            if fallback:
                return fallback
        return ""

    def resolved_ca_bundle(self) -> str:
        """解析自定义 CA bundle 路径。"""

        return (
            self.ca_bundle
            or os.getenv("MCTPO_CA_BUNDLE", "")
            or os.getenv("REQUESTS_CA_BUNDLE", "")
            or os.getenv("SSL_CERT_FILE", "")
        )

    def resolved_insecure_skip_verify(self) -> bool:
        """解析是否跳过 TLS 校验。"""

        if self.insecure_skip_verify:
            return True
        flag = os.getenv("MCTPO_INSECURE_SKIP_VERIFY", "")
        return flag.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class MemoryConfig:
    """记忆系统配置。"""

    top_k_episodic: int = 3
    top_k_skill: int = 2
    prune_threshold: float = 0.05
    bootstrap_skills: bool = True
    embedding: MemoryEmbeddingConfig = field(default_factory=MemoryEmbeddingConfig)


@dataclass(slots=True)
class RolloutConfig:
    """rollout 配置。"""

    num_trajectories: int = 4
    max_steps: int = 6
    branch_factor: int = 2
    search_depth: int = 2
    retrieval_cost_penalty: float = 0.2
    budget_floor: float = 1.0
    exploration_mode: str = "ucb"
    ucb_scale: float = 0.6
    thompson_std: float = 0.15
    token_budget: float = 120.0
    latency_budget: float = 6.0
    monetary_budget: float = 0.25
    token_weight: float = 0.2
    latency_weight: float = 0.15
    monetary_weight: float = 0.25


@dataclass(slots=True)
class TrainConfig:
    """训练配置。"""

    epochs: int = 1
    seed: int = 7
    checkpoint_dir: str = "outputs/tevo/checkpoints"
    preference_epochs: int = 30
    preference_lr: float = 0.01
    bandit_decay: float = 0.8
    preference_hidden_dim: int = 16
    preference_loss: str = "dpo"
    preference_beta: float = 1.0
    replay_buffer_capacity: int = 512
    replay_batch_size: int = 32
    replay_priority_alpha: float = 1.0
    replay_importance_beta: float = 0.4
    replay_source_quota: dict[str, float] = field(
        default_factory=lambda: {
            "trajectory_preference": 0.4,
            "branch_preference": 0.35,
            "retrieval_contrast": 0.25,
        }
    )
    dynamic_source_quota: bool = True
    source_quota_adaptation_rate: float = 0.2
    source_quota_floor: float = 0.15
    source_quota_max_delta: float = 0.08
    source_quota_inertia: float = 0.7
    roi_target: float = 1.0
    roi_tuning_rate: float = 0.1
    multi_seeds: list[int] = field(default_factory=lambda: [7, 11, 23])
    benchmark_protocol_name: str = "mctpo-benchmark-v1"


@dataclass(slots=True)
class LLMConfig:
    """LLM 与 planner/critic 配置。"""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8010/v1"
    model: str = "Qwen/Qwen3-8B"
    api_key: str = ""
    api_key_env: str = "MCTPO_LLM_API_KEY"
    timeout_seconds: float = 20.0
    seed: int | None = None
    max_tokens: int = 4096
    context_budget: dict[str, Any] | None = None
    planner_temperature: float = 0.2
    critic_temperature: float = 0.0
    cache_enabled: bool = True
    cache_dir: str = "outputs/tevo/cache"
    cache_max_entries: int = 1000
    cache_ttl_seconds: float = 86400.0
    cache_version: str = "v1"
    planner_schema_version: str = "planner-v1"
    critic_schema_version: str = "critic-v1"
    planner_mode: str = "hybrid"
    critic_mode: str = "hybrid"
    ablation_modes: list[str] = field(default_factory=lambda: ["heuristic-only", "llm-only", "hybrid"])
    compare_models: list[str] = field(default_factory=lambda: ["deepseek-chat"])
    model_costs: dict[str, float] = field(default_factory=lambda: {"deepseek-chat": 1.0})
    min_request_interval_seconds: float = 0.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    insecure_skip_verify: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    sampling_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from tevo.llm.context_budget import validate_context_budget
        self.context_budget = validate_context_budget(self.context_budget)
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)
                                      or not 0 <= self.seed < 2**32):
            raise ValueError("LLM seed must be an integer in [0, 2**32)")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("LLM max_tokens must be a positive integer")
        if (isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int)
            or self.max_retries < 0):
            raise ValueError("LLM max_retries must be a non-negative integer")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value)
               for value in (self.timeout_seconds, self.retry_backoff_seconds)):
            raise ValueError("LLM timeout/retry configuration must be finite and non-negative")
        if self.timeout_seconds <= 0 or self.retry_backoff_seconds < 0:
            raise ValueError("LLM timeout/retry configuration must be finite and non-negative")

    def resolved_api_key(self) -> str:
        """优先返回显式 key，否则回退到环境变量。"""

        if self.api_key:
            return self.api_key
        primary = os.getenv(self.api_key_env, "")
        if primary:
            return primary
        for fallback_env in ["LLM_API_KEY", "MCTPO_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]:
            fallback = os.getenv(fallback_env, "")
            if fallback:
                return fallback
        return ""


@dataclass(slots=True)
class OutputConfig:
    """输出配置。"""

    base_dir: str = "outputs"
    export_parquet: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    """MCTPO 总配置。"""

    project_name: str = "mctpo"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。"""

        return asdict(self)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """递归更新字典。"""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_dot_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """应用点号路径覆盖。"""

    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        current = payload
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    return payload


def _build_config(payload: dict[str, Any]) -> ExperimentConfig:
    """从原始字典构建配置对象。"""

    logging = LoggingConfig(**payload.get("logging", {}))
    memory_payload = dict(payload.get("memory", {}))
    embedding = MemoryEmbeddingConfig(**memory_payload.pop("embedding", {}))
    memory = MemoryConfig(**memory_payload, embedding=embedding)
    rollout = RolloutConfig(**payload.get("rollout", {}))
    train = TrainConfig(**payload.get("train", {}))
    llm = LLMConfig(**payload.get("llm", {}))
    output = OutputConfig(**payload.get("output", {}))
    return ExperimentConfig(
        project_name=payload.get("project_name", "mctpo"),
        logging=logging,
        memory=memory,
        rollout=rollout,
        train=train,
        llm=llm,
        output=output,
    )


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """加载 YAML 配置并应用覆盖。

    该实现综合了 ReMe 的分层配置覆盖思想与 MetaClaw 的 dataclass 配置风格，
    但刻意保持为轻量、可测试的研究原型版本。
    """

    config_path = Path(path)
    payload = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if overrides:
        payload = _apply_dot_overrides(payload, overrides)
    return _build_config(payload)
