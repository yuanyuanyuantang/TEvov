# flake8: noqa: E402
# pylint: disable=too-many-return-statements
"""用于 BFCL-v3（多轮）任务的最小化 ReAct Agent。"""

import re
import os
import sys
import time
import json
import warnings
import tempfile
import datetime
import gc
from pathlib import Path
from typing import Dict, List, Any

try:
    import ray
except ImportError:
    ray = None
import requests
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from copy import deepcopy
from uuid import uuid4

# 工具画像注入（纯函数，仅依赖标准库）：把 Tool Profile Pool 的演化字段渲染进
# LLM 可见的工具 description。bfcl_agent 在 ray worker 里运行，仓库根不一定在
# sys.path 上，因此显式补齐后再导入。
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tevo.tool_pool.profile_injection import (  # noqa: E402
    ALL_PROFILE_FIELDS,
    DEFAULT_INJECT_FIELDS,
    DEFAULT_MAX_CHARS,
    apply_tool_profiles,
)
from tevo.reproducibility import canonical_hash, derive_seed
from tevo.llm.context_budget import (apply_context_budget, request_tokenization,
                                    validate_context_budget, verify_prompt_count)
from tevo.state.param_binding import tool_output_is_error

from bfcl_utils import (
    load_test_case,
    handle_user_turn,
    handle_tool_calls,
    compile_tools,
    extract_tool_schema,
    extract_single_turn_response,
    extract_multi_turn_responses,
    capture_and_print_score_files,
    create_error_response,
)
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
    reset_multi_turn_func_call_instances,
)
from bfcl_eval.eval_checker.eval_runner import (
    multi_turn_runner,
    ast_file_runner,
)
from bfcl_eval.utils import (
    is_multi_turn,
    is_relevance_or_irrelevance,
    find_file_with_suffix,
    load_file,
)

os.environ.setdefault("BFCL_DATA_PATH", str(Path(__file__).parent / "data" / "multiturn_data_base_val.jsonl"))
os.environ.setdefault("BFCL_ANSWER_PATH", str(Path(__file__).parent / "data" / "possible_answer"))
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("QWEN_API_KEY", "dummy")
load_dotenv(Path(_REPO_ROOT) / ".env")


def _env_bool(name: str, *, default: bool = False) -> bool:
    """读取 1/true/yes/on 风格的布尔环境变量；未设置时返回 default。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning(f"Ignoring invalid {name}={raw!r}; using default {default}")
    return default


def _resolve_profile_fields() -> tuple[str, ...]:
    """TEVO_INJECT_TOOL_PROFILE_FIELDS=when_to_use,failure_cues → 字段子集（字段级消融用）。

    默认用 DEFAULT_INJECT_FIELDS（不含与 tool schema 重复的 input_requirements）。
    """
    raw = os.environ.get("TEVO_INJECT_TOOL_PROFILE_FIELDS", "").strip()
    if not raw:
        return DEFAULT_INJECT_FIELDS
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    selected = tuple(field for field in ALL_PROFILE_FIELDS if field in wanted)
    if not selected:
        logger.warning(f"No valid fields in TEVO_INJECT_TOOL_PROFILE_FIELDS={raw!r}; using defaults")
        return DEFAULT_INJECT_FIELDS
    return selected


def _resolve_profile_max_chars() -> int:
    """每个工具注入段落的字符上限（TEVO_TOOL_PROFILE_MAX_CHARS 覆盖）。"""
    raw = os.environ.get("TEVO_TOOL_PROFILE_MAX_CHARS", "").strip()
    if not raw:
        return DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Ignoring invalid TEVO_TOOL_PROFILE_MAX_CHARS={raw!r}")
        return DEFAULT_MAX_CHARS
    return value if value > 0 else DEFAULT_MAX_CHARS


class LocalBFCLAgent:
    """用于 BFCL-v3（多轮）任务的最小化 ReAct Agent。"""

    def __init__(
        self,
        index: int,
        task_ids: List[str],
        experiment_name: str,
        data_path: str = os.getenv("BFCL_DATA_PATH"),
        answer_path: Path = Path(os.getenv("BFCL_ANSWER_PATH")),
        model_name: str = "qwen3-8b",
        model_path: str = "Qwen/Qwen3-8B",
        temperature: float = 0.6,
        max_interactions: int = 30,
        max_response_size: int = 2000,
        num_trials: int = 4,
        max_retries: int = 99,
        retry_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 600.0,
        enable_thinking: bool = False,
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        use_tevo_observer: bool = False,
        use_tool_profiles: bool = False,
        tool_scope: str = "task",
        delete_freq: int = 10,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
        trial_seeds: list[int] | None = None,
        max_tokens: int | None = None,
        max_workers: int = 1,
        context_budget: dict | None = None,
        per_turn_memory: bool = False,
    ):

        self.index: int = index
        self.task_ids: List[str] = task_ids
        self.categories: List[str] = [task_id.rsplit("_", 1)[0] if "_" in task_id else task_id for task_id in task_ids]
        self.experiment_name: str = experiment_name
        self.data_path: str = data_path
        self.answer_path: Path = answer_path
        # Use model_path as the API model name when model_name is a short
        # name that vLLM (running without --served-model-name) won't recognise.
        _api_model_name = model_name
        if model_path and "/" in model_path and "/" not in model_name:
            _api_model_name = model_path
        self.model_name: str = _api_model_name
        self.model_path: str = model_path
        self.temperature: float = temperature
        self.max_interactions: int = max_interactions
        self.max_response_size: int = max_response_size
        self.num_trials: int = num_trials
        self.max_retries: int = max_retries
        self.retry_backoff_seconds: float = retry_backoff_seconds
        self.request_timeout_seconds: float = request_timeout_seconds
        if request_timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("request_timeout_seconds must be positive and max_retries nonnegative")
        if num_trials < 1 or max_workers < 1:
            raise ValueError("num_trials and max_workers must be positive")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer or None")
        if trial_seeds is not None:
            if len(trial_seeds) != num_trials or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in trial_seeds):
                raise ValueError("trial_seeds must contain one nonnegative integer per trial")
            if len(set(trial_seeds)) != len(trial_seeds):
                raise ValueError("Explicit trial_seeds must be distinct")
        self.trial_seeds = list(trial_seeds) if trial_seeds is not None else None
        self.max_tokens = max_tokens
        self.context_budget = validate_context_budget(context_budget)
        if self.context_budget is not None and max_tokens is None:
            raise ValueError("Context admission requires explicit max_tokens")
        self.max_workers = max_workers
        self.request_metrics: list[dict[str, Any]] = []
        self.context_budget_metrics: list[dict[str, Any]] = []
        self._active_run_id = 0
        self._active_task_index = 0
        self._active_episode_id = ""
        self._evaluation_error: dict[str, str] | None = None
        self._trial_seed()  # Validate a legacy environment override before starting any requests.
        self.enable_thinking: bool = enable_thinking
        self.use_memory: bool = use_memory
        # 逐用户轮检索（两臂消融开关）：每来一个新的用户轮，用当前用户原文前缀
        # U_0..U_t 重查冻结记忆并把第一消息里的记忆块整体替换（不累积）。
        if per_turn_memory and not use_memory:
            raise ValueError("per_turn_memory requires use_memory")
        self.per_turn_memory: bool = bool(per_turn_memory)
        self.use_memory_addition: bool = use_memory_addition if (use_memory or use_tevo_observer) else False
        self.use_memory_deletion: bool = use_memory_deletion if use_memory else False
        self.use_tevo_observer: bool = use_tevo_observer
        # 工具画像注入：env TEVO_INJECT_TOOL_PROFILES 优先于构造参数（消融开关）。
        # 与 use_memory 解耦，可单独做 "只画像不记忆" 消融。
        self.use_tool_profiles: bool = _env_bool(
            "TEVO_INJECT_TOOL_PROFILES", default=bool(use_tool_profiles)
        )
        self.tool_profile_cache: dict[str, dict[str, Any]] = {}
        self.tool_profile_fields: tuple[str, ...] = _resolve_profile_fields()
        self.tool_profile_max_chars: int = _resolve_profile_max_chars()
        self.tool_scope: str = tool_scope
        self.delete_freq: int = delete_freq
        self.freq_threshold: int = freq_threshold
        self.utility_threshold: float = utility_threshold
        self.memory_base_url: str = memory_base_url
        self._logged_val_observer_readonly: bool = False
        self._logged_val_observer_deferred: bool = False
        self._logged_train_observer_deferred: bool = False
        self._logged_train_memory_read_disabled: bool = False
        self.dynamic_context_refresh_count: int = 0
        if self._is_train_observer_run() and self._tevo_allow_learning_for_current_run() and self.max_workers > 1:
            raise ValueError("Train acquisition learning requires max_workers=1; actor-local deferral is not a global worker barrier")
        self.compact_results: bool = os.environ.get("TEVO_COMPACT_BFCL_RESULTS", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }

        if self.tool_scope not in {"task", "all"}:
            raise ValueError(f"Unsupported tool_scope: {self.tool_scope}")

        self.llm_base_url: str = os.environ.get("LLM_BASE_URL", "")
        self.thinking_extra_body_mode: str = os.environ.get(
            "TEVO_LLM_THINKING_EXTRA_BODY", "auto"
        ).strip().lower()
        self.llm_client = OpenAI(
            api_key=os.environ.get("LLM_API_KEY", "dummy"),
            base_url=self.llm_base_url,
            timeout=self.request_timeout_seconds,
            max_retries=0,  # Actor retries are logged; hidden SDK retries would evade the request budget.
        )

        self.history: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.retrieved_memory_list: List[List[List[Any]]] = [[] for _ in range(num_trials)]
        self.test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.original_test_entry: List[List[Dict[str, Any]]] = [[] for _ in range(num_trials)]
        self.tool_schema: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        # 注入画像后的 schema（base 仍是 self.tool_schema，注入版每次任务开始时重建）
        self.injected_tool_schema: List[List[List[dict]]] = [[] for _ in range(num_trials)]
        self.current_turn = [[0 for _ in range(len(task_ids))] for _ in range(num_trials)]
        # 逐轮检索状态：env 送达的用户轮原文、已注入记忆块对应的轮号（-1=无）、
        # 注入日志（每次注入一条，供运行后验证单块替换与查询一致性）。
        self.user_turn_texts: List[List[List[str]]] = [[[] for _ in range(len(task_ids))] for _ in range(num_trials)]
        self.memory_refreshed_turn: List[List[int]] = [[-1 for _ in range(len(task_ids))] for _ in range(num_trials)]
        self.memory_injection_log: List[List[List[dict]]] = [[[] for _ in range(len(task_ids))] for _ in range(num_trials)]
        self.global_tool_schema: List[dict] | None = None
        if self.tool_scope == "all":
            self.global_tool_schema = self._load_global_tool_schema()
            logger.info(f"BFCL all-tools mode enabled: {len(self.global_tool_schema)} tools exposed to LLM")

        for run_id in range(num_trials):
            for task_index in range(len(task_ids)):
                self.init_state(run_id, task_index)

    def init_state(self, run_id, i) -> Dict[str, Any]:
        """初始化智能体状态。"""
        self.test_entry[run_id].append(load_test_case(self.data_path, self.task_ids[i]))
        self.original_test_entry[run_id].append(self.test_entry[run_id][i].get("extra", {}))
        task_tool_schema = extract_tool_schema(deepcopy(self.test_entry[run_id][i].get("tools", [{}])))
        self.tool_schema[run_id].append(self._visible_tool_schema(task_tool_schema))
        self.injected_tool_schema[run_id].append([])

        msg = self.test_entry[run_id][i].get("messages", [])
        self.history[run_id].append(msg)
        self.retrieved_memory_list[run_id].append([])
        self.current_turn[run_id][i] = 1
        self.user_turn_texts[run_id][i] = []
        self.memory_injection_log[run_id][i] = []
        if self.per_turn_memory:
            # The per-turn query rule starts from the verbatim first user turn;
            # a task without one cannot be served by the frozen per-turn contexts.
            first = msg[0] if msg else None
            if not isinstance(first, dict) or first.get("role") != "user" \
                    or not isinstance(first.get("content"), str) or not first["content"].strip():
                raise ValueError(f"per-turn memory requires a verbatim first user message: {self.task_ids[i]}")
            self.user_turn_texts[run_id][i].append(first["content"])

    def _bfcl_env_model_name(self, run_id: int) -> str:
        return f"env_handler_actor{self.index}_run{run_id}"

    def _bfcl_eval_model_name(self, run_id: int) -> str:
        return f"env_handler_eval_actor{self.index}_run{run_id}"

    def _reset_bfcl_instance_cache(
        self,
        run_id: int,
        task_index: int,
        *,
        include_env: bool = True,
        include_eval: bool = True,
    ) -> None:
        try:
            test_entry = self.original_test_entry[run_id][task_index]
        except (IndexError, TypeError):
            return
        if not isinstance(test_entry, dict):
            return
        test_entry_id = str(test_entry.get("id") or self.task_ids[task_index])
        involved_classes = test_entry.get("involved_classes", [])
        if not isinstance(involved_classes, list):
            involved_classes = []

        if include_env:
            reset_multi_turn_func_call_instances(
                model_name=self._bfcl_env_model_name(run_id),
                test_entry_id=test_entry_id,
                involved_classes=involved_classes,
                is_evaL_run=False,
            )
        if include_eval:
            eval_model_name = self._bfcl_eval_model_name(run_id)
            for model_name in (eval_model_name, f"{eval_model_name}_ground_truth"):
                reset_multi_turn_func_call_instances(
                    model_name=model_name,
                    test_entry_id=test_entry_id,
                    involved_classes=involved_classes,
                    is_evaL_run=True,
                )

    def _visible_tool_schema(self, task_tool_schema: list[dict]) -> list[dict]:
        """返回当前实验模式下给 LLM 可见的工具列表。"""
        if self.tool_scope == "all":
            return self.global_tool_schema or []
        return task_tool_schema

    def _load_global_tool_schema(self) -> list[dict]:
        """从 BFCL 多轮函数文档加载全局工具池。"""
        func_doc_dir = Path(self.data_path).resolve().parent / "multi_turn_func_doc"
        if not func_doc_dir.exists():
            raise FileNotFoundError(f"BFCL function doc directory not found: {func_doc_dir}")

        functions: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for file_path in sorted(func_doc_dir.glob("*.json")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    function = json.loads(line)
                    name = str(function.get("name", "")).strip()
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    functions.append(function)

        if not functions:
            raise ValueError(f"No BFCL function docs loaded from {func_doc_dir}")

        tools = compile_tools({"id": "multi_turn_base_all_tools", "function": functions})
        return extract_tool_schema(tools)

    def get_query_without_experience(self, query: str):
        """获取不带经验的查询文本。"""
        if "\n\nTEvo External Memory Context" in query:
            query = query.split("\n\nTEvo External Memory Context")[0].split("Task:\n")[-1]
        if "\n\nSome Related Experience" in query:
            query = query.split("\n\nSome Related Experience")[0].split("Task:\n")[-1]
        return query

    def get_traj_from_task_history(self, task_id: str, task_history: list, reward: float):
        """从任务历史中构建轨迹。"""
        return {
            "task_id": task_id,
            "messages": task_history,
            "score": reward,
        }

    def handle_api_response(self, response: requests.Response):
        """处理 API 响应并进行适当的错误检查。"""
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

        return response.json()

    def _merge_retrieved_memory_list(self, run_id: int, task_index: int, memory_list: list[Any]) -> None:
        if not memory_list:
            return
        existing = self.retrieved_memory_list[run_id][task_index]
        merged: list[Any] = []
        seen: set[str] = set()
        for memory in list(existing or []) + list(memory_list or []):
            if not isinstance(memory, dict):
                continue
            record_id = str(memory.get("id", "")).strip()
            key = record_id or json.dumps(memory, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            merged.append(memory)
        self.retrieved_memory_list[run_id][task_index] = merged

    def _retrieve_tevo_memory_context(self, run_id: int, task_index: int, query: str | None = None) -> str | None:
        """Retrieve TEvo memory context as compact text for injection into the user message.

        ReMe-style: return text only (no system role wrapper), to be appended
        to the first user message.  This avoids inserting system-role messages
        mid-conversation and keeps the prompt closer to the ReMe format.

        ``query`` is supplied by the per-turn refresh hook (the frozen
        cumulative user-text prefix U_0..U_t); a non-None query marks a refresh,
        so the retrieved list REPLACES the previous turn's list instead of
        merging into it.  Without a query the historical first-turn behaviour
        (latest user goal, merge) is unchanged.
        """
        if self._disable_train_memory_read_during_sampling():
            return None
        refreshing = query is not None
        if query is None:
            query = self._latest_user_goal(run_id, task_index)
        # Use the ReMe-compatible simple retrieval endpoint — pure embedding
        # top-K without state-aware filtering, quotas, or compatibility checks.
        # This matches how ReMe retrieves memories in its BFCL agent.
        try:
            response = requests.post(
                url=f"{self.memory_base_url}retrieve_task_memory",
                json={
                    "query": query,
                    "top_k": int(os.environ.get("TEVO_TOP_K_MEMORIES", "5")),
                },
                timeout=max(self.request_timeout_seconds, 120.0),
            )
        except requests.RequestException as exc:
            logger.warning(f"TEvo retrieve_task_memory request failed: {exc}")
            return None

        context_response = self.handle_api_response(response)
        if not context_response or context_response.get("success") is False:
            logger.warning(f"TEvo retrieve_task_memory failed: {context_response}")
            return None

        metadata = context_response.get("metadata", {}) if isinstance(context_response, dict) else {}
        memory_list = metadata.get("memory_list", []) if isinstance(metadata, dict) else []
        if isinstance(memory_list, list):
            if refreshing:
                # 逐轮刷新：当前轮的召回整体替换上一轮，不跨轮累积。
                self.retrieved_memory_list[run_id][task_index] = list(memory_list)
            else:
                self._merge_retrieved_memory_list(run_id, task_index, memory_list)

        self.dynamic_context_refresh_count += 1
        memory_ids = [m.get("id", "") for m in memory_list] if isinstance(memory_list, list) else []
        memory_types = [m.get("tevo_type", "") for m in memory_list] if isinstance(memory_list, list) else []
        logger.info(
            "tevo_dynamic_context "
            f"task={self.task_ids[task_index]} run={run_id} refresh={self.dynamic_context_refresh_count} "
            f"memory_ids={memory_ids} memory_types={memory_types} "
            f"memory_count={len(memory_list) if isinstance(memory_list, list) else 0}"
        )

        context_text = str(context_response.get("answer", "")).strip()
        return context_text if context_text else None

    def _per_turn_memory_refresh_due(self, run_id: int, task_index: int) -> bool:
        """True when an env-delivered user turn arrived after the last refresh."""
        if not (self.use_memory and self.per_turn_memory):
            return False
        return len(self.user_turn_texts[run_id][task_index]) - 1 > self.memory_refreshed_turn[run_id][task_index]

    def _cumulative_user_query(self, run_id: int, task_index: int) -> str:
        """Q_t = newline-joined verbatim user turns U_0..U_t (the frozen rule)."""
        texts = self.user_turn_texts[run_id][task_index]
        if not texts:
            raise ValueError("per-turn query requires at least one env-delivered user turn")
        return "\n".join(texts)

    def _inject_memory_context(self, run_id: int, task_index: int, context_text: str | None,
                               query_sha256: str, refresh_turn: int) -> bool:
        """Write the memory block into the first user message (single block).

        First-turn injection is byte-identical to the historical behaviour
        (original text read back from the message).  A per-turn refresh replaces
        the SAME block: the task text is restored from the stored verbatim first
        user turn, so blocks never accumulate across turns.  Every injection is
        logged for the post-run verification of query identity and block
        replacement.
        """
        if not context_text or not self.history[run_id][task_index]:
            return False
        first_msg = self.history[run_id][task_index][0]
        if not isinstance(first_msg, dict) or first_msg.get("role") != "user":
            return False
        replaced_previous_block = bool(self.per_turn_memory) and refresh_turn > 0
        if replaced_previous_block:
            stored = self.user_turn_texts[run_id][task_index]
            if not stored:
                raise ValueError("per-turn refresh requires the stored verbatim first user turn")
            orig = stored[0]
        else:
            orig = str(first_msg.get("content", ""))
        first_msg["content"] = (
            f"Task:\n{orig}\n\n"
            f"Some Related Experience to help you to complete the task:\n"
            f"{context_text}"
        )
        self.memory_injection_log[run_id][task_index].append({
            "turn": refresh_turn,
            "query_sha256": query_sha256,
            "context_sha256": canonical_hash(context_text),
            "replaced_previous_block": replaced_previous_block,
            "first_message_content_sha256": canonical_hash(first_msg["content"]),
        })
        return True

    def _refresh_tool_profiles(self) -> int:
        """任务开始时拉取一次 Tool Profile Pool 的演化画像（POST /tool-profiles）。

        画像在 train 过程中随 batch_updater 持续演化，因此按任务刷新而非构造时
        一次性拉取。服务不可用时保留上次缓存并静默降级，不阻塞任务。
        """
        try:
            response = requests.post(
                url=f"{self.memory_base_url}tool-profiles",
                json={
                    "tool_ids": [],
                    "min_total_calls": int(os.environ.get("TEVO_TOOL_PROFILE_MIN_CALLS", "0") or 0),
                },
                timeout=max(self.request_timeout_seconds, 120.0),
            )
        except requests.RequestException as exc:
            logger.warning(f"TEvo tool-profiles request failed: {exc}")
            return len(self.tool_profile_cache)

        payload = self.handle_api_response(response)
        if not payload or payload.get("success") is False:
            logger.warning(f"TEvo tool-profiles failed: {payload}")
            return len(self.tool_profile_cache)

        profiles = payload.get("profiles", {})
        if isinstance(profiles, dict):
            self.tool_profile_cache = {
                str(tool_id): profile
                for tool_id, profile in profiles.items()
                if isinstance(profile, dict)
            }
        return len(self.tool_profile_cache)

    def _rebuild_injected_tool_schema(self, run_id: int, task_index: int) -> int:
        """把画像贴到当前任务的可见工具 schema 上（base schema 不被修改）。"""
        base_schema = self.tool_schema[run_id][task_index]
        if not self.use_tool_profiles or not base_schema:
            return 0
        enriched, injected = apply_tool_profiles(
            base_schema,
            self.tool_profile_cache,
            fields=self.tool_profile_fields,
            max_chars=self.tool_profile_max_chars,
        )
        self.injected_tool_schema[run_id][task_index] = enriched
        logger.info(
            "tevo_tool_profiles "
            f"task={self.task_ids[task_index]} run={run_id} "
            f"profiles={len(self.tool_profile_cache)} visible={len(base_schema)} injected={injected}"
        )
        return injected

    def _llm_tool_schema(self, run_id: int, task_index: int) -> list[dict]:
        """给 LLM 的工具 schema：注入开启且有产物时用注入版，否则用原始 schema。"""
        if self.use_tool_profiles:
            injected = self.injected_tool_schema[run_id][task_index]
            if injected:
                return injected
        return self.tool_schema[run_id][task_index]

    def summary_memory(self, trajectories):
        """汇总对话消息并生成任务记忆。"""
        response = requests.post(
            url=f"{self.memory_base_url}summary_task_memory",
            json={
                "trajectories": trajectories,
                "success_threshold": 1.0,
                "enable_soft_comparison": True,
                "validation_threshold": 0.5,
            },
        )

        result = self.handle_api_response(response)
        if not result:
            return []

        # 从响应中提取 memory_list
        memory_list = result.get("metadata", {}).get("memory_list", [])
        logger.info(f"add new memories: {memory_list}")
        return memory_list

    def add_memory(self, memory_list):
        """将记忆添加到记忆池。"""
        response = requests.post(
            url=f"{self.memory_base_url}add_task_memory",
            json={
                "memory_list": memory_list,
            },
        )
        response.raise_for_status()

    def update_memory_information(self, memory_list, update_utility: bool = False):
        """更新记忆信息。"""
        response = requests.post(
            url=f"{self.memory_base_url}record_task_memory",
            json={
                "memory_list": memory_list,
                "update_utility": update_utility,
            },
        )
        response.raise_for_status()
        logger.info(response.json())

    def delete_memory(self):
        """从记忆池中删除记忆。"""
        response = requests.post(
            url=f"{self.memory_base_url}delete_task_memory",
            json={
                "freq_threshold": self.freq_threshold,
                "utility_threshold": self.utility_threshold,
            },
        )
        response.raise_for_status()

    def _tevo_url(self, endpoint: str) -> str:
        return f"{self.memory_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _latest_user_goal(self, run_id: int, task_index: int) -> str:
        for message in reversed(self.history[run_id][task_index]):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                content = str(message.get("content", "")).strip()
                if content and content != "[CONVERSATION_COMPLETED]":
                    return self.get_query_without_experience(content)
        first = self.history[run_id][task_index][0] if self.history[run_id][task_index] else {}
        if not isinstance(first, dict):
            return self.get_query_without_experience(str(first))
        return self.get_query_without_experience(str(first.get("content", "")))

    def _recent_error_context(self, run_id: int, task_index: int) -> tuple[str | None, list[str]]:
        errors: list[str] = []
        saw_tool_result = False
        for message in reversed(self.history[run_id][task_index][-8:]):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue
            saw_tool_result = True
            content = str(message.get("content", "")).strip()
            if not content:
                break
            if self._tool_output_success(content):
                break
            errors.append(content[:300])
        if not saw_tool_result:
            return None, []
        return (errors[0] if errors else None), errors

    @staticmethod
    def _tool_output_success(output: str) -> bool:
        return not tool_output_is_error(output)

    @staticmethod
    def _parse_output_payload(output: Any) -> Any | None:
        if isinstance(output, (dict, list)):
            return output
        text = str(output or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (SyntaxError, ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _payload_indicates_failure(payload: Any) -> bool:
        return tool_output_is_error(payload)

    @staticmethod
    def _raw_text_indicates_failure(output: Any) -> bool:
        return tool_output_is_error(output)

    @staticmethod
    def _parse_tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    def _observed_tool_calls_from_assistant(cls, assistant_msg: dict[str, Any]) -> list[dict[str, Any]]:
        observed: list[dict[str, Any]] = []
        if not isinstance(assistant_msg, dict):
            return observed
        for tool_call in cls._sanitize_tool_calls(assistant_msg.get("tool_calls", []) or []):
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            tool_id = str(function.get("name", "")).strip()
            if not tool_id:
                continue
            observed.append({
                "tool_id": tool_id,
                "args": cls._parse_tool_call_args(tool_call),
                "success": False,
                "output": "",
                "error": "",
            })
        return observed

    @staticmethod
    def _memory_ids_from_retrieved(memory_list: list[Any]) -> list[str]:
        used: list[str] = []
        for memory in memory_list or []:
            if not isinstance(memory, dict):
                continue
            record_id = str(memory.get("id", "")).strip()
            if record_id and record_id not in used:
                used.append(record_id)
        return used

    @staticmethod
    def _memory_action_matched_ids(memory_list: list[Any], tool_calls: list[dict[str, Any]]) -> list[str]:
        """归因匹配：记忆的 tool_tags 与本任务实际调用的工具有交集 → 视为动作命中。

        统一记忆重构后，信封里的 tool_tags 是唯一的归因挂钩；无 tool_tags 的
        记忆只承担 exposure（展示计数），不参与 credited/blamed。
        """
        matched: list[str] = []
        called_tools = {
            str(call.get("tool_id", "") or "").strip()
            for call in tool_calls or []
            if isinstance(call, dict)
        }
        called_tools.discard("")
        if not called_tools:
            return matched

        for memory in memory_list or []:
            if not isinstance(memory, dict):
                continue
            record_id = str(memory.get("id", "") or "").strip()
            if not record_id or record_id in matched:
                continue
            memory_tools = {
                str(tool).strip()
                for tool in (memory.get("tool_tags") or [])
                if str(tool).strip()
            }
            if memory_tools & called_tools:
                matched.append(record_id)
        return matched

    def _observer_step_state(
        self,
        *,
        task_id: str,
        user_goal: str,
        executed_tools: list[str],
        observations: list[str],
        latest_error: str = "",
        episode_id: str = "",
        turn_id: int | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "user_goal": user_goal,
            "executed_tools": list(executed_tools),
            "observations": list(observations),
            "latest_error": latest_error,
            "current_phase": "recover" if latest_error else "main",
            "metadata": {"episode_id": episode_id, "turn_id": turn_id},
        }

    def finalize_tevo_task(
        self,
        *,
        task_id: str,
        user_goal: str,
        reward: float,
        tool_calls: list[dict[str, Any]],
        final_output: str,
        messages: list[dict[str, Any]] | None = None,
        tool_schema: list[dict[str, Any]] | None = None,
        memory_used_ids: list[str] | None = None,
        retrieved_memory_ids: list[str] | None = None,
        action_matched_memory_ids: list[str] | None = None,
        credited_memory_ids: list[str] | None = None,
        blamed_memory_ids: list[str] | None = None,
        allow_learning: bool | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any] | None:
        """任务结束后回灌 TEvo。allow_learning=false 时只返回验证日志，不写学习状态。"""
        if not tool_calls:
            return None
        last_tool_call = tool_calls[-1] if tool_calls else {}
        result_payload = {
            "state": {
                "task_id": task_id,
                "user_goal": user_goal,
                "executed_tools": [str(call.get("tool_id", "")) for call in tool_calls if call.get("tool_id")],
                "observations": [str(call.get("output", "")) for call in tool_calls if call.get("output")],
            },
            "selected_tool_id": str(last_tool_call.get("tool_id", "")),
            "selected_tool_args": last_tool_call.get("args", {}) if isinstance(last_tool_call, dict) else {},
        }
        used_ids: list[str] = []
        for record_id in memory_used_ids or []:
            record_id = str(record_id).strip()
            if record_id and record_id not in used_ids:
                used_ids.append(record_id)
        if allow_learning is None:
            allow_learning = self._tevo_allow_learning_for_current_run()
        try:
            response = requests.post(
                url=self._tevo_url("finalize-task"),
                json={
                    "task_id": task_id,
                    "result": result_payload,
                    "success": reward == 1,
                    "execution_trace": {
                        "tool_calls": tool_calls,
                        "episode_id": episode_id,
                        "final_output": final_output,
                        "memory_used_ids": used_ids,
                        "retrieved_memory_ids": retrieved_memory_ids or [],
                        "action_matched_memory_ids": action_matched_memory_ids or [],
                        "credited_memory_ids": credited_memory_ids or [],
                        "blamed_memory_ids": blamed_memory_ids or [],
                        "user_goal": user_goal,
                        "messages": messages or [],
                        "tool_schema": tool_schema or [],
                    },
                    "failure_reason": "" if reward == 1 else "bfcl_reward_not_success",
                    "allow_learning": allow_learning,
                },
                timeout=max(self.request_timeout_seconds, 120.0),
            )
        except requests.RequestException as exc:
            logger.warning(f"TEvo finalize request failed: {exc}")
            return None
        result = self.handle_api_response(response)
        if result and result.get("success") is False:
            logger.warning(f"TEvo finalize failed: {result}")
        return result

    def _tevo_allow_learning_for_current_run(self) -> bool:
        if not self.use_memory_addition:
            return False
        if self._is_validation_observer_run() and not self._allow_val_observer_learning_override():
            if not self._logged_val_observer_readonly:
                logger.info(
                    "TEvo val observer is read-only by default; "
                    "set TEVO_ALLOW_VAL_OBSERVER_LEARNING=1 to write memories/profiles during val."
                )
                self._logged_val_observer_readonly = True
            return False
        return True

    def _is_validation_observer_run(self) -> bool:
        if not self.use_tevo_observer:
            return False
        experiment = str(self.experiment_name or "").lower()
        data_name = Path(str(self.data_path or "")).name.lower()
        return (
            "_val" in experiment
            or "-val" in experiment
            or "base-val" in experiment
            or "_val" in data_name
            or "-val" in data_name
        )

    def _is_train_observer_run(self) -> bool:
        if not self.use_tevo_observer:
            return False
        experiment = str(self.experiment_name or "").lower()
        data_name = Path(str(self.data_path or "")).name.lower()
        return (
            "_train" in experiment
            or "-train" in experiment
            or "base-train" in experiment
            or "_train" in data_name
            or "-train" in data_name
        )

    @staticmethod
    def _allow_val_observer_learning_override() -> bool:
        return os.environ.get("TEVO_ALLOW_VAL_OBSERVER_LEARNING", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    @staticmethod
    def _disable_memory_usage_feedback() -> bool:
        return os.environ.get("TEVO_DISABLE_MEMORY_USAGE_FEEDBACK", "").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def _disable_train_memory_read_during_sampling(self) -> bool:
        """Optional ablation: keep train trials independent while still writing memories."""
        if not (self.use_memory and self._is_train_observer_run()):
            return False
        value = os.environ.get("TEVO_DISABLE_TRAIN_MEMORY_READ_DURING_SAMPLING", "0").strip().lower()
        if value in {"0", "false", "no", "off"}:
            return False
        if not self._logged_train_memory_read_disabled:
            logger.info(
                "TEvo memory reads are disabled during train sampling; "
                "set TEVO_DISABLE_TRAIN_MEMORY_READ_DURING_SAMPLING=0 to enable online train reads."
            )
            self._logged_train_memory_read_disabled = True
        return True

    @staticmethod
    def _compact_finalize_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return payload
        compact: dict[str, Any] = {
            "success": payload.get("success"),
            "error": payload.get("error"),
            "verifier_score": payload.get("verifier_score"),
            "memory_attribution": payload.get("memory_attribution", {}),
            "memory_changes": payload.get("memory_changes", {}),
            "batch_profile_changes": payload.get("batch_profile_changes", {}),
            "cold_start_phase": payload.get("cold_start_phase"),
            "tasks_seen": payload.get("tasks_seen"),
        }
        breakdown = payload.get("verifier_breakdown", {})
        if isinstance(breakdown, dict):
            compact["verifier_breakdown"] = {
                key: breakdown.get(key)
                for key in ("correctness", "appropriateness", "efficiency", "recovery")
                if key in breakdown
            }
        return compact

    def _build_result_entry(
        self,
        *,
        run_id: int,
        task_index: int,
        start_time: str,
        reward: float,
        observed_tool_calls: list[dict[str, Any]],
        tevo_finalize_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        task_history = self.history[run_id][task_index]
        entry = {
            "run_id": run_id,
            "task_id": self.task_ids[task_index],
            "experiment_name": self.experiment_name,
            "task_completed": self.task_completed(run_id, task_index),
            "reward": reward,
            "task_start_time": start_time,
            "tevo_observer_enabled": self.use_tevo_observer,
            "tool_scope": self.tool_scope,
            "episode_id": self._active_episode_id,
            "sampling": self._sampling_metadata(),
            "request_metrics": self._jsonable_snapshot(self.request_metrics),
            "context_budget_metrics": self._jsonable_snapshot(self.context_budget_metrics),
            "usage_summary": self._usage_summary(),
            "memory_injection_log": self._jsonable_snapshot(self.memory_injection_log[run_id][task_index]),
        }
        if self._evaluation_error is not None:
            entry["evaluation_error"] = dict(self._evaluation_error)
            entry["error"] = "BFCL evaluation failed"
            entry["phase"] = "evaluation"
        if self.compact_results:
            entry.update({
                "num_messages": len(task_history),
                "observed_tool_call_count": len(observed_tool_calls),
                "tevo_finalize_result": self._compact_finalize_payload(tevo_finalize_result),
            })
            return entry
        entry.update({
            "task_history": task_history,
            "observed_tool_calls": observed_tool_calls,
            "tevo_finalize_result": tevo_finalize_result,
        })
        return entry

    def _build_error_result_entry(
        self,
        *,
        run_id: int,
        task_index: int,
        start_time: str,
        error: Exception,
    ) -> dict[str, Any]:
        task_history = self.history[run_id][task_index]
        entry = {
            "run_id": run_id,
            "task_id": self.task_ids[task_index],
            "experiment_name": self.experiment_name,
            "task_completed": self.task_completed(run_id, task_index),
            "reward": 0.0,
            "task_start_time": start_time,
            "error": repr(error),
            "episode_id": self._active_episode_id,
            "sampling": self._sampling_metadata(),
            "request_metrics": self._jsonable_snapshot(self.request_metrics),
            "context_budget_metrics": self._jsonable_snapshot(self.context_budget_metrics),
            "usage_summary": self._usage_summary(),
            "memory_injection_log": self._jsonable_snapshot(self.memory_injection_log[run_id][task_index]),
        }
        if self.compact_results:
            entry["num_messages"] = len(task_history)
            return entry
        entry["task_history"] = task_history
        return entry

    def _release_run_state(self, run_id: int, task_index: int) -> None:
        """Drop bulky per-run state once its result/finalizer has been recorded."""
        self._reset_bfcl_instance_cache(run_id, task_index)
        for bucket in (
            self.history,
            self.retrieved_memory_list,
            self.test_entry,
            self.original_test_entry,
            self.tool_schema,
            self.injected_tool_schema,
            self.user_turn_texts,
            self.memory_injection_log,
        ):
            try:
                bucket[run_id][task_index] = []
            except (IndexError, TypeError):
                pass
        try:
            self.memory_refreshed_turn[run_id][task_index] = -1
        except (IndexError, TypeError):
            pass
        gc.collect()

    def _defer_val_observer_learning_until_task_group_done(self) -> bool:
        """保持 pass@k 的 trial 独立性：同一 task 的多次 trial 全部结束后再写库。"""
        if not (self.use_tevo_observer and self._is_validation_observer_run()):
            return False
        if not self._allow_val_observer_learning_override():
            return False
        value = os.environ.get("TEVO_DEFER_VAL_OBSERVER_LEARNING", "1").strip().lower()
        if value in {"0", "false", "no", "off"}:
            return False
        if not self._logged_val_observer_deferred:
            logger.info(
                "TEvo val observer learning is deferred until all trials of the same task finish; "
                "set TEVO_DEFER_VAL_OBSERVER_LEARNING=0 to commit immediately."
            )
            self._logged_val_observer_deferred = True
        return True

    def _defer_train_observer_learning_until_sampling_done(self) -> bool:
        """Sample all train trajectories before writing memory (single actor only)."""
        if not (self.use_tevo_observer and self._is_train_observer_run()):
            return False
        value = os.environ.get("TEVO_DEFER_TRAIN_OBSERVER_LEARNING", "1").strip().lower()
        if value not in {"1", "true", "yes", "on"}:
            return False
        if not self._logged_train_observer_deferred:
            logger.info(
                "TEvo train observer learning is deferred until this actor's train samples finish; "
                "set TEVO_DEFER_TRAIN_OBSERVER_LEARNING=0 to commit immediately."
            )
            self._logged_train_observer_deferred = True
        return True

    def _flush_deferred_tevo_finalizers(
        self, finalizers: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> None:
        """Commit completed sampling, then extract per-task comparisons from saved goals."""
        by_task: dict[str, list[dict[str, Any]]] = {}
        for _, kwargs in finalizers:
            by_task.setdefault(kwargs["task_id"], []).append(self._jsonable_snapshot(kwargs))
        for result_entry, kwargs in finalizers:
            try:
                deferred_result = self.finalize_tevo_task(**kwargs, allow_learning=True)
            except Exception as exc:
                logger.exception("Deferred TEvo finalize failed")
                deferred_result = {"success": False, "error_type": type(exc).__name__}
            result_entry["tevo_deferred_finalize_result"] = (
                self._compact_finalize_payload(deferred_result) if self.compact_results else deferred_result
            )
        for task_id, task_trials in by_task.items():
            if len(task_trials) < 2:
                continue
            # History has been released already; these snapshots retain the original goal.
            trials = [{
                "reward": kwargs["reward"],
                "tool_calls": kwargs["tool_calls"],
                "messages": kwargs["messages"],
                "tool_schema": kwargs["tool_schema"],
            } for kwargs in task_trials]
            try:
                response = requests.post(
                    url=self._tevo_url("comparative-extract"),
                    json={"task_id": task_id, "user_goal": task_trials[0]["user_goal"], "trials": trials},
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(f"Comparative extraction failed for task {task_id}: {type(exc).__name__}")
        finalizers.clear()

    def _parse_qwen3_tool_calls(self, content: str) -> list[dict] | None:
        """Parse Qwen3-8B tool calls from content text.

        Qwen3-8B outputs tool calls as:
</think>

{"name": "func", "arguments": {...}}
        """
        import json
        import re

        tool_calls = []
        pattern = re.compile(r'\n\{"name"\s*:\s*"([^"]+)",\s*"arguments"\s*:\s*(.*?)\}\s*\n', re.DOTALL)
        for match in pattern.finditer(content):
            name = match.group(1)
            args_str = match.group(2).strip()
            if not args_str.endswith("}"):
                args_str += "}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
        return tool_calls if tool_calls else None

    @staticmethod
    def _parse_qwen3_xml_tool_calls(content: str) -> list[dict] | None:
        """Parse Qwen3-8B XML-wrapped tool calls from content text.

        vLLM with qwen3_xml parser may output:
<tool_call>
{"name": "func", "arguments": {...}}
</tool_call>
        """
        import json
        import re

        tool_calls = []

        def _extract_xml_call(match):
            call_text = match.group(1).strip()
            try:
                call_data = json.loads(call_text)
                name = call_data.get("name", "")
                arguments = call_data.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
            except (json.JSONDecodeError, Exception):
                return ""
            tool_calls.append({
                "id": f"call_xml_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
            return ""

        re.sub(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            _extract_xml_call,
            content or "",
            flags=re.DOTALL,
        )
        return tool_calls if tool_calls else None

    @staticmethod
    def _jsonable_snapshot(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    @staticmethod
    def _arguments_to_json_string(arguments: Any) -> str:
        if isinstance(arguments, dict):
            return json.dumps(arguments, ensure_ascii=False)
        if arguments in (None, ""):
            return "{}"
        if not isinstance(arguments, str):
            return "{}"

        text = arguments.strip()
        if not text:
            return "{}"

        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        return "{}"

    @classmethod
    def _sanitize_tool_calls(cls, tool_calls: Any) -> list[dict[str, Any]]:
        clean_calls: list[dict[str, Any]] = []
        if not isinstance(tool_calls, list):
            return clean_calls

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            call_id = str(tool_call.get("id", "")).strip() or f"call_{len(clean_calls)}"
            clean_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": cls._arguments_to_json_string(function.get("arguments", "{}")),
                },
            })
        return clean_calls

    @classmethod
    def _sanitize_assistant_message(cls, msg: Any) -> dict[str, Any]:
        if not isinstance(msg, dict):
            return {"role": "assistant", "content": str(msg or "")}

        clean = dict(msg)
        clean["role"] = "assistant"
        clean["content"] = "" if msg.get("content") is None else str(msg.get("content", ""))
        tool_calls = cls._sanitize_tool_calls(clean.get("tool_calls", []) or [])
        if tool_calls:
            clean["tool_calls"] = tool_calls
        else:
            clean.pop("tool_calls", None)
        return clean

    @classmethod
    def _messages_for_llm_request(cls, messages: list[Any]) -> list[dict[str, Any]]:
        clean_messages: list[dict[str, Any]] = []
        for message in messages or []:
            if not isinstance(message, dict):
                clean_messages.append({"role": "user", "content": str(message or "")})
                continue
            clean = dict(message)
            role = str(clean.get("role", "")).strip() or "user"
            clean["role"] = role
            if role == "assistant":
                clean = cls._sanitize_assistant_message(clean)
            elif "content" in clean and not isinstance(clean.get("content"), str):
                clean["content"] = str(clean.get("content", ""))
            clean_messages.append(clean)
        return clean_messages

    @classmethod
    def _messages_for_tevo_finalize(cls, messages: list[Any]) -> list[dict[str, Any]]:
        return cls._jsonable_snapshot(cls._messages_for_llm_request(messages))

    def _base_trial_seed(self) -> int | None:
        if self.trial_seeds is not None:
            return self.trial_seeds[self._active_run_id]
        raw = os.environ.get("TEVO_LLM_SEED", "0").strip()
        if not raw:
            return None
        try:
            seed = int(raw)
            if seed < 0:
                raise ValueError
            return seed
        except ValueError:
            raise ValueError("TEVO_LLM_SEED must be an integer or empty") from None

    def _trial_seed(self) -> int | None:
        base = self._base_trial_seed()
        if self.trial_seeds is not None:
            return derive_seed(base, self.task_ids[self._active_task_index])
        return base  # Preserve the frozen baseline's legacy environment-seed protocol.

    @staticmethod
    def _request_hash(value: Any) -> str:
        return canonical_hash(value)

    def _sampling_metadata(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "trial_seed": self._base_trial_seed(),
            "effective_seed": self._trial_seed(),
            "seed_mode": "explicit_task_derived" if self.trial_seeds is not None else "legacy_env",
            "max_tokens": self.max_tokens,
            "context_budget": self.context_budget,
            "thinking": self.enable_thinking,
            "request_timeout_seconds": self.request_timeout_seconds,
        }

    def _usage_summary(self) -> dict[str, int | float | None]:
        summary: dict[str, int | float | None] = {
            "latency_seconds": sum(event["latency_seconds"] for event in self.request_metrics),
        }
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            counts = [(event.get("usage") or {}).get(name) for event in self.request_metrics]
            summary[name] = sum(counts) if counts and all(isinstance(count, int) for count in counts) else None
        return summary

    def call_llm(self, messages: list, tool_schemas: list[dict]) -> dict[str, Any]:
        """调用 LLM。"""
        for i in range(self.max_retries + 1):
            request_started = time.perf_counter()
            event: dict[str, Any] = {
                "request_index": len(self.request_metrics),
                "run_id": self._active_run_id,
                "task_id": self.task_ids[self._active_task_index],
                "attempt": i,
                "usage": None,
                "status": "error",
                "provider_request_sent": False,
                "generation_transport_invoked": False,
            }
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": self._messages_for_llm_request(messages),
                    "temperature": self.temperature,
                    "stream": False,  # Always non-streaming for reliable tool call parsing
                }
                seed = self._trial_seed()
                if seed is not None:
                    kwargs["seed"] = seed
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens

                # Only add tools if there are tools available (API might reject empty lists)
                if tool_schemas and len(tool_schemas) > 0:
                    kwargs["tools"] = tool_schemas

                # Add provider-specific extra body if needed for Qwen thinking.
                extra_body = self._thinking_extra_body()
                if extra_body:
                    kwargs["extra_body"] = extra_body

                kwargs, admission = apply_context_budget(kwargs, self.context_budget, self._request_tokenization)
                if admission is not None:
                    self.context_budget_metrics.append(admission)

                event.update({
                    "seed": seed,
                    "requested_max_tokens": self.max_tokens,
                    "effective_max_tokens": kwargs.get("max_tokens"),
                    "context_budget": admission,
                    "prompt_sha256": self._request_hash(kwargs["messages"]),
                    "tool_schema_sha256": self._request_hash(tool_schemas),
                    "request_sha256": self._request_hash(kwargs),
                })
                event["generation_transport_invoked"] = True
                event["provider_request_sent"] = None  # Transport errors may leave delivery unknown.
                response = self.llm_client.chat.completions.create(**kwargs)
                usage = getattr(response, "usage", None)
                event.update({
                    "status": "success",
                    "provider_request_sent": True,
                    "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    "response_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                })
                verify_prompt_count(admission, event["usage"])
                out_msg = response.choices[0].message
                result = out_msg.model_dump(exclude_unset=True, exclude_none=True)
                answer_content = result.get("content") or ""
                reasoning_content = result.get("reasoning_content") or result.get("reasoning") or ""
                tool_info = result.get("tool_calls") or []
                tool_info = [
                    tc for tc in tool_info
                    if tc.get("function", {}).get("name") or tc.get("function", {}).get("arguments")
                ]
                # vLLM qwen3_xml parser may not extract tool_calls in thinking mode,
                # so parse them from content as fallback.  Try both bare JSON and
                # XML-wrapped <tool_call> formats (matches ReMe's dual-format support).
                if not tool_info and tool_schemas:
                    parsed = self._parse_qwen3_tool_calls(answer_content)
                    if not parsed:
                        parsed = self._parse_qwen3_xml_tool_calls(answer_content)
                    if parsed:
                        tool_info = parsed
                        # Strip tool call text from content
                        answer_content = ""
                msg = {
                    "role": "assistant",
                    "content": answer_content,
                    "reasoning_content": reasoning_content,
                }
                if tool_info:
                    msg["tool_calls"] = tool_info
                return self._sanitize_assistant_message(msg)
            except Exception as e:
                failed_admission = getattr(e, "context_budget_audit", None)
                if failed_admission is not None:
                    event["context_budget"] = failed_admission
                    if not any(item is failed_admission for item in self.context_budget_metrics):
                        self.context_budget_metrics.append(failed_admission)
                event["status"] = "error"
                event["error_type"] = type(e).__name__
                event["latency_seconds"] = time.perf_counter() - request_started
                if getattr(e, "retryable", True) is False:
                    raise
                logger.exception(f"encounter error with {e.args}")
                if i < self.max_retries:
                    sleep_s = self.retry_backoff_seconds * (2**i)
                    time.sleep(sleep_s)
            finally:
                event.setdefault("latency_seconds", time.perf_counter() - request_started)
                self.request_metrics.append(event)

        raise RuntimeError("call llm error after max retries")

    def _request_tokenization(self, payload: dict) -> str:
        return request_tokenization(payload, base_url=self.llm_base_url,
                                    api_key=os.environ.get("LLM_API_KEY", "dummy"),
                                    timeout_seconds=self.request_timeout_seconds)

    def _thinking_extra_body(self) -> dict[str, Any]:
        mode = self.thinking_extra_body_mode
        if mode == "auto":
            base_url = (self.llm_base_url or "").lower()
            if "127.0.0.1" in base_url or "localhost" in base_url:
                mode = "chat_template_kwargs"
            else:
                mode = "enable_thinking"
        if mode in {"0", "false", "no", "off", "none"}:
            return {}
        if mode in {"chat_template_kwargs", "vllm"}:
            return {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if mode in {"enable_thinking", "qwen", "dashscope", "api"}:
            return {"enable_thinking": self.enable_thinking}
        logger.warning(
            f"Unknown TEVO_LLM_THINKING_EXTRA_BODY={self.thinking_extra_body_mode!r}; "
            "falling back to enable_thinking"
        )
        return {"enable_thinking": self.enable_thinking}

    def env_step(self, run_id: int, index: int, messages: str) -> str:
        """
        处理对话中的一步。
        同时支持单轮和多轮场景。

        参数：
            messages: 对话消息列表，最后一条应为 assistant 响应
            test_entry: 包含 initial_config、involved_classes、question 等信息的测试条目
            **kwargs: 兼容性附加参数

        返回：
            包含下一条消息及可用工具（如适用）的字典
        """
        try:
            if not messages:
                return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])

            last_message = messages[-1] if isinstance(messages[-1], dict) else {}
            if last_message.get("role") != "assistant":
                return create_error_response(
                    "Last message must be from assistant",
                )

            if "tool_calls" in last_message and len(last_message["tool_calls"]) > 0:
                try:
                    tool_calls = self._sanitize_tool_calls(last_message["tool_calls"])
                    decoded_calls = self._convert_tool_calls_to_execution_format(
                        tool_calls,
                    )
                    # decoded_calls: [function(param=xxx)]
                    print(f"decoded_calls: {decoded_calls}")
                    if is_empty_execute_response(decoded_calls):
                        warnings.warn(
                            f"is_empty_execute_response: {is_empty_execute_response(decoded_calls)}",
                        )
                        return handle_user_turn(
                            self.original_test_entry[run_id][index],
                            self.current_turn[run_id][index],
                        )
                    return handle_tool_calls(
                        tool_calls,
                        decoded_calls,
                        self.original_test_entry[run_id][index],
                        self.current_turn[run_id][index],
                        execution_model_name=self._bfcl_env_model_name(run_id),
                    )
                except Exception as e:
                    warnings.warn(f"Errors during tool invocation: {str(e)}")
                    return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])
            else:
                return handle_user_turn(self.original_test_entry[run_id][index], self.current_turn[run_id][index])

        except Exception as e:
            return create_error_response(f"Failed to process request: {str(e)}")

    def _convert_tool_calls_to_execution_format(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[str]:
        """
        将 OpenAI 格式的工具调用转换为执行格式。

        参数：
            tool_calls: OpenAI 格式的工具调用列表

        返回：
            字符串格式的函数调用列表
        """
        execution_list = []

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            function_name = function.get("name", "")

            try:
                arguments = function.get("arguments", "{}")
                if isinstance(arguments, str):
                    args_dict = json.loads(arguments)
                else:
                    args_dict = arguments

                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args_dict.items()])
                execution_list.append(f"{function_name}({args_str})")

            except Exception:
                execution_list.append(f"{function_name}()")

        return execution_list

    def get_reward(self, run_id, index) -> float:
        """计算奖励。"""
        self._evaluation_error = None
        try:
            if not self.history[run_id][index] or not self.original_test_entry[run_id][index]:
                return 0.0

            model_name = self._bfcl_eval_model_name(run_id)
            self._reset_bfcl_instance_cache(run_id, index, include_env=False, include_eval=True)
            
            handler = OpenAICompletionsHandler(
                model_name,
                temperature=1.0,
            )  # FIXME: 魔法数字

            model_result_data = self._convert_conversation_to_eval_format(run_id, index)

            prompt_data = [self.original_test_entry[run_id][index]]

            # Grading needs calls/state only. Missing usage is kept as None;
            # cost helpers require measured numeric usage and must not see fabricated zeros.
            possible_answer = []

            if is_relevance_or_irrelevance(self.categories[index]):
                accuracy, _ = self._eval_relevance_test(
                    handler,
                    model_result_data,
                    prompt_data,
                    model_name,
                    self.category,
                )
            else:
                # 查找对应的 possible_answer 文件

                possible_answer_file = find_file_with_suffix(
                    self.answer_path,
                    self.categories[index],
                )
                possible_answer = load_file(possible_answer_file, sort_by_id=True)
                possible_answer = [item for item in possible_answer if item["id"] == self.task_ids[index]]
                if is_multi_turn(self.categories[index]):
                    accuracy, _ = self._eval_multi_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )
                else:
                    accuracy, _ = self._eval_single_turn_test(
                        handler,
                        model_result_data,
                        prompt_data,
                        possible_answer,
                        model_name,
                        self.categories[index],
                    )
            return accuracy

        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._evaluation_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            return 0
        finally:
            self._reset_bfcl_instance_cache(run_id, index, include_env=False, include_eval=True)

    def _convert_conversation_to_eval_format(self, run_id, index) -> Dict[str, Any]:
        """
        将对话历史转换为评测格式。

        参数：
            conversation_result: run_conversation 的结果
            original_test_entry: 原始测试条目数据

        返回：
            multi_turn_runner 或其他 runner 所需的数据格式
        """
        if is_multi_turn(self.categories[index]):
            turns_data = extract_multi_turn_responses(self.history[run_id][index])
        else:
            turns_data = extract_single_turn_response(self.history[run_id][index])

        model_result_data = {
            "id": self.task_ids[index],
            "result": turns_data,
            "latency": self._usage_summary()["latency_seconds"],
            "input_token_count": self._usage_summary()["prompt_tokens"],
            "output_token_count": self._usage_summary()["completion_tokens"],
        }

        return model_result_data

    def _eval_multi_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        评估多轮测试。

        参数：
            handler: 模型处理器实例
            model_result_data: 模型结果数据
            prompt_data: 提示词数据
            possible_answer: 候选答案数据
            model_name: 模型名称
            test_category: 测试类别

        返回：
            (accuracy, total_count) 元组
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = multi_turn_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                model_name=model_name,
                test_category=test_category,
                score_dir=score_dir,
            )
            capture_and_print_score_files(
                score_dir,
                model_name,
                test_category,
                "multi_turn",
            )
            return accuracy, total_count

    def _eval_single_turn_test(
        self,
        handler,
        model_result_data,
        prompt_data,
        possible_answer,
        model_name,
        test_category,
    ):
        """
        评估单轮 AST 测试。

        参数：
            handler: 模型处理器实例
            model_result_data: 模型结果数据
            prompt_data: 提示词数据
            possible_answer: 候选答案数据
            model_name: 模型名称
            test_category: 测试类别

        返回：
            (accuracy, total_count) 元组
        """
        language = "Python"
        if "java" in test_category.lower():
            language = "Java"
        elif "js" in test_category.lower() or "javascript" in test_category.lower():
            language = "JavaScript"

        with tempfile.TemporaryDirectory() as temp_dir:
            score_dir = Path(temp_dir)
            accuracy, total_count = ast_file_runner(
                handler=handler,
                model_result=[model_result_data],
                prompt=prompt_data,
                possible_answer=possible_answer,
                language=language,
                test_category=test_category,
                model_name=model_name,
                score_dir=score_dir,
            )
            capture_and_print_score_files(
                score_dir,
                model_name,
                test_category,
                "single_turn",
            )
            return accuracy, total_count

    def execute(self):
        """执行智能体。"""
        result = []
        counter = 0
        trial_plan = [
            (run_id, task_index, task_id)
            for run_id in range(self.num_trials)
            for task_index, task_id in enumerate(self.task_ids)
        ]
        deferred_tevo_finalizers_by_task: list[list[tuple[dict[str, Any], dict[str, Any]]]] = [
            [] for _ in self.task_ids
        ]
        deferred_train_tevo_finalizers: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for run_id, task_index, task_id in tqdm(trial_plan, desc=f"ray_index={self.index}"):
            self._active_run_id = run_id
            self._active_task_index = task_index
            # Episode identity is part of teacher prompts and comparison embeddings.
            # Use the original seed: the derived sampling seed is only 31 bits.
            self._active_episode_id = (
                canonical_hash(["bfcl-explicit-episode-v1", self.experiment_name, task_id, self.trial_seeds[run_id]])[:32]
                if self.trial_seeds is not None else uuid4().hex
            )
            self.request_metrics = []
            self.context_budget_metrics = []
            self._evaluation_error = None
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._reset_bfcl_instance_cache(run_id, task_index)
            observed_tool_calls: list[dict[str, Any]] = []
            observed_executed_tools: list[str] = []
            observed_outputs: list[str] = []
            tevo_finalize_result = None
            try:
                for i in range(self.max_interactions):
                    action_turn_id = self.current_turn[run_id][task_index]
                    action_user_goal = self._latest_user_goal(run_id, task_index)
                    if self.use_tool_profiles and i == 0:
                        # 每个任务开始时刷新画像缓存并重建注入后的工具 schema。
                        self._refresh_tool_profiles()
                        self._rebuild_injected_tool_schema(run_id, task_index)
                    if self.use_memory and (i == 0 or self._per_turn_memory_refresh_due(run_id, task_index)):
                        # ReMe-style: retrieve memory context and write it into the
                        # first user message (not injected as a system role).
                        # First turn: once, using the latest user goal.  Per-turn
                        # arms: on every new env-delivered user turn, the block is
                        # REPLACED in place with the frozen U_0..U_t recall.
                        if i == 0:
                            query_used = self._latest_user_goal(run_id, task_index)
                            context_text = self._retrieve_tevo_memory_context(run_id, task_index)
                            refresh_turn = 0
                        else:
                            query_used = self._cumulative_user_query(run_id, task_index)
                            context_text = self._retrieve_tevo_memory_context(run_id, task_index, query=query_used)
                            refresh_turn = len(self.user_turn_texts[run_id][task_index]) - 1
                        if self._inject_memory_context(run_id, task_index, context_text,
                                                       canonical_hash(query_used), refresh_turn):
                            self.memory_refreshed_turn[run_id][task_index] = refresh_turn
                        llm_messages = self.history[run_id][task_index]
                        llm_output = self.call_llm(
                            llm_messages,
                            self._llm_tool_schema(run_id, task_index),
                        )
                    elif self.use_memory:
                        llm_output = self.call_llm(
                            self.history[run_id][task_index],
                            self._llm_tool_schema(run_id, task_index),
                        )
                    else:
                        llm_output = self.call_llm(
                            self.history[run_id][task_index],
                            self._llm_tool_schema(run_id, task_index),
                        )
                    pending_observed_calls = self._observed_tool_calls_from_assistant(llm_output)
                    self.history[run_id][task_index].append(llm_output)

                    env_output = self.env_step(run_id, task_index, self.history[run_id][task_index])
                    # 与环境交互后，env_output 可能有以下返回：
                    # 1. 触发包含可用工具列表的新查询：
                    #    {"messages": [{"role": "user", "content": user_query}], "tools": tools}
                    # 2. 返回工具调用结果：{"messages":
                    #    [{"role": "tool", "content": {<exec_results>}, 'tool_call_id': 'chatcmpl-tool-xxx'}]}
                    #    <exec_results>：成功时返回结果字典，例如 {"travel_cost_list": [x]}；
                    #    失败时返回错误信息，
                    #    例如 {"error": "cd: temporary: No such directory. You cannot use path ..."}
                    # 3. 对话完成：
                    #    {"messages": [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]}
                    # 4. 程序错误：{"messages": [{"role": "env", "content": f"[ERROR] {error_message}"}]}

                    # 更新 tool_list
                    if "tools" in env_output:
                        turn_tool_schema = extract_tool_schema(deepcopy(env_output["tools"]))
                        self.tool_schema[run_id][task_index] = self._visible_tool_schema(turn_tool_schema)
                        self._rebuild_injected_tool_schema(run_id, task_index)

                    new_tool_calls = []
                    new_tool_call_ids = []
                    next_user_msg = ""
                    pending_user_turn_text = None
                    for idx, msg in enumerate(env_output.get("messages", [])):
                        if msg["role"] == "tool":
                            new_tool_calls.append(msg.get("content", ""))
                            new_tool_call_ids.append(msg.get("tool_call_id", ""))
                        elif msg["role"] == "user":
                            next_user_msg = msg.get("content", "")
                            pending_user_turn_text = next_user_msg
                            self.current_turn[run_id][task_index] += 1
                        else:  # 处理 env 角色消息
                            next_user_msg = msg.get("content", "")

                    if self.per_turn_memory and not new_tool_calls \
                            and pending_user_turn_text is not None \
                            and next_user_msg == pending_user_turn_text:
                        # 只收集真正被追加进历史的用户轮原文（与数据集逐字一致，
                        # 运行期查询据此构造，查找不中即 fail closed）。
                        self.user_turn_texts[run_id][task_index].append(pending_user_turn_text)

                    if new_tool_calls:
                        for idx, call in enumerate(new_tool_calls):
                            self.history[run_id][task_index].append(
                                {"role": "tool", "content": str(call), "tool_call_id": new_tool_call_ids[idx]},
                            )
                    else:
                        self.history[run_id][task_index].append({"role": "user", "content": next_user_msg})

                    if self.use_tevo_observer and pending_observed_calls:
                        for call_idx, observed_call in enumerate(pending_observed_calls):
                            executed = call_idx < len(new_tool_calls)
                            step_output = str(new_tool_calls[call_idx]) if executed else ""
                            step_success = executed and self._tool_output_success(step_output)
                            step_error = "" if step_success else (
                                step_output[:300] if executed else "No tool result was returned for this action."
                            )
                            observed_call["success"] = step_success
                            observed_call["output"] = step_output
                            observed_call["error"] = step_error
                            observed_call["execution_status"] = "executed" if executed else "not_executed"
                            observed_call["episode_id"] = self._active_episode_id
                            observed_call["turn_id"] = action_turn_id
                            observed_call["state"] = self._observer_step_state(
                                task_id=task_id,
                                user_goal=action_user_goal,
                                executed_tools=observed_executed_tools,
                                observations=observed_outputs,
                                latest_error=step_error,
                                episode_id=self._active_episode_id,
                                turn_id=action_turn_id,
                            )
                            observed_tool_calls.append(observed_call)
                            if executed:
                                observed_executed_tools.append(observed_call["tool_id"])
                            if step_output:
                                observed_outputs.append(step_output)

                    logger.info(f"index={self.index} task_id={task_id} run_id={run_id} iteration={i}")

                    if self.task_completed(run_id, task_index):
                        break

                reward = self.get_reward(run_id, task_index)
                tevo_finalize_deferred = False
                tevo_finalize_defer_scope = ""
                if self.use_tevo_observer:
                    last_message = self.history[run_id][task_index][-1] if self.history[run_id][task_index] else {}
                    final_output = (
                        str(last_message.get("content", ""))
                        if isinstance(last_message, dict)
                        else str(last_message)
                    )
                    retrieved_memory_ids = self._memory_ids_from_retrieved(
                        self.retrieved_memory_list[run_id][task_index],
                    )
                    action_matched_memory_ids = self._memory_action_matched_ids(
                        self.retrieved_memory_list[run_id][task_index],
                        observed_tool_calls,
                    )
                    credited_memory_ids = action_matched_memory_ids if reward == 1 else []
                    blamed_memory_ids = action_matched_memory_ids if reward != 1 else []
                    finalize_kwargs = {
                        "task_id": task_id,
                        "episode_id": self._active_episode_id,
                        "user_goal": self._latest_user_goal(run_id, task_index),
                        "reward": reward,
                        "tool_calls": self._jsonable_snapshot(observed_tool_calls),
                        "final_output": final_output,
                        "messages": self._messages_for_tevo_finalize(self.history[run_id][task_index]),
                        "tool_schema": self._jsonable_snapshot(self.tool_schema[run_id][task_index]),
                        "memory_used_ids": action_matched_memory_ids,
                        "retrieved_memory_ids": retrieved_memory_ids,
                        "action_matched_memory_ids": action_matched_memory_ids,
                        "credited_memory_ids": credited_memory_ids,
                        "blamed_memory_ids": blamed_memory_ids,
                    }
                    defer_train_learning = (
                        bool(observed_tool_calls)
                        and self._defer_train_observer_learning_until_sampling_done()
                        and self._tevo_allow_learning_for_current_run()
                    )
                    defer_val_learning = (
                        bool(observed_tool_calls)
                        and self._defer_val_observer_learning_until_task_group_done()
                        and self._tevo_allow_learning_for_current_run()
                    )
                    if defer_train_learning:
                        tevo_finalize_deferred = True
                        tevo_finalize_defer_scope = "train_actor_all_samples"
                    elif defer_val_learning:
                        tevo_finalize_result = self.finalize_tevo_task(
                            **finalize_kwargs,
                            allow_learning=False,
                        )
                        tevo_finalize_deferred = True
                        tevo_finalize_defer_scope = "val_task_group"
                    else:
                        tevo_finalize_result = self.finalize_tevo_task(**finalize_kwargs)
                elif self.use_memory:
                    if self.use_memory_addition:
                        new_traj_list = [
                            self.get_traj_from_task_history(task_id, self.history[run_id][task_index], reward),
                        ]
                        previous_memories = self.summary_memory(new_traj_list)
                        if reward == 1:
                            self.add_memory(previous_memories)

                    # 更新已检索记忆的 freq 与 utility 属性
                    update_utility: bool = reward == 1
                    if not self._disable_memory_usage_feedback():
                        self.update_memory_information(self.retrieved_memory_list[run_id][task_index], update_utility)

                counter += 1
                if self.use_memory_deletion and counter % self.delete_freq == 0:
                    self.delete_memory()

                result_entry = self._build_result_entry(
                    run_id=run_id,
                    task_index=task_index,
                    start_time=start_time,
                    reward=reward,
                    observed_tool_calls=observed_tool_calls,
                    tevo_finalize_result=tevo_finalize_result,
                )
                if tevo_finalize_deferred:
                    result_entry["tevo_finalize_deferred"] = True
                    result_entry["tevo_finalize_defer_scope"] = tevo_finalize_defer_scope
                    if tevo_finalize_defer_scope == "train_actor_all_samples":
                        deferred_train_tevo_finalizers.append((result_entry, finalize_kwargs))
                    else:
                        deferred_tevo_finalizers_by_task[task_index].append((result_entry, finalize_kwargs))
                result.append(result_entry)

            except Exception as e:
                logger.exception(f"encounter error with {e.args}")
                result.append(self._build_error_result_entry(
                    run_id=run_id,
                    task_index=task_index,
                    start_time=start_time,
                    error=e,
                ))
            finally:
                self._release_run_state(run_id, task_index)

            if run_id == self.num_trials - 1:
                deferred_tevo_finalizers = deferred_tevo_finalizers_by_task[task_index]
                self._flush_deferred_tevo_finalizers(deferred_tevo_finalizers)

        if deferred_train_tevo_finalizers:
            logger.info(
                f"Flushing {len(deferred_train_tevo_finalizers)} deferred train TEvo finalizers "
                "after this actor's train samples have completed."
            )
            self._flush_deferred_tevo_finalizers(deferred_train_tevo_finalizers)
        return result

    def task_completed(self, run_id, index):
        """
        检查任务是否完成。

        返回：
            完成返回 True，否则返回 False
        """
        if not self.history[run_id][index]:
            return False
        last_message = self.history[run_id][index][-1]
        return isinstance(last_message, dict) and last_message.get("content") == "[CONVERSATION_COMPLETED]"


BFCLAgent = ray.remote(LocalBFCLAgent) if ray is not None else LocalBFCLAgent


def main():
    """运行 BFCLAgent 的主函数。"""
    with open(os.getenv("BFCL_DATA_PATH"), "r", encoding="utf-8") as f:
        task_ids = [json.loads(l)["id"] for l in f]
    dataset_name = "dev"
    agent = LocalBFCLAgent(
        index=0,
        task_ids=[task_ids[0]],
        experiment_name=f"qwen3_8b_{dataset_name}",
    )
    result = agent.execute()
    logger.info(f"result={json.dumps(result)}")


if __name__ == "__main__":
    main()
