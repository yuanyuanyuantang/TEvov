"""BFCL 连续执行会话封装。

BFCL 原生 `execute_multi_turn_func_call()` 实际通过模块级全局实例缓存实现跨步状态延续。
这里将其显式包装成 session，确保：
- 同一任务内连续复用环境；
- 新任务不会误复用旧实例；
- 任务结束后主动清理全局实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

_BFCL_SESSION_AVAILABLE = False
_bfcl_multi_turn_utils = None
try:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "benchmark" / "bfcl" / "gorilla" / "berkeley-function-call-leaderboard"))
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as _bfcl_multi_turn_utils
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    _BFCL_SESSION_AVAILABLE = True
except ImportError:
    pass


@dataclass(slots=True)
class BFCLSessionExecutor:
    """单任务 BFCL 连续执行会话。"""

    task_id: str
    initial_config: dict[str, Any]
    involved_classes: list[str]
    model_name_prefix: str = "tevo-pipeline-session"
    long_context: bool = False
    is_eval_run: bool = True
    _session_token: str = field(init=False, repr=False)
    _model_name: str = field(init=False, repr=False)
    _started: bool = field(init=False, repr=False, default=False)
    _closed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._session_token = uuid4().hex[:8]
        self._model_name = f"{self.model_name_prefix}-{self._session_token}"
        self._started = False
        self._closed = False

    @property
    def available(self) -> bool:
        return _BFCL_SESSION_AVAILABLE

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("BFCL session already closed")
        if self._started:
            return
        if not self.available:
            raise RuntimeError("BFCL session runtime unavailable")
        self._clear_instances()
        # 传空调用列表即可触发 involved_instances 初始化。
        execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=self.initial_config,
            involved_classes=self.involved_classes,
            model_name=self._model_name,
            test_entry_id=self.task_id,
            long_context=self.long_context,
            is_evaL_run=self.is_eval_run,
        )
        self._started = True

    def execute(self, call_string: str) -> tuple[list[str], dict[str, Any]]:
        if not self._started:
            self.start()
        return execute_multi_turn_func_call(
            func_call_list=[call_string],
            initial_config=self.initial_config,
            involved_classes=self.involved_classes,
            model_name=self._model_name,
            test_entry_id=self.task_id,
            long_context=self.long_context,
            is_evaL_run=self.is_eval_run,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._clear_instances()
        self._closed = True

    def _instance_names(self) -> list[str]:
        effective_model_name = self._model_name + ("_eval" if self.is_eval_run else "")
        sanitized = effective_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        return [
            f"{sanitized}_{self.task_id}_{class_name.lower()}_instance"
            for class_name in self.involved_classes
        ]

    def _clear_instances(self) -> None:
        if _bfcl_multi_turn_utils is None:
            return
        module_dict = vars(_bfcl_multi_turn_utils)
        for instance_name in self._instance_names():
            module_dict.pop(instance_name, None)
