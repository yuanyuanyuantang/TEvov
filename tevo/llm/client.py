"""OpenAI-compatible LLM client used by TEvo."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path
import threading
from typing import Any
from urllib import error, request
import ssl

from tevo.config.shared_config import LLMConfig
from tevo.llm.context_budget import apply_context_budget, request_tokenization, verify_prompt_count


class DeepSeekLLMClient:
    """Use an OpenAI-compatible chat completions endpoint."""

    _LOCKS_GUARD = threading.Lock()
    _CACHE_LOCKS: dict[str, threading.RLock] = {}

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self._cache_lock = self._lock_for_cache_dir(self.cache_dir)
        self._last_request_at = 0.0
        self.context_budget_metrics: list[dict[str, Any]] = []
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "requests": 0,
            "retries": 0,
            "rate_limited_waits": 0,
            "rate_limit_errors": 0,
            "auth_errors": 0,
            "auth_recoveries": 0,
        }
        if self.config.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lock_for_cache_dir(cls, cache_dir: Path) -> threading.RLock:
        key = str(cache_dir.resolve())
        with cls._LOCKS_GUARD:
            lock = cls._CACHE_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._CACHE_LOCKS[key] = lock
            return lock

    @property
    def enabled(self) -> bool:
        """Return true when the client is explicitly enabled and has an API key."""

        return self.config.enabled and bool(self.config.resolved_api_key())

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        schema_hash: str = "default",
        sampling_profile: str | None = None,
    ) -> dict[str, Any]:
        """Send one chat completion request."""

        if not self.enabled:
            raise RuntimeError("LLM client disabled")

        sampling_payload = self._resolve_sampling_payload(
            temperature=temperature,
            sampling_profile=sampling_profile,
        )
        payload = {
            "cache_version": self.config.cache_version,
            "schema_hash": schema_hash,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # extra_body fields are merged at top-level of the HTTP request (OpenAI SDK convention)
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        if sampling_payload:
            payload.update(sampling_payload)
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        return self._submit_chat_payload(payload)

    def chat_with_tools(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = "required",
        temperature: float | None = None,
        schema_hash: str = "default",
        sampling_profile: str | None = None,
        parallel_tool_calls: bool = False,
    ) -> dict[str, Any]:
        """Send one chat completion request with OpenAI-compatible tool schemas."""

        if not self.enabled:
            raise RuntimeError("LLM client disabled")

        sampling_payload = self._resolve_sampling_payload(
            temperature=temperature,
            sampling_profile=sampling_profile,
        )
        payload = {
            "cache_version": self.config.cache_version,
            "schema_hash": schema_hash,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "tools": list(tools),
            "parallel_tool_calls": bool(parallel_tool_calls),
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        if sampling_payload:
            payload.update(sampling_payload)
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        return self._submit_chat_payload(payload)

    def _submit_chat_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        admission = None
        try:
            payload, admission = apply_context_budget(payload, self.config.context_budget, self._request_tokenization)
        except Exception as exc:
            if getattr(exc, "context_budget_audit", None):
                self.context_budget_metrics.append(exc.context_budget_audit)
            raise
        if admission is not None:
            self.context_budget_metrics.append(admission)
        cache_payload = {**payload, "_native_token_ids_sha256": admission["token_ids_sha256"]} if admission is not None else payload
        cache_key = self._cache_key(cache_payload)
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            verify_prompt_count(admission, cached.get("raw", {}).get("usage"))
            return {**cached, "context_budget": admission} if admission is not None else cached
        self._stats["cache_misses"] += 1
        raw = self._request_with_retry(payload)

        result = self._parse_chat_response(raw)
        verify_prompt_count(admission, result.get("raw", {}).get("usage"))
        self._save_cache(cache_key, result)
        return {**result, "context_budget": admission} if admission is not None else result

    def _request_tokenization(self, payload: dict) -> str:
        return request_tokenization(payload, base_url=self.config.base_url,
                                    api_key=self.config.resolved_api_key(), timeout_seconds=self.config.timeout_seconds)

    def _parse_chat_response(self, raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError(f"response root is {type(parsed).__name__}, expected object")
            error_payload = parsed.get("error")
            if error_payload:
                raise RuntimeError(f"LLM API returned error payload: {self._preview_payload(error_payload)}")

            choice = self._extract_choice_message(parsed)
            content = self._coerce_message_content(choice.get("content", ""))
            if not content.strip():
                content = self._coerce_message_content(choice.get("reasoning_content", ""))
            if not content.strip():
                content = self._coerce_message_content(choice.get("reasoning", ""))
            tool_calls = self._parse_tool_calls(choice.get("tool_calls"))
        except Exception as exc:  # noqa: BLE001
            preview = self._preview_raw_response(raw)
            raise RuntimeError(f"LLM response parse failed: {exc}; raw={preview}") from exc
        return {
            "raw": parsed,
            "content": content,
            "tool_calls": tool_calls,
            "created_at": time.time(),
        }

    def _extract_choice_message(self, parsed: dict[str, Any]) -> dict[str, Any]:
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise KeyError("missing choices[0]")
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                return message
            # Some OpenAI-compatible providers surface assistant fields directly on
            # the choice object when message is omitted/null.
            fallback_message = self._coerce_choice_to_message(choice)
            if fallback_message is not None:
                return fallback_message
            if message is not None:
                raise TypeError(
                    f"choices[{index}].message has unsupported type {type(message).__name__}"
                )
        raise TypeError("choices contained no usable message payload")

    def _coerce_choice_to_message(self, choice: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(choice, dict):
            return None
        has_message_fields = any(
            key in choice for key in ("content", "reasoning_content", "reasoning", "tool_calls")
        )
        if not has_message_fields:
            return None
        return {
            "content": choice.get("content", ""),
            "reasoning_content": choice.get("reasoning_content", ""),
            "reasoning": choice.get("reasoning", ""),
            "tool_calls": choice.get("tool_calls", []),
        }

    @staticmethod
    def _preview_payload(payload: Any, *, limit: int = 400) -> str:
        try:
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(payload)
        return text[:limit]

    @classmethod
    def _preview_raw_response(cls, raw: str, *, limit: int = 400) -> str:
        text = str(raw).replace("\n", "\\n")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @staticmethod
    def _coerce_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            return "\n".join(parts)
        return str(content or "")

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []
        tool_calls: list[dict[str, Any]] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            arguments_raw = function.get("arguments", "{}")
            tool_calls.append(
                {
                    "id": str(raw_call.get("id", "")),
                    "name": name,
                    "arguments": self._parse_tool_arguments(arguments_raw),
                    "arguments_raw": arguments_raw,
                }
            )
        return tool_calls

    @staticmethod
    def _parse_tool_arguments(arguments_raw: Any) -> dict[str, Any]:
        if isinstance(arguments_raw, dict):
            return dict(arguments_raw)
        raw_text = str(arguments_raw or "").strip()
        if not raw_text:
            return {}
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) >= 3:
                raw_text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(raw_text)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(raw_text)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            return {}

    def _resolve_sampling_payload(
        self,
        *,
        temperature: float | None,
        sampling_profile: str | None,
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        profile_payload = {}
        if sampling_profile:
            raw_profile = self.config.sampling_profiles.get(sampling_profile, {})
            if isinstance(raw_profile, dict):
                profile_payload = dict(raw_profile)
        if profile_payload:
            resolved.update(profile_payload)
        if temperature is not None:
            resolved["temperature"] = temperature
        elif "temperature" not in resolved:
            if sampling_profile == "exploration":
                resolved["temperature"] = self.config.planner_temperature
            else:
                resolved["temperature"] = self.config.critic_temperature
        return resolved

    def _cache_key(self, payload: dict[str, Any]) -> str:
        if self.config.context_budget is not None:
            payload = {"payload": payload, "context_budget": self.config.context_budget}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _wait_for_rate_limit(self) -> None:
        min_interval = max(self.config.min_request_interval_seconds, 0.0)
        if min_interval <= 0:
            return
        elapsed = time.time() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            self._stats["rate_limited_waits"] += 1

    def _build_request(self, payload: dict[str, Any], api_key: str) -> request.Request:
        base_url = self.config.base_url.rstrip("/")
        target_url = f"{base_url}/chat/completions"
        return request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

    def _request_with_retry(self, payload: dict[str, Any]) -> str:
        last_error: Exception | None = None
        current_api_key = self.config.resolved_api_key()
        ssl_context = None
        if self.config.insecure_skip_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        for attempt in range(self.config.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                self._stats["requests"] += 1
                req = self._build_request(payload, current_api_key)
                with request.urlopen(req, timeout=self.config.timeout_seconds, context=ssl_context) as response:
                    self._last_request_at = time.time()
                    return response.read().decode("utf-8")
            except error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="ignore")
                if exc.code == 429:
                    self._stats["rate_limit_errors"] += 1
                if exc.code in {401, 403}:
                    self._stats["auth_errors"] += 1
                    refreshed_key = self.config.resolved_api_key()
                    if refreshed_key and refreshed_key != current_api_key:
                        current_api_key = refreshed_key
                        self._stats["auth_recoveries"] += 1
                last_error = RuntimeError(f"LLM HTTP error: {exc.code} {message}")
            except Exception as exc:  # noqa: BLE001
                last_error = RuntimeError(f"LLM request failed: {exc}")
            if attempt < self.config.max_retries:
                self._stats["retries"] += 1
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM request failed without explicit error")

    def _load_cache(self, cache_key: str) -> dict[str, Any] | None:
        if not self.config.cache_enabled:
            return None
        path = self._cache_path(cache_key)
        with self._cache_lock:
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                return None
            created_at = float(payload.get("created_at", 0.0))
            if self.config.cache_ttl_seconds > 0 and created_at and time.time() - created_at > self.config.cache_ttl_seconds:
                path.unlink(missing_ok=True)
                return None
            return payload

    def _save_cache(self, cache_key: str, result: dict[str, Any]) -> None:
        if not self.config.cache_enabled:
            return
        path = self._cache_path(cache_key)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        with self._cache_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
            self._prune_cache()

    def _prune_cache(self) -> None:
        entries = sorted(self.cache_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for entry in entries[self.config.cache_max_entries :]:
            entry.unlink(missing_ok=True)

    def export_stats(self) -> dict[str, int]:
        """Export LLM request and cache stats."""

        return dict(self._stats)
