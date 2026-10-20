"""Exact vLLM chat admission; no text truncation or tokenizer retries."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Callable
from urllib import request
from urllib.parse import urlsplit, urlunsplit

from tevo.reproducibility import canonical_hash


class ContextBudgetError(RuntimeError):
    retryable = False

    def __init__(self, message: str, audit: dict | None = None):
        super().__init__(message)
        self.context_budget_audit = audit or {}


def validate_context_budget(value: Any) -> dict | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {"policy", "context_window_tokens"}
            or value.get("policy") != "server_tokenize_headroom"):
        raise ValueError("context_budget requires server_tokenize_headroom and context_window_tokens")
    window = value["context_window_tokens"]
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("context_window_tokens must be a positive integer")
    return dict(value)


def tokenization_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    if parts.scheme not in {"http", "https"} or not parts.netloc or not parts.path.endswith("/v1"):
        raise ContextBudgetError("Native tokenization requires a /v1 API root")
    return urlunsplit((parts.scheme, parts.netloc, parts.path[:-3] + "/tokenize", "", ""))


def request_tokenization(payload: dict, *, base_url: str, api_key: str, timeout_seconds: float) -> str:
    """One native HTTP request, intentionally exposed for ledger/audit wrappers."""
    url = tokenization_endpoint(base_url)
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def apply_context_budget(payload: dict, policy: dict | None,
                         tokenize: Callable[[dict], str]) -> tuple[dict, dict | None]:
    policy = validate_context_budget(policy)
    if policy is None:
        return payload, None
    audit = {**policy, "status": "error", "usage": None, "tokenizer_attempts": 0,
             "requested_max_tokens": payload.get("max_tokens")}
    started = time.perf_counter()
    try:
        wire = {**payload, **payload.get("extra_body", {})}
        if (wire.get("truncate_prompt_tokens") is not None or "enable_thinking" in wire
                or wire.get("max_completion_tokens") is not None):
            raise ContextBudgetError("Use max_tokens, complete history and native chat_template_kwargs")
        maximum = wire.get("max_tokens")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ContextBudgetError("Context admission requires an explicit positive max_tokens")
        minimum = wire.get("min_tokens", 0)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ContextBudgetError("min_tokens must be a nonnegative integer")
        fields = ("model", "messages", "tools", "chat_template", "chat_template_kwargs",
                  "add_generation_prompt", "continue_final_message", "add_special_tokens")
        body = deepcopy({name: wire[name] for name in fields if name in wire})
        body.setdefault("add_generation_prompt", True)
        body.setdefault("continue_final_message", False)
        body.setdefault("add_special_tokens", False)
        audit["tokenizer_request_sha256"] = canonical_hash(body)
        audit["tokenizer_attempts"] = 1
        raw = tokenize(body)  # No implicit retries, including parser/transport failures.
        response = json.loads(raw)
        audit["tokenizer_response_sha256"] = canonical_hash(response)
        count, window, tokens = response.get("count"), response.get("max_model_len"), response.get("tokens")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or isinstance(window, bool) or not isinstance(window, int) or window <= 0
                or not isinstance(tokens, list) or count != len(tokens)
                or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens)):
            raise ContextBudgetError("Invalid native tokenization response")
        audit.update({"prompt_tokens": count, "server_max_model_len": window,
                      "token_ids_sha256": canonical_hash(tokens)})
        if window != policy["context_window_tokens"]:
            raise ContextBudgetError("Native context window differs from the declared window")
        effective = min(maximum, window - count)
        audit.update({"effective_max_tokens": effective, "output_budget_reduced": effective < maximum})
        if effective <= 0:
            raise ContextBudgetError("Complete prompt leaves no generation headroom")
        if minimum > effective:
            raise ContextBudgetError("Headroom is below the declared minimum generation budget")
        adjusted = {**payload, "max_tokens": effective}
        if "max_tokens" in payload.get("extra_body", {}):
            adjusted["extra_body"] = {**payload["extra_body"], "max_tokens": effective}
        audit.update({"status": "admitted", "effective_request_sha256": canonical_hash(adjusted)})
        return adjusted, audit
    except Exception as exc:
        audit["error_type"] = type(exc).__name__
        if getattr(exc, "retryable", True) is False:
            exc.context_budget_audit = audit
            raise
        raise ContextBudgetError("Native context admission failed", audit) from exc
    finally:
        audit["latency_seconds"] = time.perf_counter() - started


def verify_prompt_count(audit: dict | None, usage: dict | None) -> None:
    if audit is None:
        return
    actual = (usage or {}).get("prompt_tokens")
    audit["generation_prompt_count_matches"] = None if actual is None else actual == audit["prompt_tokens"]
    if actual is not None and actual != audit["prompt_tokens"]:
        audit.update({"status": "prompt_count_mismatch", "error_type": "ContextBudgetError"})
        raise ContextBudgetError("Generation prompt usage differs from native tokenization", audit)
