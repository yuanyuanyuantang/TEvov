"""Context admission uses rendered tokens and never silently truncates history."""

import json

import pytest

from tevo.config.shared_config import LLMConfig
from tevo.llm.client import DeepSeekLLMClient
from tevo.llm.context_budget import ContextBudgetError, apply_context_budget, tokenization_endpoint, validate_context_budget


POLICY = {"policy": "server_tokenize_headroom", "context_window_tokens": 32768}


def native(count, window=32768):
    return json.dumps({"count": count, "max_model_len": window, "tokens": [1] * count})


def test_native_off_by_one_uses_unchanged_tools_history_and_thinking():
    payload = {"model": "qwen", "max_tokens": 8192,
               "messages": [{"role": "user", "content": "完整历史"}],
               "tools": [{"type": "function", "function": {"name": "echo"}}],
               "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    captured = []
    adjusted, audit = apply_context_budget(payload, POLICY, lambda body: captured.append(body) or native(24577))
    assert adjusted["max_tokens"] == 8191 and payload["max_tokens"] == 8192
    assert adjusted["messages"] == payload["messages"] and adjusted["tools"] == payload["tools"]
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured[0]["add_generation_prompt"] is True and captured[0]["add_special_tokens"] is False
    assert audit["output_budget_reduced"] and audit["usage"] is None and audit["tokenizer_attempts"] == 1


@pytest.mark.parametrize("raw", [native(32768), native(10, 4096),
    '{"count":10,"max_model_len":32768,"tokens":[1]}',
    '{"count":true,"max_model_len":32768,"tokens":[1]}'])
def test_no_headroom_or_invalid_native_identity_fails_closed_once(raw):
    calls = []
    with pytest.raises(ContextBudgetError) as error:
        apply_context_budget({"model": "qwen", "max_tokens": 8192, "messages": []}, POLICY,
                             lambda body: calls.append(body) or raw)
    assert error.value.retryable is False and len(calls) == 1
    assert error.value.context_budget_audit["status"] == "error"


def test_client_transport_failure_does_not_generate_or_retry(tmp_path):
    client = DeepSeekLLMClient(LLMConfig(enabled=True, api_key="dummy", cache_enabled=False,
                              max_retries=99, context_budget=POLICY, cache_dir=str(tmp_path)))
    calls = []
    def failure(body):
        calls.append(body)
        raise TimeoutError("private transport detail")
    client._request_tokenization = failure
    client._request_with_retry = lambda body: pytest.fail("Generation issued after failed admission")
    with pytest.raises(ContextBudgetError, match="Native context admission failed"):
        client.chat("system", "goal")
    assert len(calls) == 1 and client.context_budget_metrics[0]["usage"] is None


def test_client_cache_identity_includes_policy_even_when_effective_limit_equal(tmp_path):
    config = LLMConfig(enabled=True, api_key="dummy", max_tokens=8192, context_budget=POLICY,
                       cache_dir=str(tmp_path))
    client = DeepSeekLLMClient(config)
    client._request_tokenization = lambda body: native(100, config.context_budget["context_window_tokens"])
    generations = []
    def generate(body):
        generations.append(body)
        return json.dumps({"choices": [{"message": {"content": "done"}}], "usage": {"prompt_tokens": 100}})
    client._request_with_retry = generate
    first = client.chat("system", "goal")
    client.chat("system", "goal")
    assert len(generations) == 1 and first["context_budget"]["generation_prompt_count_matches"] is True
    config.context_budget = {**POLICY, "context_window_tokens": 32769}
    client.chat("system", "goal")
    assert len(generations) == 2 and all(row["max_tokens"] == 8192 for row in generations)
    assert len(client.context_budget_metrics) == 3  # Cache admission is also an audited tokenizer request.


def test_prompt_usage_mismatch_retains_real_usage_and_rejects_result(tmp_path):
    client = DeepSeekLLMClient(LLMConfig(enabled=True, api_key="dummy", cache_enabled=False,
                              context_budget=POLICY, cache_dir=str(tmp_path)))
    client._request_tokenization = lambda body: native(100)
    client._request_with_retry = lambda body: json.dumps({"choices": [{"message": {"content": "done"}}],
                                                        "usage": {"prompt_tokens": 101}})
    with pytest.raises(ContextBudgetError, match="differs from native"):
        client.chat("system", "goal")
    assert client.context_budget_metrics[0]["generation_prompt_count_matches"] is False


@pytest.mark.parametrize("value", [{"policy": "estimate_chars", "context_window_tokens": 32768},
    {**POLICY, "context_window_tokens": True}, {**POLICY, "unknown": 1}])
def test_context_policy_schema_is_closed(value):
    with pytest.raises(ValueError):
        validate_context_budget(value)


def test_native_endpoint_preserves_proxy_prefix_and_rejects_non_v1_root():
    assert tokenization_endpoint("https://example.org/service/v1/") == "https://example.org/service/tokenize"
    with pytest.raises(ContextBudgetError):
        tokenization_endpoint("https://example.org/completions")


def test_alternative_budget_or_truncation_cannot_bypass_admission():
    payload = {"model": "qwen", "messages": [], "max_tokens": 8192}
    for extra in ({"max_completion_tokens": 10000}, {"truncate_prompt_tokens": 100}):
        with pytest.raises(ContextBudgetError):
            apply_context_budget({**payload, **extra}, POLICY, lambda body: pytest.fail("invalid request tokenized"))
    with pytest.raises(ContextBudgetError, match="minimum"):
        apply_context_budget({**payload, "min_tokens": 2}, POLICY, lambda body: native(32767))
