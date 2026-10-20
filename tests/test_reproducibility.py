import json
import random
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tevo.reproducibility import canonical_hash, derive_seed, redact_secrets, seed_python, validate_split
from tevo.config.shared_config import LLMConfig
from tevo.llm.client import DeepSeekLLMClient


def test_seed_and_hash_are_stable_without_dict_order_dependency():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert derive_seed(7, "task", 0) != derive_seed(7, "task", 1)
    seed_python(7)
    first = random.random()
    seed_python(7)
    assert first == random.random()
    with pytest.raises(ValueError):
        seed_python(-1)


def test_manifest_redacts_nested_credentials_but_keeps_usage():
    value = {"llm": {"api_key": "private", "api_key_env": "LLM_API_KEY"},
             "usage": {"total_tokens": 99}, "base_url": "https://user:pw@example.org/v1"}
    result = redact_secrets(value)
    assert result["llm"] == {"api_key": "[REDACTED]", "api_key_env": "LLM_API_KEY"}
    assert result["usage"]["total_tokens"] == 99
    assert result["base_url"] == "https://example.org/v1"
    assert value["llm"]["api_key"] == "private"


def test_split_validation_rejects_overlap_and_duplicates(tmp_path):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps({"id": "train_1"}) + "\n")
    val.write_text(json.dumps({"id": "val_1"}) + "\n")
    assert validate_split(train, val)["overlap"] == 0
    val.write_text(json.dumps({"id": "train_1"}) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        validate_split(train, val)
    val.write_text((json.dumps({"id": "val_1"}) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_split(train, val)


def test_client_preserves_seed_budget_and_thinking_in_cache_identity(tmp_path):
    config = LLMConfig(enabled=True, api_key="dummy", cache_dir=str(tmp_path), seed=17,
                       max_tokens=123, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    client = DeepSeekLLMClient(config)
    client._submit_chat_payload = lambda payload: payload
    plain = client.chat("system", "goal")
    tools = client.chat_with_tools(system_prompt="system", user_prompt="goal", tools=[])
    for payload in (plain, tools):
        assert payload["seed"] == 17
        assert payload["max_tokens"] == 123
        assert payload["chat_template_kwargs"]["enable_thinking"] is True
    changed = {**plain, "seed": 18}
    assert client._cache_key(plain) != client._cache_key(changed)
    with pytest.raises(ValueError, match="max_tokens"):
        LLMConfig(max_tokens=0)


def test_actual_http_client_can_construct_under_socks_proxy_without_network(monkeypatch):
    from openai import OpenAI
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    with OpenAI(api_key="dummy", base_url="http://127.0.0.1:1/v1", max_retries=0) as client:
        assert str(client.base_url) == "http://127.0.0.1:1/v1/"


@pytest.mark.parametrize("override", [{"timeout_seconds": float("nan")},
                                    {"timeout_seconds": float("inf")},
                                    {"retry_backoff_seconds": float("inf")},
                                    {"max_retries": 1.5}, {"max_retries": True}])
def test_llm_rejects_unbounded_or_invalid_retry_configuration(override):
    with pytest.raises(ValueError):
        LLMConfig(**override)


def test_importing_current_method_preserves_callers_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", "import os; before=os.getcwd(); from tevo.memory import parameter_compiler; from tevo.experiment import suite; assert os.getcwd()==before"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
