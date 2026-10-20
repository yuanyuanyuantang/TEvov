"""Embedding helpers used by ToolRAG-Evo memory and retrieval."""

from __future__ import annotations

import json
import math
import ssl
import time
from typing import Protocol
from urllib import error, request

from tevo.config.shared_config import MemoryEmbeddingConfig


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [round(value / norm, 8) for value in vector]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two dense vectors."""

    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same length: {len(vec1)} != {len(vec2)}")
    dot_product = sum(left * right for left, right in zip(vec1, vec2))
    norm1 = math.sqrt(sum(value * value for value in vec1))
    norm2 = math.sqrt(sum(value * value for value in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


class EmbeddingModelProtocol(Protocol):
    """Interface required by ToolRAG-Evo indices and memory stores."""

    backend_name: str
    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Embed one text string."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""


class OpenAICompatibleEmbeddingModel:
    """Use an OpenAI-compatible `/embeddings` endpoint for indexing and retrieval."""

    backend_name = "openai-compatible"

    def __init__(self, config: MemoryEmbeddingConfig) -> None:
        if not config.base_url.strip():
            raise ValueError("memory embedding base_url is required")
        if not config.model.strip():
            raise ValueError("memory embedding model is required")
        self.config = config
        self.model_name = config.model
        self.dimension = max(int(config.dimensions), 0)

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = max(int(self.config.batch_size), 1)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = self._build_payload(batch)
            raw = self._request_with_retry(payload)
            vectors.extend(self._parse_embeddings(raw, expected_count=len(batch)))
        return vectors

    def _build_payload(self, batch: list[str]) -> dict[str, object]:
        base_url = self.config.base_url.lower()
        if "minimaxi.com" in base_url:
            return {
                "model": self.config.model,
                "texts": batch,
                "type": "db",
            }
        payload: dict[str, object] = {
            "model": self.config.model,
            "input": batch,
            "encoding_format": self.config.encoding_format,
        }
        if self.config.dimensions > 0:
            payload["dimensions"] = int(self.config.dimensions)
        return payload

    def _parse_embeddings(self, raw: str, expected_count: int) -> list[list[float]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"memory embedding response parse failed: {exc}") from exc

        vectors_payload = payload.get("vectors")
        if isinstance(vectors_payload, list):
            vectors: list[list[float]] = []
            for item in vectors_payload:
                if not isinstance(item, list) or not item:
                    raise RuntimeError("memory embedding response contained an empty vector")
                vector = _normalize_vector([float(value) for value in item])
                if self.dimension == 0:
                    self.dimension = len(vector)
                elif len(vector) != self.dimension:
                    raise RuntimeError(
                        "memory embedding dimension mismatch: "
                        f"expected {self.dimension}, got {len(vector)} from model {self.config.model}"
                    )
                vectors.append(vector)
            if len(vectors) != expected_count:
                raise RuntimeError(
                    "memory embedding response size mismatch: "
                    f"expected {expected_count}, got {len(vectors)}"
                )
            return vectors

        base_resp = payload.get("base_resp")
        if isinstance(base_resp, dict):
            status_code = int(base_resp.get("status_code", 0) or 0)
            status_msg = str(base_resp.get("status_msg", "")).strip()
            if status_code != 0:
                raise RuntimeError(
                    "memory embedding provider returned an error: "
                    f"{status_code} {status_msg}".strip()
                )

        data = payload.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("memory embedding response missing `data` list")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for item in ordered:
            if not isinstance(item, dict):
                raise RuntimeError("memory embedding response contained a non-dict item")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("memory embedding response contained an empty embedding")
            vector = _normalize_vector([float(value) for value in embedding])
            if self.dimension == 0:
                self.dimension = len(vector)
            elif len(vector) != self.dimension:
                raise RuntimeError(
                    "memory embedding dimension mismatch: "
                    f"expected {self.dimension}, got {len(vector)} from model {self.config.model}"
                )
            vectors.append(vector)
        if len(vectors) != expected_count:
            raise RuntimeError(
                "memory embedding response size mismatch: "
                f"expected {expected_count}, got {len(vectors)}"
            )
        return vectors

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if self.config.resolved_insecure_skip_verify():
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        ca_bundle = self.config.resolved_ca_bundle()
        if ca_bundle:
            return ssl.create_default_context(cafile=ca_bundle)
        return None

    def _build_request(self, payload: dict[str, object], api_key: str) -> request.Request:
        target_url = f"{self.config.base_url.rstrip('/')}/embeddings"
        return request.Request(
            target_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

    def _request_with_retry(self, payload: dict[str, object]) -> str:
        current_api_key = self.config.resolved_api_key()
        if not current_api_key:
            raise RuntimeError(
                "memory embedding request failed: no API key found. "
                f"Pass `--memory-embedding-api-key`, set `{self.config.api_key_env}`, "
                "or reuse the chat API key/env via the benchmark CLI."
            )

        ssl_context = self._build_ssl_context()
        last_error: Exception | None = None
        active_payload = dict(payload)
        for attempt in range(self.config.max_retries + 1):
            try:
                req = self._build_request(active_payload, current_api_key)
                with request.urlopen(req, timeout=self.config.timeout_seconds, context=ssl_context) as response:
                    return response.read().decode("utf-8")
            except error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="ignore")
                if (
                    exc.code == 400
                    and "matryoshka" in message.lower()
                    and "dimensions" in active_payload
                ):
                    active_payload = dict(active_payload)
                    active_payload.pop("dimensions", None)
                    last_error = RuntimeError(
                        "memory embedding provider rejected explicit dimensions; "
                        "retrying without `dimensions`."
                    )
                    continue
                if exc.code in {401, 403}:
                    refreshed_key = self.config.resolved_api_key()
                    if refreshed_key and refreshed_key != current_api_key:
                        current_api_key = refreshed_key
                last_error = RuntimeError(f"memory embedding HTTP error: {exc.code} {message}")
            except Exception as exc:  # noqa: BLE001
                last_error = RuntimeError(f"memory embedding request failed: {exc}")
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_seconds * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("memory embedding request failed without explicit error")
