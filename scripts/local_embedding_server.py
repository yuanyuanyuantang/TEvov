"""Project-local OpenAI-compatible embedding server."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import socket
import threading
import site
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from pathlib import Path



DEFAULT_MODEL_NAME = "Qwen3-Embedding-0.6B"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3-Embedding-0.6B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8013
DEFAULT_DEVICE = "cuda"
DEFAULT_PRELOAD = True
DEFAULT_BACKEND = "sentence_transformers"

_embedding_model: Any | None = None
_embedding_model_path = ""
_embedding_model_lock = threading.Lock()
_runtime_prepared = False


def _load_embedding_model() -> Any:
    global _embedding_model, _embedding_model_path

    backend = _configured_backend()
    if backend == "hash":
        return None
    model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.getenv("LOCAL_EMBEDDING_DEVICE", DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
    if _embedding_model is not None and _embedding_model_path == model_path:
        return _embedding_model
    with _embedding_model_lock:
        if _embedding_model is None or _embedding_model_path != model_path:
            _prepare_local_torch_runtime()
            print(
                f"Importing sentence_transformers backend={backend} "
                f"model_path={model_path} device={device}...",
                flush=True,
            )
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model from {model_path} on device={device}...", flush=True)
            _embedding_model = SentenceTransformer(str(model_path), device=device)
            _embedding_model_path = model_path
            print("Embedding model loaded successfully.", flush=True)
    return _embedding_model


def _configured_model_name() -> str:
    return os.getenv("LOCAL_EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)


def _configured_backend() -> str:
    backend = os.getenv("LOCAL_EMBEDDING_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend in {"hash", "hashed", "hashing"}:
        return "hash"
    return "sentence_transformers"


def _prepare_local_torch_runtime() -> None:
    global _runtime_prepared

    if _runtime_prepared:
        return

    _prefer_conda_site_packages()
    _preload_conda_nvidia_libraries()
    _runtime_prepared = True


def _prefer_conda_site_packages() -> None:
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return
    conda_site = Path(conda_prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    conda_site_str = str(conda_site)
    user_site = site.getusersitepackages()
    user_site_path = Path(user_site)
    if conda_site_str in sys.path:
        sys.path.remove(conda_site_str)
        sys.path.insert(0, conda_site_str)
    elif conda_site.exists():
        sys.path.insert(0, conda_site_str)

    # Keep user-site packages available for libraries such as sentence_transformers,
    # but always place them after the conda environment so torch resolves from conda first.
    if user_site in sys.path:
        sys.path.remove(user_site)
        sys.path.append(user_site)
    elif user_site_path.exists():
        sys.path.append(user_site)


def _preload_conda_nvidia_libraries() -> None:
    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return
    nvidia_root = Path(conda_prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia"
    if not nvidia_root.exists():
        return

    preferred_libraries = [
        ("cuda_runtime", "libcudart.so"),
        ("cuda_nvrtc", "libnvrtc.so"),
        ("nvjitlink", "libnvJitLink.so"),
        ("cublas", "libcublas.so"),
        ("cublas", "libcublasLt.so"),
        ("cusparse", "libcusparse.so"),
        ("cusparselt", "libcusparseLt.so"),
        ("cufft", "libcufft.so"),
        ("curand", "libcurand.so"),
        ("cusolver", "libcusolver.so"),
        ("cudnn", "libcudnn.so"),
        ("nccl", "libnccl.so"),
    ]
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    for package_name, library_prefix in preferred_libraries:
        lib_dir = nvidia_root / package_name / "lib"
        if not lib_dir.exists():
            continue
        matches = sorted(lib_dir.glob(f"{library_prefix}*"))
        if not matches:
            continue
        try:
            ctypes.CDLL(str(matches[0]), mode=mode)
        except OSError:
            continue


def _is_model_loaded() -> bool:
    if _configured_backend() == "hash":
        return True
    model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH)
    return _embedding_model is not None and _embedding_model_path == model_path


def _resolve_request_model(payload: dict[str, Any]) -> str:
    requested = payload.get("model")
    configured = _configured_model_name()
    if requested is None or str(requested).strip() == "":
        return configured
    requested_name = str(requested).strip()
    if requested_name != configured:
        raise ValueError(
            f"requested model `{requested_name}` does not match configured local model `{configured}`"
        )
    return configured


def _coerce_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("`input` must be a string or a list of strings")


def _hash_embedding(text: str, *, dimensions: int = 1024) -> list[float]:
    vector = [0.0] * dimensions
    tokens = []
    current = []
    for ch in str(text or "").lower():
        if ch.isalnum() or ch in {"_", "-", "/", "."}:
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    if not tokens:
        tokens = [str(text or "empty")]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        for offset in range(0, len(digest), 4):
            idx = int.from_bytes(digest[offset:offset + 2], "little") % dimensions
            sign = 1.0 if digest[offset + 2] % 2 == 0 else -1.0
            vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def _embed_inputs(inputs: list[str]) -> list[list[float]]:
    if _configured_backend() == "hash":
        return [_hash_embedding(text) for text in inputs]
    model = _load_embedding_model()
    embeddings = model.encode(inputs)
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return [[float(value) for value in embedding] for embedding in embeddings]


class EmbeddingHandler(BaseHTTPRequestHandler):
    server_version = "MCTPOLocalEmbedding/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": _configured_model_name(),
                    "backend": _configured_backend(),
                    "model_path": str(os.getenv("LOCAL_EMBEDDING_MODEL_PATH", DEFAULT_MODEL_PATH)),
                    "device": os.getenv("LOCAL_EMBEDDING_DEVICE", DEFAULT_DEVICE).strip() or DEFAULT_DEVICE,
                    "model_loaded": _is_model_loaded(),
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path: {self.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = _coerce_inputs(payload.get("input"))
            model_name = _resolve_request_model(payload)
            embeddings = _embed_inputs(inputs)
            data = [
                {
                    "object": "embedding",
                    "embedding": [float(value) for value in embedding],
                    "index": index,
                }
                for index, embedding in enumerate(embeddings)
            ]
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": data,
                    "model": model_name,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            # The client already gave up waiting; avoid converting that into noisy 500 logs.
            return


def main() -> None:
    host = os.getenv("LOCAL_EMBEDDING_HOST", DEFAULT_HOST)
    port = int(os.getenv("LOCAL_EMBEDDING_PORT", str(DEFAULT_PORT)))
    preload_flag = os.getenv("LOCAL_EMBEDDING_PRELOAD", "1" if DEFAULT_PRELOAD else "0").strip().lower()
    print(
        "Starting local embedding server "
        f"backend={_configured_backend()} model={_configured_model_name()} "
        f"device={os.getenv('LOCAL_EMBEDDING_DEVICE', DEFAULT_DEVICE)} "
        f"port={port}",
        flush=True,
    )
    if preload_flag in {"1", "true", "yes", "on"}:
        _load_embedding_model()
    server = ThreadingHTTPServer((host, port), EmbeddingHandler)
    print(f"Serving local embeddings at http://{host}:{port}/v1/embeddings", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local embedding server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
