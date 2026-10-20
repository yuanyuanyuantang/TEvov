"""Small, standard-library helpers for auditable experiment provenance."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def canonical_hash(value: Any) -> str:
    """Hash JSON content independently of dictionary insertion order."""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Stable substream seed; never use Python's process-randomized hash()."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    return int(canonical_hash([base_seed, *parts])[:8], 16) % (2**31)


def seed_python(seed: int) -> dict[str, Any]:
    """Seed Python. PYTHONHASHSEED must be set before process startup."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    return {"python_seed": seed, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "remote_determinism_guaranteed": False}


def redact_secrets(value: Any) -> Any:
    """Redact credential fields recursively before storing configuration."""
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if _secret_key(str(k)) else redact_secrets(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(v) for v in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            host = parsed.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            if parsed.port:
                host += f":{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return value


def _secret_key(key: str) -> bool:
    key = key.lower().replace("-", "_")
    # The names of environment variables are reproducibility information.
    if key.endswith("_env"):
        return False
    return any(part in key for part in ("api_key", "password", "secret", "authorization")) or key in {
        "access_token", "refresh_token", "credential", "credentials"}


def validate_split(train_path: str | Path, val_path: str | Path) -> dict[str, Any]:
    """Reject overlap and duplicates instead of silently intersecting tasks."""
    def ids(path: str | Path) -> set[str]:
        seen: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                task_id = entry.get("id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(f"{path}:{line_number}: missing task id")
                if task_id in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate task {task_id}")
                seen.add(task_id)
        if not seen:
            raise ValueError(f"empty dataset: {path}")
        return seen

    train, val = ids(train_path), ids(val_path)
    overlap = train & val
    if overlap:
        raise ValueError(f"train/validation overlap: {sorted(overlap)}")
    return {"train_tasks": len(train), "validation_tasks": len(val), "overlap": 0,
            "train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path)}
