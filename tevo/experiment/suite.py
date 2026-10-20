"""Configuration-driven frozen-bank BFCL experiments, without a Ray cluster.

Each trial runs in an isolated spawned process and is saved immediately.  A
SQLite ledger bounds requests across workers.  Historical outputs are never
imported into the new-run directory or counted as new-version replication.
"""

from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import json
import math
import multiprocessing
import os
import platform
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from tevo.reproducibility import canonical_hash, derive_seed, redact_secrets, sha256_file, validate_split
from tevo.llm.context_budget import validate_context_budget

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"nomem", "memonly", "profiles", "both"}
PROFILE_FIELDS = {"when_to_use", "preconditions", "input_requirements", "expected_outcomes", "failure_cues", "recovery_hints"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resolve_path(value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unset environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(row)
    return rows


def load_plan(config_path: str | Path, only: list[str] | None = None) -> dict[str, Any]:
    """Validate all inputs without importing BFCL/Ray or contacting services."""
    def expand(value):
        if isinstance(value, dict):
            return {name: expand(item) for name, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            result = os.path.expandvars(value)
            if "$" in result:
                raise ValueError(f"Unset configuration environment variable: {value}")
            return result
        return value
    config = expand(json.loads(Path(config_path).read_text(encoding="utf-8")))
    schemas = {
        "configuration": {"version", "output_dir", "dataset", "model", "seeds", "budgets", "frozen", "experiments", "analysis_baseline", "repeats_per_seed", "sampling_seed_mode"},
        "dataset": {"train_path", "path", "answer_path", "task_ids", "max_tasks"},
        "model": {"name", "path", "base_urls", "api_key_env", "temperature", "enable_thinking", "max_tokens", "max_interactions", "max_retries", "timeout_seconds", "artifact_manifest_path", "context_budget"},
        "budgets": {"max_workers", "max_requests", "max_total_tokens", "wall_seconds"},
        "frozen": {"registry_path", "memories_path", "index_path", "reme_bank_path", "top_k", "embedding", "reme_embedding", "reme_adapter_verified", "reme_contexts_path", "per_turn_reference", "retrieval_query_mode", "retrieval_tool_filter"},
        "experiment": {"name", "method", "category", "tool_scope", "temperature", "max_interactions", "profile_fields", "profile_max_chars", "profile_budget_reallocation", "acquisition_budget_matched", "per_turn_memory"},
        "embedding": {"base_url", "model", "dimensions", "batch_size", "timeout_seconds", "api_key_env", "usage_is_measured", "send_dimensions"},
        "per_turn_reference": {"pilot_contexts_path", "diagnostic_report_path"},
    }
    def reject_unknown(value, label):
        unknown = set(value) - schemas[label]
        if unknown:
            raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")
    reject_unknown(config, "configuration")
    for section in ("dataset", "model", "budgets", "frozen"):
        reject_unknown(config.get(section, {}), section)
    for arm in config.get("experiments", []):
        reject_unknown(arm, "experiment")
    for section in ("embedding", "reme_embedding"):
        if section in config.get("frozen", {}):
            reject_unknown(config["frozen"][section], "embedding")
    if "per_turn_reference" in config.get("frozen", {}):
        reject_unknown(config["frozen"]["per_turn_reference"], "per_turn_reference")
    if config.get("version") != 1:
        raise ValueError("experiment configuration version must be 1")
    dataset = config["dataset"]
    dataset_path = resolve_path(dataset["path"])
    split = validate_split(resolve_path(dataset["train_path"]), dataset_path)
    answers = resolve_path(dataset["answer_path"])
    if not answers.is_dir() or not list(answers.glob("*.json")):
        raise ValueError("answer_path must contain the pinned BFCL answer files")
    rows = load_jsonl(dataset_path)
    ids = [row["id"] for row in rows]
    selected = dataset.get("task_ids", ids)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(ids):
        raise ValueError("task_ids must be distinct existing validation IDs")
    maximum = dataset.get("max_tasks", 0)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
        raise ValueError("max_tasks must be a nonnegative integer")
    selected = selected[:maximum] if maximum else selected
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be distinct integers in [0, 2**32)")
    repeats = config.get("repeats_per_seed", 1)
    seed_mode = config.get("sampling_seed_mode", "task_derived")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats_per_seed must be a positive integer")
    if seed_mode not in {"task_derived", "fixed"}:
        raise ValueError("sampling_seed_mode must be task_derived or fixed")
    if seed_mode == "fixed" and len(seeds) != 1:
        raise ValueError("fixed sampling requires one declared seed")
    if repeats != 1 and seed_mode != "fixed":
        raise ValueError("repeated trials require explicit fixed sampling_seed_mode")
    model = config["model"]
    artifact_path = model.get("artifact_manifest_path")
    if artifact_path:
        artifact_path = resolve_path(artifact_path)
        artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact_manifest.get("model_id") != model.get("name"):
            raise ValueError("Model artifact manifest model_id does not match the requested model")
    if "enable_thinking" in model and not isinstance(model["enable_thinking"], bool):
        raise ValueError("model.enable_thinking must be a JSON boolean")
    if not model.get("base_urls") or any(not u.startswith(("http://", "https://")) for u in model["base_urls"]):
        raise ValueError("model.base_urls must contain HTTP API roots")
    if not isinstance(model.get("max_tokens"), int) or isinstance(model["max_tokens"], bool) or model["max_tokens"] <= 0:
        raise ValueError("model.max_tokens must be explicit and positive")
    validate_context_budget(model.get("context_budget"))
    if not 0 <= float(model["temperature"]) <= 2 or float(model.get("timeout_seconds", 120)) <= 0:
        raise ValueError("invalid model temperature/timeout")
    budgets = config["budgets"]
    for name in ("max_workers", "max_requests", "max_total_tokens", "wall_seconds"):
        if not isinstance(budgets.get(name), int) or isinstance(budgets[name], bool) or budgets[name] <= 0:
            raise ValueError(f"budgets.{name} must be a positive integer")
    experiments = config["experiments"]
    names = [arm["name"] for arm in experiments]
    if not names or len(set(names)) != len(names):
        raise ValueError("experiment names must be nonempty and distinct")
    if only and set(only) - set(names):
        raise ValueError(f"Unknown selected experiments: {sorted(set(only) - set(names))}")
    jobs = []
    chosen = []
    for arm in experiments:
        if only and arm["name"] not in only:
            continue
        if not arm["name"] or arm.get("method") not in METHODS:
            raise ValueError(f"invalid experiment: {arm}")
        if arm.get("tool_scope", "task") not in {"task", "all"}:
            raise ValueError("tool_scope must be task or all")
        if not 0 <= float(arm.get("temperature", model["temperature"])) <= 2:
            raise ValueError("invalid experiment temperature")
        if "profile_fields" in arm and (not arm["profile_fields"] or set(arm["profile_fields"]) - PROFILE_FIELDS):
            raise ValueError("profile_fields must select existing profile fields")
        if int(arm.get("profile_max_chars", 400)) <= 0:
            raise ValueError("profile_max_chars must be positive")
        if int(arm.get("max_interactions", model.get("max_interactions", 30))) <= 0:
            raise ValueError("max_interactions must be positive")
        if "per_turn_memory" in arm and not isinstance(arm["per_turn_memory"], bool):
            raise ValueError("per_turn_memory must be a JSON boolean")
        if arm.get("per_turn_memory") and arm.get("method") not in {"memonly", "both"}:
            raise ValueError("per_turn_memory requires a memory-reading method")
        chosen.append(arm)
        for task_id in selected:
            for seed_index, base_seed in enumerate(seeds):
                for repeat_index in range(repeats):
                    identity = [arm["name"], task_id, base_seed]
                    if repeats > 1:
                        identity.append(repeat_index)
                    job = {"key": canonical_hash(identity)[:24], "experiment": arm["name"],
                           "task_id": task_id, "base_seed": base_seed,
                           "trial_index": seed_index * repeats + repeat_index,
                           "trial_seed": base_seed if seed_mode == "fixed" else derive_seed(base_seed, task_id), "arm": arm}
                    if repeats > 1:
                        job["repeat_index"] = repeat_index
                    jobs.append(job)
    frozen = config.get("frozen", {})
    query_mode = frozen.get("retrieval_query_mode", "cumulative_user")
    if not isinstance(frozen.get("retrieval_tool_filter", False), bool):
        raise ValueError("retrieval_tool_filter must be boolean")
    if frozen.get("retrieval_tool_filter") and frozen.get("per_turn_reference"):
        raise ValueError("task-tool filtering cannot reuse legacy per_turn_reference")
    if query_mode not in {"cumulative_user", "current_user"}:
        raise ValueError("retrieval_query_mode must be cumulative_user or current_user")
    if query_mode == "current_user" and not any(a.get("per_turn_memory") for a in chosen):
        raise ValueError("current_user retrieval requires a per_turn_memory arm")
    if query_mode == "current_user" and frozen.get("per_turn_reference"):
        raise ValueError("current_user retrieval cannot use cumulative per_turn_reference artifacts")
    if int(frozen.get("top_k", 5)) <= 0:
        raise ValueError("frozen.top_k must be positive")
    methods = {arm["method"] for arm in chosen}
    required = set()
    if methods & {"profiles", "both"}:
        required.add("registry_path")
    if methods & {"memonly", "both"}:
        required.update(("memories_path", "index_path"))
    if "reme_public96" in methods:
        required.add("reme_bank_path")
        if frozen.get("reme_contexts_path"):
            required.add("reme_contexts_path")
    bank_paths = {name: str(resolve_path(frozen[name])) for name in sorted(required)}
    for path in bank_paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if methods & {"memonly", "both"}:
        if not frozen.get("embedding", {}).get("base_url"):
            raise ValueError("frozen.embedding is required for local read-only retrieval")
    if "reme_public96" in methods and not frozen.get("reme_contexts_path") and not frozen.get("reme_embedding", {}).get("base_url"):
        raise ValueError("frozen.reme_embedding must record the official ReMe embedding endpoint")
    for section in ("embedding", "reme_embedding"):
        if section in frozen:
            settings = frozen[section]
            if int(settings.get("batch_size", 32)) <= 0 or int(settings.get("dimensions", 0)) < 0 or float(settings.get("timeout_seconds", 120)) <= 0:
                raise ValueError(f"Invalid {section} batch size, dimensions or timeout")
    # Per-turn arms re-query the frozen bank at every user turn; the frozen
    # turn texts and the reference artifacts must validate on CPU before any
    # request, so a divergent dataset or reference aborts here.
    per_turn_reference = {}
    per_turn_turns: dict[str, list[str]] = {}
    if any(arm.get("per_turn_memory") for arm in chosen):
        from tevo.experiment.per_turn import extract_user_turns, validate_turn_identity
        reference = frozen.get("per_turn_reference") or {}
        for key in (() if query_mode == "current_user" else ("pilot_contexts_path", "diagnostic_report_path")):
            value = reference.get(key)
            if not value:
                raise ValueError(f"per_turn_memory requires frozen.per_turn_reference.{key}")
            resolved = str(resolve_path(value))
            if not Path(resolved).is_file():
                raise FileNotFoundError(resolved)
            per_turn_reference[key] = resolved
        by_id = {row["id"]: row for row in rows}
        for task_id in selected:
            row = by_id[task_id]
            messages = row.get("messages") or []
            first = messages[0].get("content") if messages and isinstance(messages[0], dict) else None
            if not isinstance(first, str):
                raise ValueError(f"per_turn_memory requires a first user message: {task_id}")
            turns = extract_user_turns((row.get("extra") or {}).get("question"))
            validate_turn_identity(first, turns, task_id)
            per_turn_turns[task_id] = turns
    # Paired arms advance together so interruptions leave explicit partial
    # coverage, rather than a completed baseline and absent intervention.
    order = {task: index for index, task in enumerate(selected)}
    arm_order = {arm["name"]: index for index, arm in enumerate(chosen)}
    jobs.sort(key=lambda job: (order[job["task_id"]], job["trial_index"], arm_order[job["experiment"]]))
    for job in jobs:
        # Paired arms of the same (task, seed) share one endpoint (common
        # random numbers on the paired comparison); balance is kept across
        # tasks. The arm position no longer shifts the endpoint.
        endpoint = (order[job["task_id"]] + job["trial_index"]) % len(model["base_urls"])
        job["endpoint_index"] = endpoint
        job["llm_base_url"] = model["base_urls"][endpoint]
    snapshot_verified = False
    if "reme_contexts_path" in bank_paths:
        validate_reme_snapshot(Path(bank_paths["reme_contexts_path"]), Path(bank_paths["reme_bank_path"]), dataset_path,
                               {row["id"]: row for row in rows if row["id"] in selected})
        snapshot_verified = True
    return {"config": config, "jobs": jobs, "experiments": chosen, "task_ids": selected,
            "dataset_path": str(dataset_path), "answer_path": str(answers), "split": split,
            "bank_paths": bank_paths, "reme_snapshot_verified": snapshot_verified,
            "per_turn_turns": per_turn_turns, "per_turn_reference": per_turn_reference}


def validate_reme_snapshot(path: Path, bank_path: Path, dataset_path: Path, tasks: dict) -> dict:
    """Require official response evidence tied to this bank, dataset and queries."""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("backend") != "official_readonly_service_snapshot" or snapshot.get("verified") is not True:
        raise ValueError("ReMe snapshot must come from a verified official read-only service")
    verification = snapshot.get("bank_verification", {})
    if verification.get("verified") is not True or verification.get("records") != 96 or not snapshot.get("source_sha256"):
        raise ValueError("ReMe snapshot lacks complete public-bank/source identity evidence")
    if snapshot.get("public_bank_sha256") != sha256_file(bank_path) or snapshot.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("ReMe snapshot bank/dataset fingerprint mismatch")
    if snapshot.get("top_k") != 5:
        raise ValueError("ReMe snapshot must retain the official top_k=5 protocol")
    bank = {record["memory_id"]: record for record in load_jsonl(bank_path)}
    for task_id, task in tasks.items():
        context = snapshot.get("contexts", {}).get(task_id)
        if context is None or context.get("query_sha256") != canonical_hash(task["messages"][0]["content"]):
            raise ValueError(f"ReMe snapshot query mismatch/missing task: {task_id}")
        memories = context.get("memories", [])
        if len(memories) > 5:
            raise ValueError("ReMe snapshot exceeds the official memory budget")
        for memory in memories:
            expected = bank.get(memory.get("memory_id"))
            if expected is None or memory.get("content") != expected["content"] or memory.get("when_to_use") != expected["when_to_use"]:
                raise ValueError("ReMe snapshot contains altered/unknown public-bank content")
        response = {"answer": context["answer"], "success": True, "metadata": {"memory_list": memories}}
        if canonical_hash(response) != context.get("response_sha256"):
            raise ValueError("ReMe official response hash mismatch")
    return snapshot


def fingerprints(plan: dict[str, Any]) -> dict[str, Any]:
    """Include method/runner/checker sources, data, answers and frozen banks."""
    paths = list((ROOT / "tevo").rglob("*.py")) + list((ROOT / "benchmark/bfcl").glob("*.py"))
    paths += [ROOT / "scripts/run_experiment_suite.py"]
    vendor = ROOT / "benchmark/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval"
    paths += list(vendor.rglob("*.py"))
    code = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(set(paths)) if p.is_file()}
    artifact_path = plan["config"]["model"].get("artifact_manifest_path")
    artifact = {"manifest_path": str(resolve_path(artifact_path)), "manifest_sha256": sha256_file(resolve_path(artifact_path))} if artifact_path else None
    return {"config": canonical_hash(redact_secrets(plan["config"])),
            "selected_experiments": canonical_hash(plan["experiments"]),
            "jobs": canonical_hash([{k: v for k, v in job.items() if k != "arm"} for job in plan["jobs"]]),
            "code": canonical_hash(code), "code_files": code, "split": plan["split"],
            "answers": {p.name: sha256_file(p) for p in sorted(Path(plan["answer_path"]).glob("*.json"))},
            "banks": {name: sha256_file(path) for name, path in plan["bank_paths"].items()},
            "model_artifacts": artifact}


def inspect_providers(model: dict) -> dict:
    """Read advertised model identity and context limits before inference."""
    import requests
    providers = []
    for base_url in model["base_urls"]:
        response = requests.get(base_url.rstrip("/") + "/models", timeout=10,
                                headers={"Authorization": "Bearer " + os.environ.get(model.get("api_key_env", "TEVO_LLM_API_KEY"), "dummy")})
        response.raise_for_status()
        advertised = response.json().get("data", [])
        matching = [entry for entry in advertised if entry.get("id") == model["name"]]
        if len(matching) != 1:
            raise ValueError(f"Requested model is not uniquely advertised by {base_url}: {model['name']}")
        entry = matching[0]
        maximum = entry.get("max_model_len")
        if maximum is not None and model["max_tokens"] > int(maximum):
            raise ValueError("Requested completion budget exceeds the advertised context length")
        policy = validate_context_budget(model.get("context_budget"))
        if policy is not None and maximum != policy["context_window_tokens"]:
            raise ValueError("Declared context budget differs from the advertised context length")
        providers.append({"base_url": base_url, "model_id": entry["id"], "model_root": entry.get("root"),
                          "max_model_len": maximum, "context_budget": policy})
    return {"providers": providers, "seed_is_sent": True, "server_determinism_verified": False,
            "thinking_requested": bool(model.get("enable_thinking", True)),
            "thinking_request_body": {"chat_template_kwargs": {"enable_thinking": bool(model.get("enable_thinking", True))}},
            "advertised_metadata_does_not_identify_model_weight_hash": True}


class BudgetExceeded(RuntimeError):
    retryable = False


class RequestLedger:
    """Reserve requests atomically; reported-token limits stop subsequent work.

    Token usage is only known after responses. Concurrent in-flight requests can
    exceed the reported-token limit; this is explicitly recorded, never called a
    hard token cap. Missing provider usage remains unknown, rather than zero.
    """

    def __init__(self, path: Path, budgets: dict[str, Any]):
        self.path, self.budgets = path, budgets
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL)")
            connection.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, trial_key TEXT, kind TEXT, started REAL, finished REAL, tokens INTEGER, status TEXT)")
            connection.execute("INSERT OR IGNORE INTO meta VALUES ('started', ?)", (time.time(),))

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.path), timeout=60)
        try:
            with connection:
                yield connection
        finally:
            # SQLite's transaction context commits/rolls back but does not close.
            connection.close()

    def reserve(self, key: str, kind: str) -> int:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count, tokens = connection.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM requests").fetchone()
            started = connection.execute("SELECT value FROM meta WHERE key='started'").fetchone()[0]
            if count >= self.budgets["max_requests"] or tokens >= self.budgets["max_total_tokens"] or time.time() - started >= self.budgets["wall_seconds"]:
                raise BudgetExceeded("Experiment request/token/wall budget reached")
            cursor = connection.execute("INSERT INTO requests(trial_key,kind,started,status) VALUES (?,?,?,'in_flight')", (key, kind, time.time()))
            return cursor.lastrowid

    def finish(self, request_id: int, usage: dict | None, status: str) -> None:
        tokens = usage.get("total_tokens") if usage else None
        if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0):
            raise ValueError("invalid provider total_tokens")
        with self.connect() as connection:
            connection.execute("UPDATE requests SET finished=?, tokens=?, status=? WHERE id=?", (time.time(), tokens, status, request_id))

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            count, known_tokens, missing = connection.execute("SELECT COUNT(*), COALESCE(SUM(tokens),0), SUM(tokens IS NULL) FROM requests").fetchone()
            started = connection.execute("SELECT value FROM meta WHERE key='started'").fetchone()[0]
        return {"requests": count, "reported_total_tokens": known_tokens,
                "requests_with_unknown_token_usage": missing or 0, "wall_seconds": time.time() - started,
                "token_budget_is_hard_cap": False, "budgets": self.budgets}


def _embedding(texts: list[str], settings: dict, ledger: RequestLedger, key: str) -> tuple[list[list[float]], list[dict]]:
    import requests
    from tevo.memory.embedding import _normalize_vector
    vectors, events = [], []
    for start in range(0, len(texts), int(settings.get("batch_size", 32))):
        batch = texts[start:start + int(settings.get("batch_size", 32))]
        payload = {"model": settings["model"], "input": batch, "encoding_format": "float"}
        if settings.get("dimensions", 0) and settings.get("send_dimensions", True):
            payload["dimensions"] = settings["dimensions"]
        request_id = ledger.reserve(key, "embedding")
        started = time.monotonic()
        usage = None
        try:
            response = requests.post(settings["base_url"].rstrip("/") + "/embeddings", json=payload,
                                     headers={"Authorization": "Bearer " + os.environ.get(settings.get("api_key_env", "TEVO_LLM_API_KEY"), "dummy")},
                                     timeout=float(settings.get("timeout_seconds", 120)))
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") if settings.get("usage_is_measured", True) else None
            rows = sorted(data["data"], key=lambda row: row["index"])
            if len(rows) != len(batch) or [row["index"] for row in rows] != list(range(len(batch))):
                raise ValueError("embedding response size/index mismatch")
            received = [_normalize_vector([float(v) for v in row["embedding"]]) for row in rows]
            dimension = int(settings.get("dimensions", 0))
            if any(not vector or (dimension and len(vector) != dimension) or not all(math.isfinite(v) for v in vector) for vector in received):
                raise ValueError("embedding response has invalid vectors")
            ledger.finish(request_id, usage, "success")
            vectors.extend(received)
            events.append({"request_sha256": canonical_hash(payload), "response_sha256": canonical_hash(data),
                           "usage": usage, "latency_seconds": time.monotonic() - started,
                           "model": data.get("model", settings["model"]), "batch_size": len(batch)})
        except Exception:
            tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            reported = usage if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None
            ledger.finish(request_id, reported, "error")
            raise
    return vectors, events


def validate_frozen_bank_index(records: list[dict], index: object) -> None:
    """Fail closed on records/index divergence before retrieval.

    Mirrors ProceduralMemoryStore._load: equal id sets, index search_text equal to
    the record's search_text, non-empty finite vectors with consistent dimensions.
    A stale v5 index against v6 records must abort instead of silently scoring a
    different bank generation.
    """
    from tevo.memory.procedural_schema import MemoryRecord
    if not isinstance(index, dict):
        raise ValueError("frozen memory bank index must be an object")
    record_ids = [record.get("record_id") for record in records]
    if not all(isinstance(record_id, str) and record_id for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("frozen memory bank records have missing or duplicate record_id")
    if set(index) != set(record_ids):
        raise ValueError("frozen memory bank records/index IDs differ; restore a matching frozen snapshot")
    dimensions = set()
    for record in records:
        record_id = record["record_id"]
        item = index[record_id]
        if not isinstance(item, dict) or item.get("search_text") != MemoryRecord.from_dict(record).search_text:
            raise ValueError(f"frozen memory bank index text mismatch: {record_id}")
        vector = item.get("vector")
        if not isinstance(vector, list) or not vector or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in vector):
            raise ValueError(f"frozen memory bank invalid vector: {record_id}")
        dimensions.add(len(vector))
    if len(dimensions) > 1:
        raise ValueError("frozen memory bank index contains inconsistent vector dimensions")


def validate_cached_tevo_contexts(contexts: object, records: list[dict], task_ids: list[str]) -> None:
    """Fail closed when cached retrieval contexts do not match the current bank and plan.

    contexts.json is only trustworthy if it is a faithful record of this bank and
    this plan: every plan task has an entry; every entry carries a list ``memories``
    whose items are verbatim copies of current bank records (rank_memories stores
    ``{**record, ...}``); and ``text`` equals ``format_memories(memories, "tevo")``,
    the body the model actually reads (a legitimate empty recall is an explicit
    ``memories=[]`` with the canonical empty text). Missing tasks or fields, tampered
    text, an unknown id or a divergent when_to_use/content must abort instead of
    silently scoring stale memories. run_suite additionally binds bank hashes into
    the manifest fingerprint and the contexts hash into artifacts.json; this is the
    local defense for direct callers and for resumes of a swapped bank.
    """
    tevo = contexts.get("tevo") if isinstance(contexts, dict) else None
    if not isinstance(tevo, dict):
        raise ValueError("cached retrieval contexts lack a tevo section")
    missing = sorted(task for task in task_ids if task not in tevo)
    if missing:
        raise ValueError(f"cached retrieval contexts miss plan tasks: {missing}")
    by_id = {record["record_id"]: record for record in records}
    for task, entry in tevo.items():
        memories = entry.get("memories") if isinstance(entry, dict) else None
        _validate_verbatim_memories(memories, by_id, f"cached retrieval contexts {task}")
        if entry.get("text") != format_memories(memories, "tevo"):
            raise ValueError(f"cached retrieval contexts text mismatch: {task}")


def _validate_verbatim_memories(memories: object, by_id: dict, label: str) -> None:
    """Memories must be verbatim copies of current bank records (when_to_use/content)."""
    if not isinstance(memories, list):
        raise ValueError(f"{label}: entry malformed")
    for memory in memories:
        if not isinstance(memory, dict):
            raise ValueError(f"{label}: memory malformed")
        record_id = memory.get("id")
        record = by_id.get(record_id)
        if record is None:
            raise ValueError(f"{label}: unknown memory {record_id}")
        if memory.get("when_to_use") != record.get("when_to_use") or memory.get("content") != record.get("content"):
            raise ValueError(f"{label}: memory mismatch {record_id}")


def validate_cached_per_turn_contexts(contexts: object, records: list[dict], turns_by_task: dict,
                                      *, query_mode: str = "cumulative_user") -> None:
    """Fail closed when cached per-turn contexts diverge from the frozen turn texts.

    Every task must carry one entry per dataset user turn, in order; each turn's
    ``query_text`` must equal the frozen cumulative-uplift text U_0..U_t
    verbatim as the runtime lookup identity. In current_user mode the actual
    embedding query is U_t, stored separately with its vector and hash.
    ``text``/``memories`` must reproduce ``format_memories`` over
    verbatim bank records, exactly like the first-turn section.
    """
    from tevo.experiment.per_turn import cumulative_queries
    section = contexts.get("tevo_per_turn") if isinstance(contexts, dict) else None
    if not isinstance(section, dict):
        raise ValueError("cached retrieval contexts lack a tevo_per_turn section")
    missing = sorted(task for task in turns_by_task if task not in section)
    if missing:
        raise ValueError(f"cached per-turn contexts miss plan tasks: {missing}")
    by_id = {record["record_id"]: record for record in records}
    for task, turns in turns_by_task.items():
        entry = section.get(task)
        rows = entry.get("turns") if isinstance(entry, dict) else None
        if not isinstance(rows, list) or len(rows) != len(turns):
            raise ValueError(f"cached per-turn contexts turn count mismatch: {task}")
        queries = cumulative_queries(turns)
        for turn_index, (row, expected_query) in enumerate(zip(rows, queries)):
            if not isinstance(row, dict) or row.get("turn") != turn_index:
                raise ValueError(f"cached per-turn contexts turn identity mismatch: {task}:{turn_index}")
            if row.get("query_text") != expected_query:
                raise ValueError(f"cached per-turn contexts query text mismatch: {task}:{turn_index}")
            if row.get("query_text_sha256") != canonical_hash(expected_query):
                raise ValueError(f"cached per-turn contexts query hash mismatch: {task}:{turn_index}")
            if query_mode == "current_user":
                current = turns[turn_index]
                if row.get("retrieval_query_text") != current or row.get("retrieval_query_sha256") != canonical_hash(current):
                    raise ValueError(f"cached per-turn current-user query mismatch: {task}:{turn_index}")
                vector = row.get("retrieval_query_vector")
                if not isinstance(vector, list) or not vector or any(
                        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector):
                    raise ValueError(f"cached per-turn invalid query vector: {task}:{turn_index}")
                if row.get("query_vector_sha256") != canonical_hash(vector):
                    raise ValueError(f"cached per-turn query vector hash mismatch: {task}:{turn_index}")
            _validate_verbatim_memories(row.get("memories"), by_id, f"cached per-turn contexts {task}:{turn_index}")
            if row.get("text") != format_memories(row["memories"], "tevo"):
                raise ValueError(f"cached per-turn contexts text mismatch: {task}:{turn_index}")
        if query_mode == "current_user":
            first = contexts["tevo"][task]
            if rows[0]["memories"] != first["memories"] or rows[0]["query_vector_sha256"] != first["query_vector_sha256"]:
                raise ValueError(f"cached per-turn first context mismatch: {task}")


def lookup_per_turn_entry(per_turn_context: object, query: str) -> dict:
    """Exact-text lookup of the frozen per-turn entry for a runtime query.

    The runtime query is rebuilt from the env-delivered verbatim user texts; a
    miss means the run diverged from the frozen dataset turns, so this fails
    closed instead of serving the wrong turn's memories.
    """
    rows = per_turn_context.get("turns") if isinstance(per_turn_context, dict) else None
    if not isinstance(rows, list):
        raise ValueError("per-turn retrieval requested but the frozen per-turn contexts are missing")
    for row in rows:
        if isinstance(row, dict) and row.get("query_text") == query:
            return row
    raise ValueError(f"per-turn query is not in the frozen per-turn contexts: {query[:160]!r}")


def rank_memories(records: list[dict], index: dict, query_vector: list[float], top_k: int, method: str,
                  *, available_tools: set[str] | None = None) -> list[dict]:
    from tevo.memory.embedding import cosine_similarity
    from tevo.memory.procedural_schema import MEMORY_TAGS, MemoryRecord
    if available_tools is not None and (method != "tevo" or not available_tools
            or any(not isinstance(t, str) or not t for t in available_tools)):
        raise ValueError("task-tool filtering requires tevo and a nonempty tool-name set")
    if method == "tevo":
        # Preserve the service's exact semantics: index insertion order, one
        # stable score sort, per-tag top-k, tag-order concatenation, rounded
        # scores and then another stable global sort (including rounded ties).
        lookup = {record["record_id"]: MemoryRecord.from_dict(record) for record in records}
        scored = []
        for identity, item in index.items():
            record = lookup.get(identity)
            if record is None or record.stale or not item.get("vector"):
                continue
            if available_tools is not None and (not record.tool_tags or not set(record.tool_tags) <= available_tools):
                continue
            scored.append((record, cosine_similarity(query_vector, item["vector"])))
        scored.sort(key=lambda item: item[1], reverse=True)
        groups = {tag: [] for tag in MEMORY_TAGS}
        for record, score in scored:
            group = groups.setdefault(record.type_tag, [])
            if len(group) < top_k:
                group.append((record, score))
        memory_list = []
        for group in groups.values():
            for record, score in group:
                if score < 0:
                    continue
                memory_list.append({"id": record.record_id, "tevo_type": record.type_tag,
                                    "when_to_use": record.when_to_use, "content": record.content,
                                    "tool_tags": list(record.tool_tags), "score": round(float(record.confidence or .5), 6),
                                    "usage_count": record.usage_count, "success_count": record.success_count,
                                    "source": record.source, "metadata": dict(record.metadata),
                                    "similarity": round(score, 6)})
        memory_list.sort(key=lambda record: record["similarity"], reverse=True)
        return memory_list[:top_k]
    candidates = []
    for record in records:
        identity = record.get("memory_id")
        if identity not in index:
            raise ValueError(f"missing frozen memory vector: {identity}")
        score = cosine_similarity(query_vector, index[identity]["vector"])
        candidates.append((record, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    chosen, seen = [], set()
    # Official ReMe recall retrieves top_k before de-duplicating content, so
    # duplicates reduce the returned count rather than being refilled.
    pool = candidates[:top_k]
    for record, score in pool:
        if record.get("content", "") in seen:
            continue
        seen.add(record.get("content", ""))
        chosen.append({**record, "similarity": round(score, 6),
                       "id": record.get("record_id", record.get("memory_id")),
                       "tevo_type": record.get("type_tag", "generic_rule")})
        if len(chosen) == top_k:
            break
    return chosen


def task_tool_names(task: dict) -> set[str]:
    """Use the executor's task function list, not the converted global tool list."""
    functions = task.get("extra", {}).get("function")
    if (not isinstance(functions, list) or not functions
            or any(not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]
                   for f in functions)):
        raise ValueError("task-tool filtering requires extra.function contracts")
    return {f["name"] for f in functions}


def validate_task_filtered_contexts(contexts: dict, records: list[dict], index: dict,
                                    tasks: dict, task_ids: list[str], top_k: int) -> None:
    """Reject old/unfiltered caches, even when their texts match the bank."""
    names = {task: task_tool_names(tasks[task]) for task in task_ids}
    if contexts.get("retrieval_tool_filter") is not True or contexts.get("available_tools_sha256") != canonical_hash(
            {task: sorted(value) for task, value in names.items()}):
        raise ValueError("cached task-tool filter identity mismatch")
    for task in task_ids:
        rows = [contexts["tevo"][task], *contexts.get("tevo_per_turn", {}).get(task, {}).get("turns", [])]
        for row in rows:
            vector = row.get("retrieval_query_vector")
            dimension = len(next(iter(index.values()))["vector"])
            if (not isinstance(vector, list) or len(vector) != dimension or not vector
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
                    or row.get("query_vector_sha256") != canonical_hash(vector)):
                raise ValueError("cached task-tool query vector mismatch")
            expected = rank_memories(records, index, vector, top_k, "tevo", available_tools=names[task])
            if row["memories"] != expected or row["text"] != format_memories(expected, "tevo"):
                raise ValueError("cached task-tool filtered ranking mismatch")


def format_memories(records: list[dict], method: str) -> str:
    if method == "reme":
        return "\n".join(f"Experience {i} :\n When to use: {r.get('when_to_use', '')}\n Content: {r.get('content', '')}\n" for i, r in enumerate(records, 1))
    parts = []
    ordered = sorted(records, key=lambda record: -float(record.get("similarity", record.get("score", .5)) or 0.0))
    for i, record in enumerate(ordered, 1):
        when = str(record.get("when_to_use", "") or "").strip()
        content = str(record.get("content", "") or "").strip()
        if not when and not content:
            continue
        # v6 表单：无审计标头（type_tag/tool_tags 留在 record metadata 供分析），
        # 模型只读取场景与正文；检索元信息不进入注入文本。
        lines = [f"Experience {i}:"]
        if when:
            lines.append("When to use: " + when)
        if content:
            lines.append(content)
        parts.append("\n".join(lines))
    return "\n\n".join(parts) or "No relevant procedural experiences found."


def _prepare_per_turn_contexts(plan: dict, frozen: dict, contexts: dict, records: list[dict], index: dict,
                               query_vectors: list[list[float]], top_k: int, ledger: RequestLedger) -> None:
    """Precompute the per-turn retrieval contexts and cross-check them fail-closed.

    Turn 0 reuses the first-turn batch embedded above (pilot composition, one
    6-input POST); the later turns are embedded as one second POST (14 inputs at
    the frozen batch size), mirroring the amended diagnostic's composition.  The
    results are then checked against two frozen references before anything is
    stored: the pilot ``contexts.json`` (turn 0 must reproduce the first-turn
    vectors, memories and identical injected text) and the amended diagnostic
    report (every turn's query text hash, query vector is bitwise-equal, and the
    re-ranked top-5 ids equal the reported rows).  Any divergence aborts the run
    before a trial is launched.
    """
    from tevo.experiment.per_turn import cumulative_queries, partition_turn_queries
    if frozen.get("retrieval_query_mode") == "current_user":
        tasks = ({t["id"]: t for t in load_jsonl(Path(plan["dataset_path"]))}
                 if frozen.get("retrieval_tool_filter") else {})
        # Runtime still uses the cumulative prefix as an exact lookup identity;
        # retrieval uses only the current visible user turn, as in the pilot.
        later = [(task, turn, query) for task in plan["task_ids"]
                 for turn, query in enumerate(plan["per_turn_turns"][task]) if turn]
        vectors, events = _embedding([q for _, _, q in later], frozen["embedding"], ledger,
                                     "prepare_current_user_queries")
        contexts["embedding_events"].extend(events)
        by_turn = {(task, 0): vector for task, vector in zip(plan["task_ids"], query_vectors)}
        by_turn.update({(task, turn): vector for (task, turn, _), vector in zip(later, vectors)})
        for task in plan["task_ids"]:
            turns = plan["per_turn_turns"][task]
            rows = []
            for turn, (current, lookup) in enumerate(zip(turns, cumulative_queries(turns))):
                vector = by_turn[task, turn]
                chosen = rank_memories(records, index, vector, top_k, "tevo",
                    **({"available_tools": task_tool_names(tasks[task])} if tasks else {}))
                rows.append({"turn": turn, "query_text": lookup, "query_text_sha256": canonical_hash(lookup),
                             "retrieval_query_text": current, "retrieval_query_sha256": canonical_hash(current),
                             "retrieval_query_vector": vector, "query_vector_sha256": canonical_hash(vector),
                             "text": format_memories(chosen, "tevo"), "memories": chosen})
            contexts["tevo_per_turn"][task] = {"user_turns": len(turns), "turns": rows}
        contexts["retrieval_query_mode"] = "current_user"
        validate_cached_per_turn_contexts(contexts, records, plan["per_turn_turns"], query_mode="current_user")
        return
    reference = plan["per_turn_reference"]
    pilot_contexts = json.loads(Path(reference["pilot_contexts_path"]).read_text(encoding="utf-8"))
    report = json.loads(Path(reference["diagnostic_report_path"]).read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("per-turn reference report is not a passed diagnostic report")
    _, later_items = partition_turn_queries(plan["per_turn_turns"], plan["task_ids"])
    later_vectors, later_events = _embedding([item[2] for item in later_items], frozen["embedding"], ledger,
                                             "prepare_tevo_later_turn_queries")
    contexts["embedding_events"].extend(later_events)
    vectors: dict[str, list] = {task: [None] * len(plan["per_turn_turns"][task]) for task in plan["task_ids"]}
    for task, vector in zip(plan["task_ids"], query_vectors):
        vectors[task][0] = vector  # the first-turn batch embedded by prepare_contexts (pilot composition)
    for (task, turn_index, _), vector in zip(later_items, later_vectors):
        vectors[task][turn_index] = vector
    for task in plan["task_ids"]:
        turns = plan["per_turn_turns"][task]
        queries = cumulative_queries(turns)
        entry = contexts["tevo"][task]
        if canonical_hash(vectors[task][0]) != entry["query_vector_sha256"]:
            raise ValueError(f"per-turn turn-0 vector does not reproduce the stored first-turn hash: {task}")
        pilot_entry = (pilot_contexts.get("tevo") or {}).get(task)
        if not isinstance(pilot_entry, dict) or entry["memories"] != pilot_entry.get("memories") \
                or entry["text"] != pilot_entry.get("text"):
            raise ValueError(f"per-turn turn-0 retrieval diverges from the pilot contexts: {task}")
        reported_vectors = (report.get("query_vectors") or {}).get(task)
        reported_rows = ((report.get("tasks") or {}).get(task) or {}).get("turn_rows")
        if not isinstance(reported_vectors, list) or not isinstance(reported_rows, list) \
                or len(reported_vectors) != len(turns) or len(reported_rows) != len(turns):
            raise ValueError(f"per-turn reference report does not cover the frozen turns: {task}")
        rows = []
        for turn_index, query in enumerate(queries):
            vector = vectors[task][turn_index]
            reported = reported_vectors[turn_index]
            if vector is None or not isinstance(reported, dict) or reported.get("turn") != turn_index \
                    or reported.get("text_sha256") != canonical_hash(query) or reported.get("vector") != vector:
                raise ValueError(f"per-turn query diverges from the reference report: {task}:{turn_index}")
            chosen = rank_memories(records, index, vector, top_k, "tevo")
            reported_top5 = reported_rows[turn_index].get("top5") if isinstance(reported_rows[turn_index], dict) else None
            if not isinstance(reported_top5, list) \
                    or [memory["id"] for memory in chosen] != [row.get("id") for row in reported_top5]:
                raise ValueError(f"per-turn retrieval diverges from the reference report: {task}:{turn_index}")
            rows.append({"turn": turn_index, "query_text": query, "query_text_sha256": canonical_hash(query),
                         "query_vector_sha256": canonical_hash(vector),
                         "text": format_memories(chosen, "tevo"), "memories": chosen})
        contexts["tevo_per_turn"][task] = {"user_turns": len(turns), "turns": rows}
    contexts["per_turn_reference"] = {
        "pilot_contexts_path": reference["pilot_contexts_path"],
        "pilot_contexts_sha256": sha256_file(Path(reference["pilot_contexts_path"])),
        "diagnostic_report_path": reference["diagnostic_report_path"],
        "diagnostic_report_sha256": sha256_file(Path(reference["diagnostic_report_path"])),
    }


def prepare_contexts(plan: dict, output: Path, ledger: RequestLedger) -> dict:
    """Local read-only bank retrieval. No service load/clear/update endpoints.

    Bank validation is CPU-only and precedes every embedding request: the fresh
    path validates the frozen bank before query embedding, and a cached
    contexts.json is still re-validated against the current bank and plan
    (structure, task coverage, memory identity and the exact injected text;
    fail closed on a cache built from another bank generation).
    """
    methods = {arm["method"] for arm in plan["experiments"]}
    path = output / "contexts.json"
    if path.exists():
        contexts = json.loads(path.read_text(encoding="utf-8"))
        if methods & {"memonly", "both"}:
            if contexts.get("retrieval_tool_filter", False) != plan["config"]["frozen"].get("retrieval_tool_filter", False):
                raise ValueError("cached task-tool filter identity mismatch")
            records = json.loads(Path(plan["bank_paths"]["memories_path"]).read_text(encoding="utf-8"))
            index = json.loads(Path(plan["bank_paths"]["index_path"]).read_text(encoding="utf-8"))
            validate_frozen_bank_index(records, index)
            validate_cached_tevo_contexts(contexts, records, plan["task_ids"])
            if plan["per_turn_turns"]:
                validate_cached_per_turn_contexts(contexts, records, plan["per_turn_turns"],
                    query_mode=plan["config"]["frozen"].get("retrieval_query_mode", "cumulative_user"))
            if plan["config"]["frozen"].get("retrieval_tool_filter"):
                tasks = {t["id"]: t for t in load_jsonl(Path(plan["dataset_path"]))}
                validate_task_filtered_contexts(contexts, records, index, tasks, plan["task_ids"],
                                               int(plan["config"]["frozen"].get("top_k", 5)))
        return contexts
    if not methods & {"memonly", "both", "reme_public96"}:
        contexts = {"tevo": {}, "reme": {}, "embedding_events": []}
        atomic_json(path, contexts)
        return contexts
    frozen = plan["config"]["frozen"]
    rows = {row["id"]: row for row in load_jsonl(Path(plan["dataset_path"]))}
    queries = []
    for task in plan["task_ids"]:
        messages = rows[task].get("messages", [])
        if not messages or messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
            raise ValueError(f"BFCL converted dataset lacks first actor-visible user query: {task}")
        queries.append(messages[0]["content"])
    contexts = {"tevo": {}, "reme": {}, "embedding_events": [],
                "retrieval_protocol": "initial actor-visible user query; cosine top-k; frozen local banks"}
    top_k = int(frozen.get("top_k", 5))
    if plan["per_turn_turns"]:
        contexts["tevo_per_turn"] = {}
    if methods & {"memonly", "both"}:
        # 银行 CPU 校验先于 query embedding：旧代/坏库在发出任何请求前 fail closed。
        records = json.loads(Path(plan["bank_paths"]["memories_path"]).read_text(encoding="utf-8"))
        index = json.loads(Path(plan["bank_paths"]["index_path"]).read_text(encoding="utf-8"))
        validate_frozen_bank_index(records, index)
        if frozen.get("retrieval_tool_filter"):
            for task in plan["task_ids"]:
                task_tool_names(rows[task])  # CPU contracts must validate before embedding.
        query_vectors, events = _embedding(queries, frozen["embedding"], ledger, "prepare_tevo_queries")
        contexts["embedding_events"].extend(events)
        for task, vector in zip(plan["task_ids"], query_vectors):
            chosen = rank_memories(records, index, vector, top_k, "tevo",
                **({"available_tools": task_tool_names(rows[task])} if frozen.get("retrieval_tool_filter") else {}))
            contexts["tevo"][task] = {"text": format_memories(chosen, "tevo"), "memories": chosen, "query_vector_sha256": canonical_hash(vector)}
            if frozen.get("retrieval_tool_filter"):
                contexts["tevo"][task]["retrieval_query_vector"] = vector
        if plan["per_turn_turns"]:
            _prepare_per_turn_contexts(plan, frozen, contexts, records, index, query_vectors, top_k, ledger)
            contexts["retrieval_protocol"] += (
                "; per-turn arms embed current visible user text only; cumulative prefix is a lookup identity"
                if frozen.get("retrieval_query_mode") == "current_user" else
                "; per-turn arms embed cumulative user-text prefix U_0..U_t; frozen reference cross-checked")
        if frozen.get("retrieval_tool_filter"):
            contexts["retrieval_tool_filter"] = True
            contexts["available_tools_sha256"] = canonical_hash(
                {task: sorted(task_tool_names(rows[task])) for task in plan["task_ids"]})
            validate_task_filtered_contexts(contexts, records, index, rows, plan["task_ids"], top_k)
    if "reme_public96" in methods:
        if frozen.get("reme_contexts_path"):
            snapshot = validate_reme_snapshot(Path(plan["bank_paths"]["reme_contexts_path"]), Path(plan["bank_paths"]["reme_bank_path"]), Path(plan["dataset_path"]),
                                              {task: rows[task] for task in plan["task_ids"]})
            for task in plan["task_ids"]:
                entry = snapshot["contexts"][task]
                contexts["reme"][task] = {"text": re.sub(r"\bMemory\s*(\d+)\s*[:]", r"Experience \1 :", entry["answer"]),
                                           "memories": [{"id": memory["memory_id"], **memory} for memory in entry["memories"]],
                                           "official_response_sha256": entry["response_sha256"]}
            contexts["reme_backend"] = "official_readonly_service_snapshot"
        else:
            settings = frozen["reme_embedding"]
            query_vectors, events = _embedding(queries, settings, ledger, "prepare_reme_queries")
            contexts["embedding_events"].extend(events)
            records = load_jsonl(Path(plan["bank_paths"]["reme_bank_path"]))
            vectors, bank_events = _embedding([record["when_to_use"] for record in records], settings, ledger, "prepare_reme_bank")
            contexts["embedding_events"].extend(bank_events)
            index = {record["memory_id"]: {"vector": vector} for record, vector in zip(records, vectors)}
            atomic_json(output / "reme_public_index.json", index)
            for task, vector in zip(plan["task_ids"], query_vectors):
                chosen = rank_memories(records, index, vector, top_k, "reme")
                contexts["reme"][task] = {"text": format_memories(chosen, "reme"), "memories": chosen, "query_vector_sha256": canonical_hash(vector)}
            contexts["reme_backend"] = "local_cosine_adapter"
        contexts["reme_baseline_scope"] = "public frozen bank inference; no matched acquisition-budget or historical-server-identity claim"
    atomic_json(path, contexts)
    return contexts


def _run_trial(job: dict, plan: dict, output_path: str, contexts: dict) -> dict:
    """Executed in a spawned worker; BFCL global environment caches are private."""
    output = Path(output_path)
    model, arm = plan["config"]["model"], job["arm"]
    os.environ["LLM_BASE_URL"] = job["llm_base_url"]
    os.environ["LLM_API_KEY"] = os.environ.get(model.get("api_key_env", "TEVO_LLM_API_KEY"), "dummy")
    os.environ["TEVO_DISABLE_MEMORY_USAGE_FEEDBACK"] = "1"
    os.environ["TEVO_INJECT_TOOL_PROFILES"] = "1" if arm["method"] in {"profiles", "both"} else "0"
    os.environ["TEVO_COMPACT_BFCL_RESULTS"] = "0"
    os.environ["TEVO_LLM_THINKING_EXTRA_BODY"] = "chat_template_kwargs"
    os.environ["TEVO_INJECT_TOOL_PROFILE_FIELDS"] = ",".join(arm.get("profile_fields", ["when_to_use", "preconditions", "failure_cues", "recovery_hints"]))
    os.environ["TEVO_TOOL_PROFILE_MAX_CHARS"] = str(arm.get("profile_max_chars", 400))
    bfcl = ROOT / "benchmark/bfcl"
    sys.path[:0] = [str(bfcl), str(bfcl / "gorilla/berkeley-function-call-leaderboard")]
    from bfcl_agent import LocalBFCLAgent
    from tevo.reproducibility import seed_python
    method = arm["method"]
    profiles = {}
    if method in {"profiles", "both"}:
        records = json.loads(Path(plan["bank_paths"]["registry_path"]).read_text(encoding="utf-8"))
        profiles = {record["tool_id"]: record for record in records if not record.get("deprecated", False)}
    context = contexts["reme" if method == "reme_public96" else "tevo"].get(job["task_id"], {})
    per_turn_memory = bool(arm.get("per_turn_memory", False))
    per_turn_context = contexts.get("tevo_per_turn", {}).get(job["task_id"], {}) if per_turn_memory else {}

    class FrozenAgent(LocalBFCLAgent):
        def _refresh_tool_profiles(self):
            self.tool_profile_cache = profiles
            return len(profiles)

        def _retrieve_tevo_memory_context(self, run_id, task_index, query=None):
            if query is not None:
                # Per-turn refresh: exact-text lookup into the frozen per-turn
                # contexts; a miss means the run diverged from the frozen turns.
                entry = lookup_per_turn_entry(per_turn_context, query)
                self.retrieved_memory_list[run_id][task_index] = entry.get("memories", [])
                return entry.get("text")
            self.retrieved_memory_list[run_id][task_index] = context.get("memories", [])
            return context.get("text")

    ledger = RequestLedger(output / "requests.sqlite", plan["config"]["budgets"])
    budget_interrupted = False
    started = time.monotonic()
    try:
        seed_python(job["trial_seed"])
        fixed_sampling = plan["config"].get("sampling_seed_mode") == "fixed"
        if fixed_sampling:
            # Use the actor's existing raw-seed path (no task derivation).
            # Its UUID episode identity keeps repeated executions separate.
            os.environ["TEVO_LLM_SEED"] = str(job["trial_seed"])
        agent = FrozenAgent(index=job["ordinal"], task_ids=[job["task_id"]],
                            experiment_name="bfcl-multi-turn-base-val_" + job["experiment"],
                            data_path=plan["dataset_path"], answer_path=Path(plan["answer_path"]),
                            model_name=model["name"], model_path=model.get("path", model["name"]),
                            temperature=float(arm.get("temperature", model["temperature"])),
                            enable_thinking=bool(model.get("enable_thinking", True)),
                            num_trials=1, trial_seeds=None if fixed_sampling else [job["base_seed"]], max_tokens=model["max_tokens"], max_workers=1,
                            context_budget=model.get("context_budget"),
                            max_interactions=int(arm.get("max_interactions", model.get("max_interactions", 30))),
                            max_retries=int(model.get("max_retries", 1)), request_timeout_seconds=float(model.get("timeout_seconds", 120)),
                            use_memory=method in {"memonly", "both", "reme_public96"}, use_tool_profiles=method in {"profiles", "both"},
                            use_memory_addition=False, use_memory_deletion=False, use_tevo_observer=False,
                            memory_base_url="http://127.0.0.1:1/", tool_scope=arm.get("tool_scope", "task"),
                            per_turn_memory=per_turn_memory)
        completion = agent.llm_client.chat.completions.create

        def bounded_completion(*args, **kwargs):
            nonlocal budget_interrupted
            try:
                request_id = ledger.reserve(job["key"], "llm")
            except BudgetExceeded:
                budget_interrupted = True
                raise
            try:
                response = completion(*args, **kwargs)
                usage = response.usage.model_dump() if response.usage is not None else None
                ledger.finish(request_id, usage, "success")
                return response
            except Exception:
                ledger.finish(request_id, None, "error")
                raise

        agent.llm_client.chat.completions.create = bounded_completion
        tokenization = agent._request_tokenization

        def bounded_tokenization(payload):
            nonlocal budget_interrupted
            try:
                request_id = ledger.reserve(job["key"], "tokenize")
            except BudgetExceeded:
                budget_interrupted = True
                raise
            try:
                raw = tokenization(payload)
                ledger.finish(request_id, None, "success")
                return raw
            except Exception:
                ledger.finish(request_id, None, "error")
                raise

        agent._request_tokenization = bounded_tokenization
        records = agent.execute()
        if len(records) != 1:
            raise ValueError("single-trial worker returned unexpected number of results")
        record = records[0]
        record["status"] = "budget_interrupted" if budget_interrupted else "error" if record.get("error") or record.get("evaluation_error") else "complete"
    except Exception as exc:
        record = {"task_id": job["task_id"], "reward": 0.0, "status": "budget_interrupted" if budget_interrupted else "error",
                  "error_type": type(exc).__name__, "phase": "execution", "request_metrics": []}
    record.update({"key": job["key"], "experiment": job["experiment"], "base_seed": job["base_seed"],
                   "trial_seed": job["trial_seed"], "run_id": job["trial_index"], "elapsed_seconds": time.monotonic() - started,
                   "endpoint_index": job["endpoint_index"], "llm_base_url": job["llm_base_url"],
                   "memory_context_sha256": canonical_hash(context), "frozen_bank_read_only": True,
                   "per_turn_memory": per_turn_memory,
                   "per_turn_context_sha256": canonical_hash(per_turn_context) if per_turn_memory else None})
    if "repeat_index" in job:
        record["repeat_index"] = job["repeat_index"]
    return record


def collect_results(plan: dict, output: Path, manifest_hash: str) -> tuple[list[dict], list[dict]]:
    completed, pending = [], []
    for job in plan["jobs"]:
        path = output / "trials" / (job["key"] + ".json")
        if not path.exists():
            pending.append(job)
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        identity_names = ("key", "task_id", "experiment", "base_seed", "trial_seed", "endpoint_index", "llm_base_url")
        expected = tuple(job[name] for name in identity_names)
        actual = tuple(row.get(name) for name in identity_names)
        if actual != expected or row.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"Result identity/provenance mismatch: {path}")
        if "repeat_index" in job and (row.get("repeat_index") != job["repeat_index"] or row.get("run_id") != job["trial_index"]):
            raise ValueError(f"Result repeat identity mismatch: {path}")
        if row.get("status") not in {"complete", "error"}:
            pending.append(job)
        else:
            reward = row.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError(f"Invalid reward in {path}")
            completed.append(row)
    return completed, pending


def summarize(plan: dict, rows: list[dict], pending: list[dict], ledger: RequestLedger) -> dict:
    from tevo.evaluation.statistics import paired_comparison, summarize as strict_summary
    by_arm = {}
    complete_groups = {}
    for arm in plan["experiments"]:
        data = [row for row in rows if row["experiment"] == arm["name"]]
        per_task = len(plan["config"]["seeds"]) * plan["config"].get("repeats_per_seed", 1)
        expected = len(plan["task_ids"]) * per_task
        grouped = {task: [row["reward"] for row in data if row["task_id"] == task] for task in plan["task_ids"]}
        expected_keys = {j["key"] for j in plan["jobs"] if j["experiment"] == arm["name"]}
        complete = (len(data) == expected and {r["key"] for r in data} == expected_keys
                    and all(len(values) == per_task for values in grouped.values()))
        by_arm[arm["name"]] = {"completed_trials": len(data), "expected_trials": expected, "complete": complete,
                                "error_trials": sum(row["status"] == "error" for row in data),
                                "successes": sum(row["reward"] == 1 for row in data),
                                "trial_mean": sum(row["reward"] for row in data) / len(data) if data else None,
                                "best_of_observed_seeds": sum(max(grouped[t]) for t in grouped) / len(grouped) if complete else None,
                                "observed_seed_count": len(plan["config"]["seeds"]),
                                "repeats_per_seed": plan["config"].get("repeats_per_seed", 1),
                                "avg_at_4": sum(sum(v) / 4 for v in grouped.values()) / len(grouped) if complete and per_task == 4 else None,
                                "pass_at_4": sum(any(r == 1 for r in v) for v in grouped.values()) / len(grouped) if complete and per_task == 4 else None}
        if plan["config"].get("sampling_seed_mode") == "fixed":
            by_arm[arm["name"]]["best_of_observed_trials"] = by_arm[arm["name"]]["best_of_observed_seeds"]
            by_arm[arm["name"]]["best_of_observed_seeds"] = None
        if complete:
            groups = {task: [row for row in data if row["task_id"] == task] for task in plan["task_ids"]}
            by_arm[arm["name"]]["statistics"] = strict_summary(groups)
            complete_groups[arm["name"]] = groups
    comparisons = {}
    baseline = plan["config"].get("analysis_baseline", "nomem")
    if baseline in complete_groups:
        for name, groups in complete_groups.items():
            if name != baseline:
                comparisons[name + "_minus_" + baseline] = paired_comparison(groups, complete_groups[baseline])
    if "both" in complete_groups and "reme_public96" in complete_groups:
        comparisons["both_minus_reme_public96"] = paired_comparison(complete_groups["both"], complete_groups["reme_public96"])
    adapter_verified = "reme_public96" not in {arm["method"] for arm in plan["experiments"]} or plan.get("reme_snapshot_verified", False) or plan["config"].get("frozen", {}).get("reme_adapter_verified") is True
    complete_matrix = not pending and all(a["complete"] for a in by_arm.values())
    return {"all_trials_complete": complete_matrix, "completed_trials": len(rows), "expected_trials": len(plan["jobs"]),
            "pending_trials": len(pending), "paper_result_eligible": complete_matrix and adapter_verified and not any(row["status"] == "error" for row in rows),
            "baseline_adapter_verified": adapter_verified,
            "server_assignment": "paired arms of one (task, seed, repeat) share endpoint (task_position + trial_index) mod endpoint_count; balanced across tasks",
            "sampling_seed_mode": plan["config"].get("sampling_seed_mode", "task_derived"),
            "arms": by_arm, "paired_comparisons": comparisons, "resources": ledger.summary(),
            "monetary_cost": None, "monetary_cost_note": "No external price schedule supplied; token usage and latency are measured separately.",
            "interpretation": "Avg@4 averages four scores per task; Pass@4 is any full success among the four. Reported only for complete four-trial groups. Repeated trials are not independent tasks; fixed seeds do not guarantee server determinism. No historical results included."}


def run_suite(plan: dict, output: Path, resume: bool = False) -> dict:
    output = output.resolve()
    current = fingerprints(plan)
    manifest_hash = canonical_hash(current)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Output exists; --resume verifies provenance before continuing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprints"] != current:
            raise ValueError("Resume refused: code/config/data/checker/frozen-bank fingerprints changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing to reuse a nonempty output without a manifest")
        atomic_json(manifest_path, {"fingerprints": current, "manifest_sha256": manifest_hash,
                                  "configuration": redact_secrets(plan["config"]), "python": sys.version,
                                  "platform": platform.platform(), "historical_outputs_imported": False})
    ledger = RequestLedger(output / "requests.sqlite", plan["config"]["budgets"])
    providers = inspect_providers(plan["config"]["model"])
    provider_path = output / "providers.json"
    if provider_path.exists() and json.loads(provider_path.read_text(encoding="utf-8")) != providers:
        raise ValueError("Resume refused: advertised model identity/context length changed")
    atomic_json(provider_path, redact_secrets(providers))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider_metadata"] = redact_secrets(providers)
    atomic_json(manifest_path, manifest)
    contexts = prepare_contexts(plan, output, ledger)
    context_hash = canonical_hash(contexts)
    artifact_path = output / "artifacts.json"
    if artifact_path.exists() and json.loads(artifact_path.read_text(encoding="utf-8"))["contexts_sha256"] != context_hash:
        raise ValueError("Frozen retrieval-context artifact changed")
    atomic_json(artifact_path, {"contexts_sha256": context_hash})
    completed, pending = collect_results(plan, output, manifest_hash)
    # PYTHONHASHSEED takes effect at spawned interpreter startup.
    os.environ["PYTHONHASHSEED"] = "0"
    workers = plan["config"]["budgets"]["max_workers"]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        running = {}
        cursor = 0
        while cursor < len(pending) or running:
            while cursor < len(pending) and len(running) < workers:
                job = {**pending[cursor], "ordinal": plan["jobs"].index(pending[cursor])}
                # Send only this task's context and runtime inputs; transmitting
                # the entire matrix/bank with every trial wastes CPU and memory.
                runtime = {name: plan[name] for name in ("config", "bank_paths", "dataset_path", "answer_path")}
                trial_context = {kind: {job["task_id"]: contexts.get(kind, {}).get(job["task_id"], {})}
                                 for kind in ("tevo", "reme", "tevo_per_turn")}
                future = pool.submit(_run_trial, job, runtime, str(output), trial_context)
                running[future] = job
                cursor += 1
            finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            interrupted = False
            for future in finished:
                job = running.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {**{k: v for k, v in job.items() if k not in {"arm", "ordinal"}}, "reward": 0.0,
                           "run_id": job["trial_index"], "status": "error", "error_type": type(exc).__name__, "phase": "worker"}
                row["manifest_sha256"] = manifest_hash
                atomic_json(output / "trials" / (job["key"] + ".json"), row)
                print(json.dumps({"experiment": job["experiment"], "task_id": job["task_id"], "seed": job["base_seed"],
                                  "status": row["status"], "reward": row["reward"]}), flush=True)
                interrupted = interrupted or row["status"] == "budget_interrupted"
            if interrupted:
                cursor = len(pending)
    completed, pending = collect_results(plan, output, manifest_hash)
    summary = summarize(plan, completed, pending, ledger)
    final = fingerprints(plan)
    summary["provenance_valid"] = final == current
    if not summary["provenance_valid"]:
        summary["paper_result_eligible"] = False
        summary["provenance_error"] = "Code/data/answers/frozen banks changed during the run"
        atomic_json(output / "final_fingerprints.json", final)
    atomic_json(output / "summary.json", summary)
    return summary
