#!/usr/bin/env python3
"""Capture package, device, source and artifact provenance without dumping secrets.

This is an observation of the invoking interpreter, not a claim that all installed
packages have compatible dependencies or that GPU/remote sampling is deterministic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib import request
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tevo.reproducibility import redact_secrets, sha256_file

REPRODUCIBILITY_ENVIRONMENT = (
    "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG",
    "CUDNN_DETERMINISTIC", "VLLM_USE_V1", "TEVO_BASE_SEED",
)


def _command(arguments: list[str], *, cwd: Path | None = None) -> dict:
    try:
        result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error_type": type(error).__name__}
    return {"status": "ok" if result.returncode == 0 else "failed",
            "exit_code": result.returncode, "stdout": result.stdout.strip()}


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(packages.items()))


def capture(artifacts: list[Path], endpoints: list[str]) -> dict:
    snapshot = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "machine": platform.machine(), "release": platform.release()},
        "packages": installed_packages(),
        "source": {"commit": _command(["git", "rev-parse", "HEAD"], cwd=ROOT),
                   "changed_files": _command(["git", "status", "--short"], cwd=ROOT)},
        "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "reproducibility_environment": {key: os.environ.get(key) for key in REPRODUCIBILITY_ENVIRONMENT},
        "artifacts": {str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts},
        "endpoints": {},
        "limits": ["Installed versions are observations; run pip check in the isolated target environment.",
                   "Remote server versions and model weights require separate server-side provenance.",
                   "Seeds do not guarantee identical sampling across hardware or concurrent servers."],
    }
    for endpoint in endpoints:
        name, separator, url = endpoint.partition("=")
        if not separator or not name or not url.startswith(("http://", "https://")):
            raise ValueError("--endpoint must have the form name=http(s)://host/path")
        try:
            with request.urlopen(url, timeout=10) as response:
                result = {"status": response.status, "response": json.load(response)}
        except Exception as error:
            # Provider exception text can echo credentials or query strings.
            result = {"status": "unavailable", "error_type": type(error).__name__}
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parsed.port:
            host += f":{parsed.port}"
        public_url = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        snapshot["endpoints"][name] = {"url": public_url, "query_present": bool(parsed.query), **result}
    return redact_secrets(snapshot)


def write_constraints(packages: dict[str, str], destination: Path) -> None:
    lines = ["# Observed versions only; unused packages need not be installed.",
             "# Platform/CUDA wheels and local editable sources require separate provenance."]
    for name, version in packages.items():
        # BFCL is materialized from the source lock, not its local editable install.
        if name in {"bfcl-eval", "tevo-research"}:
            continue
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) and re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
            lines.append(f"{name}=={version}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON output; defaults to stdout")
    parser.add_argument("--constraints", type=Path, help="Optional installed-version snapshot")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--endpoint", action="append", default=[], help="Read-only JSON endpoint name=URL")
    arguments = parser.parse_args()
    try:
        snapshot = capture(arguments.artifact, arguments.endpoint)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Environment capture failed: {error}\n")
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if arguments.constraints:
        write_constraints(snapshot["packages"], arguments.constraints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
