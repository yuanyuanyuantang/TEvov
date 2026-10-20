"""Environment snapshots must not expose ambient credentials."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_environment.py"
spec = importlib.util.spec_from_file_location("capture_environment", SCRIPT)
capture_environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_environment)


def test_ambient_credentials_not_captured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-private-credential")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(capture_environment, "_command", lambda *args, **kwargs: {"status": "ok"})
    snapshot = capture_environment.capture([], [])
    assert snapshot["reproducibility_environment"]["PYTHONHASHSEED"] == "42"
    assert "test-private-credential" not in str(snapshot)
    assert "OPENAI_API_KEY" not in str(snapshot)


def test_constraints_skip_local_sources_and_nonversions(tmp_path):
    target = tmp_path / "constraints.txt"
    capture_environment.write_constraints({"bfcl-eval": "0.0.0", "numpy": "1.26.4", "private": "https://secret"}, target)
    text = target.read_text()
    assert "numpy==1.26.4" in text
    assert "bfcl-eval==" not in text
    assert "https://secret" not in text
