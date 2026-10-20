"""Offline replay must preserve provenance, coverage and checker failures."""

import hashlib
import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest

from scripts import regrade_suite as module
from tevo.reproducibility import canonical_hash


@pytest.fixture
def replay(tmp_path, monkeypatch):
    provenance = {"code": "fixed-code", "data": "fixed-data"}
    digest = canonical_hash(provenance)
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprints": provenance, "manifest_sha256": digest}))
    jobs = [{"key": f"trial{i}", "task_id": f"task{i}", "experiment": "nomem", "base_seed": 1,
             "trial_seed": i, "endpoint_index": 0, "llm_base_url": "http://localhost:1/v1"}
            for i in range(2)]
    rows = [{**job, "status": "complete", "manifest_sha256": digest, "reward": i,
             "task_history": [{"role": "assistant", "score": i}]} for i, job in enumerate(jobs)]
    trial_dir = tmp_path / "trials"
    trial_dir.mkdir()
    for row in rows:
        (trial_dir / (row["key"] + ".json")).write_text(json.dumps(row))
    plan = {"jobs": jobs, "config": {"model": {"name": "test-model"}},
            "dataset_path": str(tmp_path / "data.jsonl"), "answer_path": str(tmp_path)}
    monkeypatch.setattr(module, "fingerprints", lambda plan: provenance)
    resources = []

    class Agent:
        def __init__(self, **kwargs):
            self.history = [[[]]]
            self._evaluation_error = None
            state = {"closed": False, "released": False}
            resources.append(state)
            self.state = state
            self.llm_client = SimpleNamespace(close=lambda: state.update(closed=True))

        def get_reward(self, *args):
            assert os.environ["TEVO_INJECT_TOOL_PROFILES"] == "0"
            print("synthetic checker output")
            return self.history[0][0][0]["score"]

        def _release_run_state(self, *args):
            self.state["released"] = True

    monkeypatch.setitem(sys.modules, "bfcl_agent", SimpleNamespace(LocalBFCLAgent=Agent))
    return SimpleNamespace(root=tmp_path, plan=plan, rows=rows, resources=resources,
                           agent=Agent, provenance=provenance)


def test_replay_has_positive_negative_controls_and_suppresses_score_output(replay, monkeypatch, capsys):
    monkeypatch.setenv("TEVO_INJECT_TOOL_PROFILES", "1")
    report = module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert report["matches"] == report["checked_trials"] == 2
    assert report["complete_matrix_regraded"] is True
    assert report["model_requests"] == report["network_requests"] == 0
    assert report["checks"][0]["scorer_stdout_sha256"] == hashlib.sha256(b"synthetic checker output\n").hexdigest()
    assert capsys.readouterr().out == ""
    assert os.environ["TEVO_INJECT_TOOL_PROFILES"] == "1"
    assert all(state["closed"] and state["released"] for state in replay.resources)


def test_partial_and_wrong_rewards_cannot_be_called_complete(replay):
    (replay.root / "trials/trial1.json").unlink()
    partial = module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert partial["all_checked_match"] is True
    assert partial["pending_trials"] == 1 and partial["complete_matrix_regraded"] is False
    row = replay.rows[1]
    row["reward"] = 0
    (replay.root / "trials/trial1.json").write_text(json.dumps(row))
    incorrect = module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert incorrect["all_checked_match"] is False and incorrect["matches"] == 1
    assert incorrect["complete_matrix_regraded"] is False


def test_replay_rejects_manifest_and_source_identity_changes(replay, monkeypatch):
    monkeypatch.setattr(module, "fingerprints", lambda plan: {"different": True})
    with pytest.raises(ValueError, match="unchanged"):
        module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    monkeypatch.setattr(module, "fingerprints", lambda plan: replay.provenance)
    (replay.root / "manifest.json").write_text(json.dumps({"fingerprints": replay.provenance, "manifest_sha256": "edited"}))
    with pytest.raises(ValueError, match="manifest hash"):
        module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert not replay.resources


def test_source_drift_and_missing_history_fail_the_final_check(replay, monkeypatch):
    observations = iter([replay.provenance, {"changed": True}])
    monkeypatch.setattr(module, "fingerprints", lambda plan: next(observations))
    drift = module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert drift["provenance_unchanged"] is False and drift["all_checked_match"] is False
    monkeypatch.setattr(module, "fingerprints", lambda plan: replay.provenance)
    row = replay.rows[0]
    del row["task_history"]
    (replay.root / "trials/trial0.json").write_text(json.dumps(row))
    missing = module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert missing["checks"][0]["status"] == "missing_full_history"
    assert missing["complete_matrix_regraded"] is False


@pytest.mark.parametrize("method", ["connect", "connect_ex", "sendto"])
def test_network_attempts_are_rejected_and_resources_released(replay, monkeypatch, method):
    # A broken replay guard must fail this test without contacting a real socket.
    def unavailable(*args):
        raise AssertionError("unguarded network method")

    monkeypatch.setattr(socket.socket, method, unavailable)
    monkeypatch.setenv("TEVO_INJECT_TOOL_PROFILES", "1")

    def attempt(self, *args):
        with socket.socket() as connection:
            arguments = (b"probe", ("localhost", 1)) if method == "sendto" else (("localhost", 1),)
            return getattr(connection, method)(*arguments)

    monkeypatch.setattr(replay.agent, "get_reward", attempt)
    with pytest.raises(RuntimeError, match="offline"):
        module.regrade(replay.plan, replay.root, fingerprint_builder=module.fingerprints)
    assert os.environ["TEVO_INJECT_TOOL_PROFILES"] == "1"
    assert replay.resources[0] == {"closed": True, "released": True}
