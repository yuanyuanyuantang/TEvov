"""A failed vector validation still spent any actually reported provider usage."""
import sqlite3
from types import SimpleNamespace

import pytest

from tevo.experiment.suite import RequestLedger, _embedding


@pytest.mark.parametrize("measured,expected", [(True, 37), (False, None)])
def test_invalid_embedding_retains_only_measured_usage(tmp_path, monkeypatch, measured, expected):
    import requests
    data = {"data": [{"index": 0, "embedding": [1.0]}], "usage": {"total_tokens": 37}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: data))
    ledger = RequestLedger(tmp_path / "ledger.sqlite", {
        "max_requests": 10, "max_total_tokens": 100, "wall_seconds": 100})
    with pytest.raises(ValueError, match="invalid vectors"):
        _embedding(["one query"], {"model": "fixture", "base_url": "http://unused.invalid",
                   "dimensions": 2, "usage_is_measured": measured}, ledger, "trial")
    with sqlite3.connect(ledger.path) as connection:
        assert connection.execute("SELECT tokens,status FROM requests").fetchall() == [(expected, "error")]
    assert ledger.summary()["requests"] == 1


def test_boolean_token_usage_cannot_be_counted_as_one_token(tmp_path):
    ledger = RequestLedger(tmp_path / "ledger.sqlite", {
        "max_requests": 10, "max_total_tokens": 100, "wall_seconds": 100})
    identity = ledger.reserve("trial", "fixture")
    with pytest.raises(ValueError, match="invalid provider total_tokens"):
        ledger.finish(identity, {"total_tokens": True}, "success")
