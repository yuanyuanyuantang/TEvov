import json

import pytest

from tevo.evaluation.statistics import load_trials, paired_comparison, sign_test


def test_missing_duplicate_and_unequal_trials_fail(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {"task_id": "a", "experiment_name": "trial0", "run_id": 0, "reward": 1}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        load_trials([path], task_ids={"a", "b"}, trials_per_task=1)
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_trials([path], task_ids={"a"}, trials_per_task=2)
    with pytest.raises(ValueError, match="equal trials"):
        paired_comparison({"a": [row]}, {"a": [row, row]})


def test_sign_test_does_not_test_the_mean_difference():
    values = [1, -.125]
    result = sign_test(values)
    assert result["wins"] == result["losses"] == 1
    assert result["p_value"] == 1
    assert sum(values) / len(values) != 0


def test_cluster_comparison_preserves_tasks_and_zero_variance():
    left = {t: [{"reward": 1}, {"reward": 1}] for t in ("a", "b")}
    right = {t: [{"reward": 0}, {"reward": 0}] for t in ("a", "b")}
    result = paired_comparison(left, right)
    assert result["trial_mean"]["difference"] == 1
    assert result["trial_mean"]["task_bootstrap_ci95"] == [1, 1]
