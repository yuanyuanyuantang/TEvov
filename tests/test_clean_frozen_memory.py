"""The bounded cleanup drops records without rewriting retained evidence or vectors."""
import pytest

from scripts import clean_frozen_memory as cleanup


def test_cleanup_is_drop_only_and_keeps_compiled_records():
    def ordinary(identity, content):
        return {"record_id": identity, "type_tag": "generic_rule", "when_to_use": "For a purchase",
                "content": content, "metadata": {"candidate_assessments": [{"task_id": "t"}]}}
    valid = ordinary("good", "Use the current user's price threshold.")
    threshold = ordinary("threshold", "Buy above 300.")
    tag = ordinary("tag", "Use #FamilyHistory.")
    order = ordinary("order", "After filling, round the amount.")
    compiled = {"record_id": "compiled", "type_tag": "parameter_template", "content": "unchanged"}
    records = [valid, threshold, tag, order, compiled]
    tasks = {"t": {"tools": []}}
    groups = {"t": {"trajectories": [{"calls": [{"args": {"tags": ["#FamilyHistory"]},
                                                  "goal_snapshot": "Buy above 300."}]}]}}
    kept, dropped = cleanup.select_records(records, tasks, groups, {"order": {"issue": "reverse order"}})
    assert kept == [valid, compiled] and kept[0] is valid and kept[1] is compiled
    assert [r["record_id"] for r in dropped] == ["threshold", "tag", "order"]
    assert dropped[-1]["reason"] == "train_review_exclusion"
    assert records[1]["content"] == "Buy above 300."


def test_cleanup_refuses_writing_in_evidence_tree(tmp_path):
    source, trials = tmp_path / "source", tmp_path / "trials"
    with pytest.raises(ValueError, match="outside the input trees"):
        cleanup.clean_bank(source, trials, tmp_path / "review.json", source / "nested")
