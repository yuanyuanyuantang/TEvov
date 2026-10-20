"""Regression checks for trial identity, safe resume and closed request budgets."""

import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from tevo.memory.embedding import _normalize_vector

from tevo.experiment.suite import (
    BudgetExceeded, RequestLedger, _embedding, atomic_json, collect_results, format_memories,
    fingerprints, inspect_providers, load_plan, lookup_per_turn_entry, prepare_contexts,
    rank_memories, summarize, validate_cached_per_turn_contexts, validate_frozen_bank_index,
)
from tevo.reproducibility import canonical_hash, derive_seed, sha256_file


class ExperimentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "answers").mkdir()
        atomic_json(self.root / "answers/BFCL_v3_multi_turn_base.json", {"id": "val_a"})
        (self.root / "train.jsonl").write_text('{"id":"train_a"}\n', encoding="utf-8")
        (self.root / "val.jsonl").write_text('{"id":"val_a","messages":[{"role":"user","content":"Goal"}]}\n', encoding="utf-8")
        self.config = {
            "version": 1, "output_dir": str(self.root / "results"),
            "dataset": {"path": str(self.root / "val.jsonl"), "train_path": str(self.root / "train.jsonl"), "answer_path": str(self.root / "answers")},
            "model": {"base_urls": ["http://localhost:1/v1"], "max_tokens": 100, "temperature": .9},
            "seeds": [10, 20], "budgets": {"max_requests": 2, "max_total_tokens": 100, "wall_seconds": 600, "max_workers": 1},
            "experiments": [{"name": "a", "method": "nomem"}, {"name": "b", "method": "nomem"}],
        }
        self.path = self.root / "config.json"

    def plan(self):
        atomic_json(self.path, self.config)
        return load_plan(self.path)

    def test_offline_plan_preserves_pairing_and_distinct_seeds(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline validation contacted network")):
            plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 4)
        first, second, third, fourth = plan["jobs"]
        self.assertEqual(first["trial_seed"], second["trial_seed"])
        self.assertNotEqual(first["trial_seed"], third["trial_seed"])
        self.assertEqual(first["trial_seed"], derive_seed(10, "val_a"))
        self.assertEqual(len({job["key"] for job in plan["jobs"]}), 4)

    def test_duplicate_seeds_and_overlapping_split_are_rejected(self):
        self.config["seeds"] = [10, 10]
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.plan()
        self.config["seeds"] = [10, 20]
        (self.root / "train.jsonl").write_text('{"id":"val_a"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.plan()

    def test_fixed_seed_repeats_have_distinct_identity_and_complete_four_trial_metrics(self):
        self.config.update(seeds=[0], repeats_per_seed=4, sampling_seed_mode="fixed")
        self.config["model"]["base_urls"] = ["http://localhost:1/v1", "http://localhost:2/v1"]
        plan = self.plan()
        self.assertEqual(len(plan["jobs"]), 8)
        self.assertEqual(len({j["key"] for j in plan["jobs"]}), 8)
        self.assertEqual({j["trial_seed"] for j in plan["jobs"]}, {0})
        for index in range(4):
            a, b = plan["jobs"][index * 2:index * 2 + 2]
            self.assertEqual((a["repeat_index"], b["repeat_index"]), (index, index))
            self.assertEqual(a["endpoint_index"], b["endpoint_index"])
        rows = [{**j, "run_id": j["trial_index"], "status": "complete",
                 "reward": int(j["experiment"] == "a" and j["repeat_index"] == 0),
                 "manifest_sha256": "manifest"} for j in plan["jobs"]]
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        report = summarize(plan, rows, [], ledger)
        self.assertEqual(report["arms"]["a"]["avg_at_4"], .25)
        self.assertEqual(report["arms"]["a"]["pass_at_4"], 1.)
        self.assertEqual(report["arms"]["b"]["pass_at_4"], 0.)
        partial = summarize(plan, rows[:-2], plan["jobs"][-2:], ledger)
        self.assertIsNone(partial["arms"]["a"]["avg_at_4"])
        self.assertIsNone(partial["arms"]["a"]["pass_at_4"])
        duplicate = summarize(plan, [rows[0], rows[0], *rows[2:]], [], ledger)
        self.assertFalse(duplicate["all_trials_complete"])
        self.assertIsNone(duplicate["arms"]["b"]["pass_at_4"])
        for row in rows:
            atomic_json(self.root / "trials" / (row["key"] + ".json"), row)
        self.assertEqual(len(collect_results(plan, self.root, "manifest")[0]), 8)
        changed = {**rows[0], "repeat_index": 3}
        atomic_json(self.root / "trials" / (changed["key"] + ".json"), changed)
        with self.assertRaisesRegex(ValueError, "repeat identity"):
            collect_results(plan, self.root, "manifest")

    def test_invalid_fixed_repeat_configuration_is_rejected(self):
        for changes in ({"repeats_per_seed": 0}, {"repeats_per_seed": True},
                        {"repeats_per_seed": 4}, {"sampling_seed_mode": "unknown"},
                        {"sampling_seed_mode": "fixed"}):
            with self.subTest(changes=changes):
                original = dict(self.config)
                self.config.update(changes)
                with self.assertRaises(ValueError):
                    self.plan()
                self.config = original

    def test_context_policy_is_global_and_rejects_estimated_or_boolean_windows(self):
        policy = {"policy": "server_tokenize_headroom", "context_window_tokens": 32768}
        self.config["model"]["context_budget"] = policy
        plan = self.plan()
        self.assertEqual(plan["config"]["model"]["context_budget"], policy)
        self.config["model"]["context_budget"] = {**policy, "context_window_tokens": True}
        with self.assertRaisesRegex(ValueError, "context_window_tokens"):
            self.plan()

    def test_each_arm_is_counterbalanced_across_endpoints(self):
        rows = [{"id": f"val_{index}", "messages": [{"role": "user", "content": "Goal"}]} for index in range(4)]
        (self.root / "val.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        self.config["model"]["base_urls"] = ["http://localhost:1/v1", "http://localhost:2/v1"]
        self.config["seeds"] = [10]
        plan = self.plan()
        for arm in ("a", "b"):
            jobs = [job for job in plan["jobs"] if job["experiment"] == arm]
            self.assertEqual([job["endpoint_index"] for job in jobs].count(0), 2)
            self.assertEqual([job["endpoint_index"] for job in jobs].count(1), 2)

    def test_resume_refuses_copied_or_changed_trial_provenance(self):
        plan = self.plan()
        job = plan["jobs"][0]
        row = {**job, "manifest_sha256": "expected", "status": "complete", "reward": 1}
        atomic_json(self.root / "trials" / (job["key"] + ".json"), row)
        complete, pending = collect_results(plan, self.root, "expected")
        self.assertEqual((len(complete), len(pending)), (1, 3))
        row["base_seed"] = 99
        atomic_json(self.root / "trials" / (job["key"] + ".json"), row)
        with self.assertRaisesRegex(ValueError, "identity/provenance"):
            collect_results(plan, self.root, "expected")

    def test_budget_reserves_errors_and_keeps_unknown_usage(self):
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        first = ledger.reserve("a", "llm")
        ledger.finish(first, None, "error")
        second = ledger.reserve("b", "llm")
        ledger.finish(second, {"total_tokens": 50}, "success")
        with self.assertRaises(BudgetExceeded):
            ledger.reserve("c", "llm")
        resource = ledger.summary()
        self.assertEqual(resource["requests"], 2)
        self.assertEqual(resource["requests_with_unknown_token_usage"], 1)
        self.assertEqual(resource["reported_total_tokens"], 50)

    def test_ledger_connections_close_after_transaction(self):
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        with ledger.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
        with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
            connection.execute("SELECT 1")

    def test_ledger_reader_does_not_block_concurrent_atomic_budget(self):
        budgets = {**self.config["budgets"], "max_requests": 64, "max_total_tokens": 1000}
        path = self.root / "requests.sqlite"
        ledger = RequestLedger(path, budgets)
        reader = sqlite3.connect(path)
        self.addCleanup(reader.close)
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM requests").fetchone()
        def reserve_many(worker):
            local = RequestLedger(path, budgets)
            ids = []
            for number in range(32):
                try:
                    request = local.reserve(f"{worker}:{number}", "llm")
                except BudgetExceeded:
                    break
                local.finish(request, {"total_tokens": 1}, "success")
                ids.append(request)
            return ids
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = [request for group in pool.map(reserve_many, range(4)) for request in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 64)
        self.assertEqual(ledger.summary()["reported_total_tokens"], 64)

    def test_partial_matrix_cannot_report_best_k_or_paper_eligibility(self):
        plan = self.plan()
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        row = {**plan["jobs"][0], "reward": 1, "status": "complete"}
        report = summarize(plan, [row], plan["jobs"][1:], ledger)
        self.assertFalse(report["all_trials_complete"])
        self.assertFalse(report["paper_result_eligible"])
        self.assertIsNone(report["arms"]["a"]["best_of_observed_seeds"])

    def test_reme_recall_deduplicates_after_top_k_without_refilling(self):
        records = [{"memory_id": "a", "content": "same", "when_to_use": "Goal"},
                   {"memory_id": "b", "content": "same", "when_to_use": "Goal"},
                   {"memory_id": "c", "content": "different", "when_to_use": "Other"}]
        index = {"a": {"vector": [1., 0.]}, "b": {"vector": [.9, .1]}, "c": {"vector": [0., 1.]}}
        chosen = rank_memories(records, index, [1., 0.], 2, "reme")
        self.assertEqual([record["id"] for record in chosen], ["a"])
        self.assertEqual(format_memories(chosen, "reme"), "Experience 1 :\n When to use: Goal\n Content: same\n")

    def test_tevo_rounding_ties_and_injection_format(self):
        records = [
            {"record_id": "generic", "type_tag": "generic_rule", "when_to_use": " Goal ", "content": " Generic repair "},
            {"record_id": "repair", "type_tag": "failure_repair", "when_to_use": "Goal", "content": "Specific repair"},
            {"record_id": "stale", "type_tag": "success_trajectory", "content": "Excluded", "stale": True},
        ]
        index = {"generic": {"vector": [1., 0.]}, "repair": {"vector": [1., .00001]}, "stale": {"vector": [1., 0.]}}
        actual = rank_memories(records, index, [1., 0.], 2, "tevo")
        self.assertEqual([r["id"] for r in actual], ["repair", "generic"])
        self.assertEqual(format_memories(actual, "tevo"),
                         "Experience 1:\nWhen to use: Goal\nSpecific repair\n\n"
                         "Experience 2:\nWhen to use: Goal\nGeneric repair")

    def test_task_tool_filter_refills_before_topk_and_keeps_legacy_default(self):
        records = [{'record_id': i, 'when_to_use': i, 'content': i, 'tool_tags': tags}
                   for i, tags in [('wrong', ['other']), ('mixed', ['read', 'other']),
                                   ('empty', []), ('read', ['read']), ('write', ['write'])]]
        index = {r['record_id']: {'vector': [1., 0.]} for r in records}
        self.assertEqual([r['id'] for r in rank_memories(records, index, [1., 0.], 2, 'tevo')], ['wrong', 'mixed'])
        self.assertEqual([r['id'] for r in rank_memories(records, index, [1., 0.], 2, 'tevo',
                                                       available_tools={'read', 'write'})], ['read', 'write'])
        self.assertEqual(rank_memories(records, index, [1., 0.], 2, 'tevo', available_tools={'absent'}), [])

    def test_filtered_prepare_cache_rejects_unfiltered_and_tampered_rank_without_network(self):
        record = {'record_id': 'm', 'when_to_use': 'Goal', 'content': 'Do it', 'tool_tags': ['read']}
        index = {'m': {'search_text': 'Goal', 'vector': [1., 0.]}}
        atomic_json(self.root/'bank.json', [record]); atomic_json(self.root/'index.json', index)
        row = {'id': 'val_a', 'messages': [{'role': 'user', 'content': 'Goal'}],
               'extra': {'function': [{'name': 'read'}]}}
        (self.root/'val.jsonl').write_text(json.dumps(row)+'\n')
        self.config['experiments'] = [{'name': 'a', 'method': 'memonly'}]
        self.config['frozen'] = {'memories_path': str(self.root/'bank.json'), 'index_path': str(self.root/'index.json'),
                                 'embedding': {'base_url': 'http://localhost:1'}, 'retrieval_tool_filter': True}
        plan = self.plan(); ledger = RequestLedger(self.root/'requests.sqlite', self.config['budgets'])
        with patch('tevo.experiment.suite._embedding', return_value=([[1., 0.]], [])):
            contexts = prepare_contexts(plan, self.root, ledger)
        with patch('requests.post', side_effect=AssertionError('cache must remain offline')):
            self.assertEqual(prepare_contexts(plan, self.root, ledger), contexts)
            plan['config']['frozen']['retrieval_tool_filter'] = False
            with self.assertRaisesRegex(ValueError, 'filter identity'):
                prepare_contexts(plan, self.root, ledger)
            plan['config']['frozen']['retrieval_tool_filter'] = True
            del contexts['retrieval_tool_filter']; atomic_json(self.root/'contexts.json', contexts)
            with self.assertRaisesRegex(ValueError, 'filter identity'):
                prepare_contexts(plan, self.root, ledger)
            contexts['retrieval_tool_filter'] = True
            contexts['tevo']['val_a'].update(memories=[], text=format_memories([], 'tevo'))
            atomic_json(self.root/'contexts.json', contexts)
            with self.assertRaisesRegex(ValueError, 'filtered ranking'):
                prepare_contexts(plan, self.root, ledger)
        self.assertEqual(ledger.summary()['requests'], 0)

    def test_unverified_reme_adapter_blocks_paper_eligibility(self):
        plan = self.plan()
        plan["experiments"] = [{"name": "reme", "method": "reme_public96"}]
        plan["jobs"] = [{**plan["jobs"][0], "experiment": "reme"}, {**plan["jobs"][2], "experiment": "reme"}]
        rows = [{**job, "status": "complete", "reward": 1} for job in plan["jobs"]]
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        report = summarize(plan, rows, [], ledger)
        self.assertTrue(report["all_trials_complete"])
        self.assertFalse(report["baseline_adapter_verified"])
        self.assertFalse(report["paper_result_eligible"])

    def test_advertised_model_mismatch_is_rejected(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"id": "other", "max_model_len": 4096}]})
        model = {"name": "requested", "base_urls": ["http://localhost:1/v1"], "max_tokens": 100}
        with patch("requests.get", return_value=response), self.assertRaisesRegex(ValueError, "not uniquely advertised"):
            inspect_providers(model)

    def test_artifact_manifest_identity_and_hash_are_enforced(self):
        artifact = self.root / "model_artifacts.json"
        self.config["model"].update({"name": "local-model", "artifact_manifest_path": str(artifact)})
        atomic_json(artifact, {"model_id": "different-model", "files": {}})
        with self.assertRaisesRegex(ValueError, "model_id"):
            self.plan()
        atomic_json(artifact, {"model_id": "local-model", "files": {"weights": "first"}})
        plan = self.plan()
        before = fingerprints(plan)["model_artifacts"]["manifest_sha256"]
        atomic_json(artifact, {"model_id": "local-model", "files": {"weights": "changed"}})
        after = fingerprints(plan)["model_artifacts"]["manifest_sha256"]
        self.assertNotEqual(before, after)

    def test_reme_dimensions_are_validated_without_sending_unsupported_parameter(self):
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": [{"index": 0, "embedding": [1., 0.]}], "usage": {"total_tokens": 1}})
        settings = {"base_url": "http://localhost:1/v1", "model": "embedding", "dimensions": 2, "send_dimensions": False}
        with patch("requests.post", return_value=response) as post:
            vectors, events = _embedding(["query"], settings, ledger, "embedding")
        self.assertEqual(vectors, [[1., 0.]])
        self.assertNotIn("dimensions", post.call_args.kwargs["json"])
        response.json = lambda: {"data": [{"index": 0, "embedding": [1.]}]}
        with patch("requests.post", return_value=response), self.assertRaisesRegex(ValueError, "invalid vectors"):
            _embedding(["query"], settings, ledger, "embedding")

    def test_frozen_bank_index_divergence_fails_closed(self):
        records = [{"record_id": "m1", "type_tag": "generic_rule", "when_to_use": "Goal A", "content": "c1"}]
        # v5 旧索引（审计前缀 + content）配 v6 记录 → 必须拒绝
        with self.assertRaisesRegex(ValueError, "index text mismatch"):
            validate_frozen_bank_index(records, {"m1": {"search_text": "Only for goals matching this observed condition: Goal A", "vector": [1., 0.]}})
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            validate_frozen_bank_index(records, {})
        with self.assertRaisesRegex(ValueError, "invalid vector"):
            validate_frozen_bank_index(records, {"m1": {"search_text": "Goal A", "vector": [float("nan"), 0.]}})
        with self.assertRaisesRegex(ValueError, "invalid vector"):
            validate_frozen_bank_index(records, {"m1": {"search_text": "Goal A", "vector": []}})
        two = records + [{"record_id": "m2", "type_tag": "generic_rule", "when_to_use": "Goal B", "content": "c2"}]
        with self.assertRaisesRegex(ValueError, "inconsistent vector dimensions"):
            validate_frozen_bank_index(two, {"m1": {"search_text": "Goal A", "vector": [1., 0.]},
                                             "m2": {"search_text": "Goal B", "vector": [1., 0., 0.]}})
        validate_frozen_bank_index(records, {"m1": {"search_text": "Goal A", "vector": [1., 0.]}})

    def test_prepare_contexts_rejects_stale_index_before_retrieval(self):
        memories = self.root / "memories.json"
        index = self.root / "index.json"
        memories.write_text(json.dumps([{"record_id": "m1", "type_tag": "generic_rule", "when_to_use": "Goal", "content": "c1"}]), encoding="utf-8")
        index.write_text(json.dumps({"m1": {"search_text": "Only for goals matching this observed condition: Goal", "vector": [1., 0.]}}), encoding="utf-8")
        self.config["experiments"] = [{"name": "a", "method": "memonly"}]
        self.config["frozen"] = {"memories_path": str(memories), "index_path": str(index), "top_k": 5,
                                 "embedding": {"base_url": "http://localhost:1/v1", "model": "embedding"}}
        plan = self.plan()
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        # 顺序修复：银行 CPU 校验先于 query embedding，坏库不发出任何请求
        with patch("requests.post", side_effect=AssertionError("stale bank must fail before any embedding request")), \
                self.assertRaisesRegex(ValueError, "index text mismatch"):
            prepare_contexts(plan, self.root, ledger)
        self.assertFalse((self.root / "contexts.json").exists())

    def test_prepare_contexts_revalidates_cached_contexts_against_current_bank(self):
        memories = self.root / "memories.json"
        index = self.root / "index.json"
        record = {"record_id": "m1", "type_tag": "generic_rule", "when_to_use": "Goal A", "content": "c1"}
        memories.write_text(json.dumps([record]), encoding="utf-8")
        index.write_text(json.dumps({"m1": {"search_text": "Goal A", "vector": [1., 0.]}}), encoding="utf-8")
        self.config["experiments"] = [{"name": "a", "method": "memonly"}]
        self.config["frozen"] = {"memories_path": str(memories), "index_path": str(index), "top_k": 5,
                                 "embedding": {"base_url": "http://localhost:1/v1", "model": "embedding"}}
        plan = self.plan()
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        cached = {"tevo": {"val_a": {"text": "Experience 1:\nWhen to use: Goal A\nc1",
                                     "memories": [{**record, "similarity": 1.0, "id": "m1", "tevo_type": "generic_rule"}],
                                     "query_vector_sha256": "hash"}},
                  "reme": {}, "embedding_events": []}
        atomic_json(self.root / "contexts.json", cached)
        with patch("requests.post", side_effect=AssertionError("cache validation must not contact the network")):
            self.assertEqual(prepare_contexts(plan, self.root, ledger), cached)
        # 旧代缓存（v5 审计口径 when_to_use）配新银行 → fail closed，不静默打分
        cached["tevo"]["val_a"]["memories"][0]["when_to_use"] = "Only for goals matching this observed condition: Goal A"
        atomic_json(self.root / "contexts.json", cached)
        with self.assertRaisesRegex(ValueError, "cached retrieval contexts val_a: memory mismatch"):
            prepare_contexts(plan, self.root, ledger)
        # 缓存引用银行中不存在的记忆 → fail closed
        cached["tevo"]["val_a"]["memories"][0] = {"id": "m9", "when_to_use": "Goal A", "content": "c1"}
        atomic_json(self.root / "contexts.json", cached)
        with self.assertRaisesRegex(ValueError, "unknown memory"):
            prepare_contexts(plan, self.root, ledger)
        # 缓存命中同样校验银行自身：旧索引配新记录 → 拒绝
        index.write_text(json.dumps({"m1": {"search_text": "Only for goals matching this observed condition: Goal A", "vector": [1., 0.]}}), encoding="utf-8")
        atomic_json(self.root / "contexts.json", cached)
        with self.assertRaisesRegex(ValueError, "index text mismatch"):
            prepare_contexts(plan, self.root, ledger)
        self.assertEqual(ledger.summary()["requests"], 0)

    def test_prepare_contexts_rejects_tampered_text_missing_fields_or_tasks(self):
        # 反例（第三批复核）：text 一致性 / memories 结构 / 计划任务覆盖三项检查；合法空召回不得误伤
        memories = self.root / "memories.json"
        index = self.root / "index.json"
        record = {"record_id": "m1", "type_tag": "generic_rule", "when_to_use": "Goal A", "content": "c1"}
        memories.write_text(json.dumps([record]), encoding="utf-8")
        index.write_text(json.dumps({"m1": {"search_text": "Goal A", "vector": [1., 0.]}}), encoding="utf-8")
        self.config["experiments"] = [{"name": "a", "method": "memonly"}]
        self.config["frozen"] = {"memories_path": str(memories), "index_path": str(index), "top_k": 5,
                                 "embedding": {"base_url": "http://localhost:1/v1", "model": "embedding"}}
        plan = self.plan()
        ledger = RequestLedger(self.root / "requests.sqlite", self.config["budgets"])
        memory = {**record, "similarity": 1.0, "id": "m1", "tevo_type": "generic_rule"}
        good_text = "Experience 1:\nWhen to use: Goal A\nc1"
        with patch("requests.post", side_effect=AssertionError("cache validation must not contact the network")):
            # memories 与银行一致但 text 被篡改（旧式标头）→ 拒绝：模型实际读取的正文不可信
            atomic_json(self.root / "contexts.json", {"tevo": {"val_a": {"memories": [memory], "text": "[generic_rule] " + good_text}},
                                                      "reme": {}, "embedding_events": []})
            with self.assertRaisesRegex(ValueError, "text mismatch"):
                prepare_contexts(plan, self.root, ledger)
            # 缺 memories 字段（旧 text 完好）→ 拒绝结构缺失，不静默放过未验证正文
            atomic_json(self.root / "contexts.json", {"tevo": {"val_a": {"text": good_text}},
                                                      "reme": {}, "embedding_events": []})
            with self.assertRaisesRegex(ValueError, "entry malformed"):
                prepare_contexts(plan, self.root, ledger)
            # tevo 分区缺当前计划任务 → 拒绝任务覆盖缺失（空上下文不能悄悄通过）
            atomic_json(self.root / "contexts.json", {"tevo": {}, "reme": {}, "embedding_events": []})
            with self.assertRaisesRegex(ValueError, "miss plan tasks"):
                prepare_contexts(plan, self.root, ledger)
            # 合法空召回：显式 memories=[] + 规范空文本 → 仍然通过
            empty = {"tevo": {"val_a": {"memories": [], "text": "No relevant procedural experiences found."}},
                     "reme": {}, "embedding_events": []}
            atomic_json(self.root / "contexts.json", empty)
            self.assertEqual(prepare_contexts(plan, self.root, ledger), empty)
        self.assertEqual(ledger.summary()["requests"], 0)

    def test_reme_method_is_disabled_after_reset(self):
        self.config["experiments"] = [{"name": "reme", "method": "reme_public96"}]
        with patch("requests.post", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "invalid experiment"):
                self.plan()

    # ------------------------------------------------------------------
    # 逐轮检索两臂：plan 校验、冻结查找、缓存校验与 prepare 交叉复核链
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_vector(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [1.0 + digest[0], 1.0 + digest[1], 0.5 + digest[2] / 255.0]

    @classmethod
    def _vector(cls, text):
        return _normalize_vector(cls._raw_vector(text))

    @staticmethod
    def _embedding_post(texts):
        rows = [{"index": index, "embedding": [1.0 + digest_value[0], 1.0 + digest_value[1], 0.5 + digest_value[2] / 255.0]}
                for index, digest_value in enumerate(hashlib.sha256(text.encode("utf-8")).digest() for text in texts)]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": rows, "usage": None})

    def _per_turn_fixture(self, turn_texts=None):
        """Two tasks with two dataset user turns each, a small frozen bank, a
        pilot-style contexts file and a passed diagnostic report, all on the
        deterministic fake embedding used by the patched requests.post."""
        turn_texts = turn_texts or {"val_a": ["Goal A", "Follow-up A"], "val_b": ["Goal B", "Follow-up B"]}
        rows = []
        for task_id, turns in turn_texts.items():
            question = [[json.dumps({"role": "user", "content": turn}), []] for turn in turns]
            rows.append({"id": task_id, "messages": [{"role": "user", "content": turns[0]}], "tools": [],
                         "extra": {"question": question}})
        (self.root / "val.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        records = [{"record_id": f"m{index}", "type_tag": "generic_rule",
                    "when_to_use": f"Scenario {index}", "content": f"Body {index}"} for index in range(4)]
        index = {record["record_id"]: {"search_text": record["when_to_use"],
                                       "vector": self._vector(record["when_to_use"])} for record in records}
        (self.root / "memories.json").write_text(json.dumps(records), encoding="utf-8")
        (self.root / "index.json").write_text(json.dumps(index), encoding="utf-8")
        top_k = 2
        pilot = {"tevo": {}}
        report = {"passed": True, "query_vectors": {}, "tasks": {}}
        for task_id, turns in turn_texts.items():
            queries, parts = [], []
            for turn in turns:
                parts.append(turn)
                queries.append("\n".join(parts))
            vector = self._vector(queries[0])
            chosen = rank_memories(records, index, vector, top_k, "tevo")
            pilot["tevo"][task_id] = {"text": format_memories(chosen, "tevo"), "memories": chosen,
                                      "query_vector_sha256": canonical_hash(vector)}
            report["query_vectors"][task_id] = [{"turn": turn_index, "text_sha256": canonical_hash(query),
                                                 "vector": self._vector(query)} for turn_index, query in enumerate(queries)]
            report["tasks"][task_id] = {"turn_rows": [
                {"top5": [{"id": memory["id"]} for memory in rank_memories(records, index, self._vector(query), top_k, "tevo")]}
                for query in queries]}
        atomic_json(self.root / "pilot_contexts.json", pilot)
        atomic_json(self.root / "diagnostic_report.json", report)
        self.config["dataset"]["task_ids"] = list(turn_texts)
        self.config["experiments"] = [{"name": "pt", "method": "memonly", "per_turn_memory": True}]
        self.config["frozen"] = {
            "memories_path": str(self.root / "memories.json"), "index_path": str(self.root / "index.json"),
            "top_k": top_k, "embedding": {"base_url": "http://localhost:1/v1", "model": "embedding", "dimensions": 3},
            "per_turn_reference": {"pilot_contexts_path": str(self.root / "pilot_contexts.json"),
                                   "diagnostic_report_path": str(self.root / "diagnostic_report.json")},
        }
        return records, index, top_k

    def _prepare_with_fake_embedding(self, plan, ledger_name="requests.sqlite"):
        def fake_post(url=None, json=None, headers=None, timeout=None):
            return self._embedding_post(json["input"])
        ledger = RequestLedger(self.root / ledger_name, self.config["budgets"])
        with patch("requests.post", side_effect=fake_post):
            contexts = prepare_contexts(plan, self.root, ledger)
        return contexts, ledger

    def test_per_turn_plan_validation_and_reference_requirement(self):
        self._per_turn_fixture()
        self.config["experiments"][0]["method"] = "nomem"
        with self.assertRaisesRegex(ValueError, "memory-reading method"):
            self.plan()
        self.config["experiments"][0]["method"] = "memonly"
        self.config["experiments"][0]["per_turn_memory"] = "yes"
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            self.plan()
        self.config["experiments"][0]["per_turn_memory"] = True
        del self.config["frozen"]["per_turn_reference"]["diagnostic_report_path"]
        with self.assertRaisesRegex(ValueError, "per_turn_reference.diagnostic_report_path"):
            self.plan()
        self.config["frozen"]["per_turn_reference"]["diagnostic_report_path"] = str(self.root / "missing.json")
        with self.assertRaises(FileNotFoundError):
            self.plan()
        self.config["frozen"]["per_turn_reference"]["diagnostic_report_path"] = str(self.root / "diagnostic_report.json")
        plan = self.plan()
        self.assertEqual(plan["per_turn_turns"]["val_a"], ["Goal A", "Follow-up A"])
        # 数据集内首轮与 extra.question 不一致 → 拒绝启动（查询规则不可信）
        rows = [json.loads(line) for line in (self.root / "val.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["messages"][0]["content"] = "Different first message"
        (self.root / "val.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from messages"):
            self.plan()

    def test_per_turn_contexts_cross_check_chain_and_cache_fail_closed(self):
        records, index, top_k = self._per_turn_fixture()
        plan = self.plan()
        contexts, ledger = self._prepare_with_fake_embedding(plan, "requests_a.sqlite")
        # 两个 POST：首轮 2 条复用 tevo 批次 + 后续轮 2 条
        self.assertEqual(ledger.summary()["requests"], 2)
        self.assertEqual(set(contexts["tevo_per_turn"]), {"val_a", "val_b"})
        for task_id in ("val_a", "val_b"):
            entry = contexts["tevo_per_turn"][task_id]
            self.assertEqual(entry["user_turns"], 2)
            self.assertEqual([row["turn"] for row in entry["turns"]], [0, 1])
            self.assertEqual(entry["turns"][1]["query_text"], "\n".join(plan["per_turn_turns"][task_id]))
            for row in entry["turns"]:
                self.assertEqual(row["query_vector_sha256"], canonical_hash(self._vector(row["query_text"])))
                self.assertEqual(row["text"], format_memories(row["memories"], "tevo"))
            # 首轮逐轮行与固定首轮分区一致（Q_0 同键同结果）
            self.assertEqual(entry["turns"][0]["memories"], contexts["tevo"][task_id]["memories"])
        self.assertEqual(contexts["per_turn_reference"]["diagnostic_report_sha256"],
                         sha256_file(self.root / "diagnostic_report.json"))
        # 缓存命中：不再请求网络；逐轮分区仍被校验
        with patch("requests.post", side_effect=AssertionError("cache validation must not contact the network")):
            self.assertEqual(prepare_contexts(plan, self.root, ledger), contexts)
        # 篡改逐轮缓存轮文本 → fail closed
        broken = json.loads(json.dumps(contexts))
        broken["tevo_per_turn"]["val_a"]["turns"][1]["query_text"] = "Goal A\nDifferent"
        with self.assertRaisesRegex(ValueError, "query text mismatch"):
            validate_cached_per_turn_contexts(broken, records, plan["per_turn_turns"])
        # 篡改缓存记忆正文 → fail closed
        broken = json.loads(json.dumps(contexts))
        broken["tevo_per_turn"]["val_b"]["turns"][0]["memories"][0]["content"] = "Tampered"
        with self.assertRaisesRegex(ValueError, "memory mismatch"):
            validate_cached_per_turn_contexts(broken, records, plan["per_turn_turns"])
        # 轮数不齐 → fail closed
        broken = json.loads(json.dumps(contexts))
        broken["tevo_per_turn"]["val_a"]["turns"].pop()
        with self.assertRaisesRegex(ValueError, "turn count mismatch"):
            validate_cached_per_turn_contexts(broken, records, plan["per_turn_turns"])

    def test_per_turn_contexts_abort_on_reference_divergence(self):
        reference_dir = self.root
        self._per_turn_fixture()
        plan = self.plan()
        # 报告未通过 → 拒绝
        report = json.loads((reference_dir / "diagnostic_report.json").read_text(encoding="utf-8"))
        report["passed"] = False
        atomic_json(reference_dir / "diagnostic_report.json", report)
        with self.assertRaisesRegex(ValueError, "not a passed diagnostic report"):
            self._prepare_with_fake_embedding(plan, "requests_b.sqlite")
        report["passed"] = True
        atomic_json(reference_dir / "diagnostic_report.json", report)
        # 报告向量与本次重嵌不一致 → 拒绝（批次/端点漂移不静默通过）
        report["query_vectors"]["val_a"][1]["vector"][0] += 1e-3
        atomic_json(reference_dir / "diagnostic_report.json", report)
        with self.assertRaisesRegex(ValueError, "diverges from the reference report"):
            self._prepare_with_fake_embedding(plan, "requests_c.sqlite")
        report["query_vectors"]["val_a"][1]["vector"][0] -= 1e-3
        # 报告 top-5 与本次复算不一致 → 拒绝
        report["tasks"]["val_a"]["turn_rows"][1]["top5"] = list(reversed(report["tasks"]["val_a"]["turn_rows"][1]["top5"]))
        atomic_json(reference_dir / "diagnostic_report.json", report)
        with self.assertRaisesRegex(ValueError, "retrieval diverges"):
            self._prepare_with_fake_embedding(plan, "requests_d.sqlite")

    def test_per_turn_contexts_abort_on_pilot_context_divergence(self):
        records, index, top_k = self._per_turn_fixture()
        pilot = json.loads((self.root / "pilot_contexts.json").read_text(encoding="utf-8"))
        pilot["tevo"]["val_a"]["memories"][0]["content"] = "Tampered pilot memory"
        atomic_json(self.root / "pilot_contexts.json", pilot)
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "turn-0 retrieval diverges from the pilot contexts"):
            self._prepare_with_fake_embedding(plan, "requests_e.sqlite")

    def test_lookup_per_turn_entry_is_exact_text_and_fails_closed(self):
        entry = {"turns": [{"turn": 0, "query_text": "Goal A", "text": "t0"},
                           {"turn": 1, "query_text": "Goal A\nFollow-up A", "text": "t1"}]}
        self.assertEqual(lookup_per_turn_entry(entry, "Goal A\nFollow-up A")["text"], "t1")
        with self.assertRaisesRegex(ValueError, "not in the frozen per-turn contexts"):
            lookup_per_turn_entry(entry, "Goal A\nOther")
        with self.assertRaisesRegex(ValueError, "frozen per-turn contexts are missing"):
            lookup_per_turn_entry({}, "Goal A")

    def test_paired_arms_share_one_endpoint_per_task_and_seed(self):
        rows = [{"id": f"val_{index}", "messages": [{"role": "user", "content": "Goal"}]} for index in range(4)]
        (self.root / "val.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        self.config["model"]["base_urls"] = ["http://localhost:1/v1", "http://localhost:2/v1"]
        self.config["seeds"] = [10]
        plan = self.plan()
        by_task = {}
        for job in plan["jobs"]:
            by_task.setdefault(job["task_id"], {})[job["experiment"]] = job["endpoint_index"]
        for task_id, assignments in by_task.items():
            self.assertEqual(len(set(assignments.values())), 1, task_id)
        # 跨任务仍然两两均衡（每臂 2/2）
        for arm in ("a", "b"):
            counts = [job["endpoint_index"] for job in plan["jobs"] if job["experiment"] == arm]
            self.assertEqual((counts.count(0), counts.count(1)), (2, 2))

    def test_current_user_queries_need_no_historical_reference_and_reject_cache_drift(self):
        records, _, _ = self._per_turn_fixture()
        self.config['frozen'].pop('per_turn_reference')
        self.config['frozen']['retrieval_query_mode'] = 'current_user'
        plan = self.plan()
        seen = []
        def fake_post(url=None, json=None, headers=None, timeout=None):
            seen.extend(json['input'])
            return self._embedding_post(json['input'])
        ledger = RequestLedger(self.root / 'requests.sqlite', self.config['budgets'])
        with patch('requests.post', side_effect=fake_post):
            contexts = prepare_contexts(plan, self.root, ledger)
        self.assertEqual(seen, ['Goal A', 'Goal B', 'Follow-up A', 'Follow-up B'])
        row = contexts['tevo_per_turn']['val_a']['turns'][1]
        self.assertEqual(row['query_text'], 'Goal A\nFollow-up A')  # Runtime lookup identity.
        self.assertEqual(row['retrieval_query_text'], 'Follow-up A')
        self.assertEqual(row['retrieval_query_vector'], self._vector('Follow-up A'))
        with patch('requests.post', side_effect=AssertionError('cached load must be offline')):
            self.assertEqual(prepare_contexts(plan, self.root, ledger), contexts)
            for field, bad in [('retrieval_query_text', 'Future goal'),
                               ('retrieval_query_sha256', 'wrong'),
                               ('retrieval_query_vector', [float('nan')]),
                               ('query_vector_sha256', 'wrong')]:
                with self.subTest(field=field):
                    broken = json.loads(json.dumps(contexts))
                    broken['tevo_per_turn']['val_a']['turns'][1][field] = bad
                    (self.root / 'contexts.json').write_text(json.dumps(broken), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        prepare_contexts(plan, self.root, ledger)
        self.assertEqual(ledger.summary()['requests'], 2)
        self.assertEqual(len(records), 4)

    def test_current_user_query_configuration_is_explicit_and_unambiguous(self):
        self._per_turn_fixture()
        self.config['frozen']['retrieval_query_mode'] = 'current_user'
        with self.assertRaisesRegex(ValueError, 'cannot use cumulative'):
            self.plan()
        self.config['frozen'].pop('per_turn_reference')
        self.config['experiments'][0]['per_turn_memory'] = False
        with self.assertRaisesRegex(ValueError, 'requires a per_turn_memory'):
            self.plan()
        self.config['experiments'][0]['per_turn_memory'] = True
        self.config['frozen']['retrieval_query_mode'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'retrieval_query_mode'):
            self.plan()


if __name__ == "__main__":
    unittest.main()
