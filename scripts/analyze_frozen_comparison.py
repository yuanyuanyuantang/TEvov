"""Offline integrity checks and task-paired analysis for the frozen two-arm run."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import socket
import sqlite3
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tevo.experiment.suite import (collect_results, fingerprints, load_jsonl, load_plan,
    rank_memories, validate_cached_tevo_contexts, validate_cached_per_turn_contexts,
    validate_frozen_bank_index, validate_task_filtered_contexts, task_tool_names)
from tevo.evaluation.statistics import paired_comparison
from tevo.memory.evidence import write_private_json
from tevo.reproducibility import canonical_hash, sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def injected(goal, text):
    return 'Task:\n' + goal + '\n\nSome Related Experience to help you to complete the task:\n' + text


def verify_trial(row, goal, context, per, messages_for_request):
    """Rebuild every prompt; the stored first message contains the LAST refresh."""
    require(row['status'] == 'complete' and not isinstance(row['reward'], bool)
            and row['reward'] in (0, 1), 'incomplete or invalid trial')
    require(row['base_seed'] == row['trial_seed'] == row['sampling']['effective_seed'] == 0
            and row['sampling']['temperature'] == .9, 'sampling drift')
    require(row['frozen_bank_read_only'] and not row.get('tevo_observer_enabled'), 'writable evaluation')
    require(row['memory_context_sha256'] == canonical_hash(context), 'initial context drift')
    require(row['per_turn_memory'] == bool(per), 'retrieval mode drift')
    history = row['task_history']
    logs = row['memory_injection_log']
    turns = per['turns'] if per else [{'turn': 0, 'text': context['text']}]
    require(0 < len(logs) <= len(turns) and (bool(per) or len(logs) == 1), 'injection count')
    if per:
        require(row['per_turn_context_sha256'] == canonical_hash(per), 'per-turn context drift')
    for log, expected in zip(logs, turns):
        require(log['turn'] == expected['turn'], 'injection turn drift')
        # The actor trims its first query; prompts and later lookup identities stay verbatim.
        query_hash = (canonical_hash(goal.strip()) if expected['turn'] == 0
                      else expected['query_text_sha256'])
        require(log['query_sha256'] == query_hash, 'injection query drift')
        require(log['context_sha256'] == canonical_hash(expected['text']), 'injection text drift')
        require(log['first_message_content_sha256'] == canonical_hash(injected(goal, expected['text'])),
                'injection message drift')
        require(log['replaced_previous_block'] == (bool(per) and expected['turn'] > 0), 'block replacement drift')
    require(history[0]['content'] == injected(goal, turns[len(logs)-1]['text']), 'final memory block drift')
    positions = [i for i, message in enumerate(history) if message['role'] == 'assistant']
    require(len(positions) == len(row['request_metrics']), 'request/history mismatch')
    for pos, request in zip(positions, row['request_metrics']):
        require(request['seed'] == 0 and request['status'] == 'success', 'request sampling/error drift')
        prefix = [dict(message) for message in history[:pos]]
        if per:
            turn = sum(message['role'] == 'user' for message in prefix) - 1
            require(0 <= turn < len(logs), 'future or uninjected turn')
            prefix[0]['content'] = injected(goal, turns[turn]['text'])
        require(canonical_hash(messages_for_request(prefix)) == request['prompt_sha256'], 'request prompt drift')
    return len(positions)


def analyze(configs):
    sys.path[:0] = [str(ROOT / 'benchmark/bfcl'),
                   str(ROOT / 'benchmark/bfcl/gorilla/berkeley-function-call-leaderboard')]
    from bfcl_agent import LocalBFCLAgent
    groups, paired, arms, vectors, episode_ids, plans = {}, {}, {}, {}, set(), {}
    for name, filename in configs.items():
        plan = load_plan(filename)
        plans[name] = plan
        require(len(plan['task_ids']) == 150 and len(plan['jobs']) == 600, 'formal grid must be 150 x 4')
        require(plan['config']['seeds'] == [0] and plan['config']['repeats_per_seed'] == 4
                and plan['config']['sampling_seed_mode'] == 'fixed', 'formal sampling grid drift')
        output = ROOT / plan['config']['output_dir']
        manifest = read(output / 'manifest.json')
        require(manifest['fingerprints'] == fingerprints(plan)
                and manifest['manifest_sha256'] == canonical_hash(manifest['fingerprints']), 'provenance drift')
        rows, pending = collect_results(plan, output, manifest['manifest_sha256'])
        require(not pending and len(rows) == 600, 'incomplete formal grid')
        contexts = read(output / 'contexts.json')
        records = read(plan['bank_paths']['memories_path'])
        index = read(plan['bank_paths']['index_path'])
        validate_frozen_bank_index(records, index)
        validate_cached_tevo_contexts(contexts, records, plan['task_ids'])
        tasks = {task['id']: task for task in load_jsonl(Path(plan['dataset_path']))}
        per_turn = any(arm.get('per_turn_memory') for arm in plan['experiments'])
        if plan['config']['frozen'].get('retrieval_tool_filter'):
            validate_task_filtered_contexts(contexts, records, index, tasks, plan['task_ids'], 5)
        if per_turn:
            require(plan['config']['frozen']['retrieval_query_mode'] == 'current_user', 'wrong query mode')
            validate_cached_per_turn_contexts(contexts, records, plan['per_turn_turns'], query_mode='current_user')
            vectors = {task: contexts['tevo_per_turn'][task]['turns'][0]['retrieval_query_vector']
                       for task in plan['task_ids']}
            for task_id, entry in contexts['tevo_per_turn'].items():
                for turn in entry['turns']:
                    kwargs = ({'available_tools': task_tool_names(tasks[task_id])}
                              if plan['config']['frozen'].get('retrieval_tool_filter') else {})
                    require(rank_memories(records, index, turn['retrieval_query_vector'], 5, 'tevo', **kwargs) == turn['memories'],
                            'per-turn rerank mismatch')
        require(canonical_hash(contexts) == read(output / 'artifacts.json')['contexts_sha256'], 'artifact drift')
        groups[name] = {task: [] for task in plan['task_ids']}
        paired[name] = {}
        checked = 0
        for row in rows:
            task = row['task_id']
            per = contexts.get('tevo_per_turn', {}).get(task) if per_turn else None
            checked += verify_trial(row, tasks[task]['messages'][0]['content'], contexts['tevo'][task], per,
                                    LocalBFCLAgent._messages_for_llm_request)
            require(row['experiment'] == name, 'arm drift')
            require(row.get('episode_id') and row['episode_id'] not in episode_ids, 'duplicate episode')
            episode_ids.add(row['episode_id'])
            groups[name][task].append(row)
            key = task, row['repeat_index']
            require(key not in paired[name], 'duplicate repeat')
            paired[name][key] = row
        require(all(len(items) == 4 and {r['repeat_index'] for r in items} == set(range(4))
                    for items in groups[name].values()), 'missing repeat')
        summary = read(output / 'summary.json')
        require(summary['paper_result_eligible'] and summary['provenance_valid'], 'suite engineering checks failed')
        grade = read(output / 'regrade.json')
        require(grade['complete_matrix_regraded'] and grade['matches'] == grade['checked_trials'] == 600
                and grade['blocked_network_attempts'] == 0 and grade['provenance_unchanged']
                and grade['source_manifest_sha256'] == manifest['manifest_sha256'], 'regrade failed')
        expected_checks = {r['key']: r for r in rows}
        require(len(grade['checks']) == 600 and canonical_hash(grade['checks']) == grade['checks_sha256'], 'regrade digest')
        require({c['key'] for c in grade['checks']} == set(expected_checks), 'regrade trial grid')
        for check in grade['checks']:
            require(check['matches'] and check['evaluation_error'] is None
                    and check['stored_reward'] == check['regraded_reward'] == expected_checks[check['key']]['reward'], 'regrade row')
        avg = sum(sum(r['reward'] for r in items)/4 for items in groups[name].values())/150
        passed = sum(any(r['reward'] for r in items) for items in groups[name].values())/150
        require(abs(avg-summary['arms'][name]['avg_at_4']) < 1e-12
                and abs(passed-summary['arms'][name]['pass_at_4']) < 1e-12, 'metric mismatch')
        with sqlite3.connect(output / 'requests.sqlite') as db:
            costs = [dict(zip(('kind', 'status', 'requests', 'reported_tokens', 'unknown_usage'), r))
                     for r in db.execute('SELECT kind,status,count(*),sum(tokens),sum(tokens IS NULL) FROM requests GROUP BY kind,status')]
            last_embed = db.execute("SELECT max(finished) FROM requests WHERE kind='embedding'").fetchone()[0]
            first_llm = db.execute("SELECT min(started) FROM requests WHERE kind='llm'").fetchone()[0]
        require(all(c['status'] == 'success' for c in costs), 'ledger request failure')
        require(last_embed is None or last_embed < first_llm, 'runtime embedding occurred')
        arms[name] = dict(tasks=150, trials=600, avg_at_4=avg, pass_at_4=passed,
                          prompt_hashes_rebuilt=checked, costs=costs, resources=summary['resources'],
                          regrade_sha256=sha256_file(output/'regrade.json'))
    require(plans['control']['task_ids'] == plans['variant']['task_ids'], 'task pairing drift')
    require(plans['control']['config']['model'] == plans['variant']['config']['model'], 'model drift')
    require(plans['control']['config']['budgets'] == plans['variant']['config']['budgets'], 'budget drift')
    require(paired['control'].keys() == paired['variant'].keys(), 'paired repeat grid drift')
    for key, left in paired['control'].items():
        require(all(left[field] == paired['variant'][key][field]
                    for field in ('trial_seed', 'endpoint_index', 'llm_base_url')), 'paired endpoint/seed drift')
    # Control can be reranked offline using the shared, identically batched initial queries.
    control = plans['control']
    ctx = read(ROOT/control['config']['output_dir']/'contexts.json')
    records, index = (read(control['bank_paths'][key]) for key in ('memories_path', 'index_path'))
    tasks = {task['id']: task for task in load_jsonl(Path(control['dataset_path']))}
    for task, vector in vectors.items():
        require(canonical_hash(vector) == ctx['tevo'][task]['query_vector_sha256'], 'initial embedding drift')
        kwargs = ({'available_tools': task_tool_names(tasks[task])}
                  if control['config']['frozen'].get('retrieval_tool_filter') else {})
        require(rank_memories(records, index, vector, 5, 'tevo', **kwargs) == ctx['tevo'][task]['memories'], 'control rerank mismatch')
    return {'passed': True, 'kind': 'frozen_pipeline_bfcl_pass4_comparison', 'total_trials': 1200,
            'arms': arms, 'variant_minus_control': paired_comparison(groups['variant'], groups['control']),
            'primary_metric': 'Avg@4', 'secondary_metric': 'Pass@4',
            'scope': 'Whole pipeline comparison on previously used validation tasks; no isolated component causality, no ReMe superiority claim, no new training data.',
            'uncertainty_unit': 'task, not repeat; fixed seed does not imply deterministic generation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--variant', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def deny(*a, **k):
        raise RuntimeError('Final verification must remain offline')
    with ExitStack() as guards:
        for method in ('connect', 'connect_ex', 'sendto'):
            guards.enter_context(patch.object(socket.socket, method, deny))
        result = analyze({'control': args.control, 'variant': args.variant})
    write_private_json(args.output, result, exclusive=True)
    print(json.dumps({'passed': True, 'arms': result['arms']}))


if __name__ == '__main__':
    main()
