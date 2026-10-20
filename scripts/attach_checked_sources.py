#!/usr/bin/env python3
"""Attach a bounded set of checked TRAIN relations to an existing ordinary bank.

No generation or embedding. Keep original prose, scene keys, index order and
vectors. Relations establish recorded fits, not causal or general applicability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import rewrite_procedural_memory as rw
from tevo.memory.evidence import write_private_json


def attach(records, options, lookup, tasks, episode_tasks):
    """Target-anchored, two-family attachments; ambiguous parameters abstain."""
    result, decisions = [], []
    for original in records:
        refs = {(r['episode_id'], r['call_index']) for r in original['metadata']['evidence_refs']}
        related = []
        for option in options.values():
            if option['kind'] == 'tool_output' and not (
                option['parameter'].endswith('_id') and option['source_path'][-1] == option['parameter']
            ):
                continue  # No semantic inference from coincidentally equal scalars.
            if option['target_tool'] not in original['tool_tags']:
                continue
            if option['kind'] == 'tool_output':
                # A nearby target call does not make every parameter its topic.
                phrase = r'\b' + r'[\s_-]*'.join(map(re.escape, option['parameter'].split('_'))) + r'\b'
                if not re.search(phrase, original['when_to_use'] + ' ' + original['content'], re.I):
                    continue
            if not any((s['source']['episode_id'], s['source']['target_call']) in refs
                       for s in option['supports']):
                continue
            related.append(option)
        decision = {'record_id': original['record_id'], 'options': [o['option_id'] for o in related]}
        counts = Counter((o['target_tool'], o['parameter']) for o in related)
        if not related:
            decision['status'] = 'no_in_scope_target_edge'
        elif max(counts.values()) > 1 or len(related) > 2:
            decision['status'] = 'ambiguous_or_too_many_relations'
        else:
            chosen = {o['option_id']: o for o in related}
            proposal = {'record_id': original['record_id'], 'option_ids': sorted(chosen)}
            try:
                updated = rw.check_attachment(proposal, original, chosen, lookup, tasks, episode_tasks)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                decision.update(status='fallback', reason=str(exc))
            else:
                sources = updated['metadata']['new_parameter_sources']
                by_episode = deepcopy(original['metadata'].get('parameter_sources_by_episode', {}))
                for source in sources:
                    by_episode.setdefault(source['episode_id'], []).append(source)
                updated['source'] = 'trace_checked_attachment'
                updated['metadata'].update(
                    parameter_sources_by_episode=by_episode,
                    publication_status='trace_checks_passed_semantic_transfer_unproven',
                    attachment_scope='capacity_minus_state_or_same_named_response_id',
                    automatic_selection='target_evidence_anchor_and_tool_tag; abstain_on_competing_relations',
                )
                result.append(updated)
                decision.update(status='attached', derived_id=updated['record_id'])
        if decision['status'] != 'attached':
            result.append(deepcopy(original))
        decisions.append(decision)
    return result, decisions


def repair_inventory(trials, tasks):
    """Find local error->successful-retry edges outside the two summary episodes.

    These are missed evidence opportunities, not verified repair explanations.
    """
    groups = rw.fresh.group_trials(trials, tasks)
    rows = []
    for group in groups:
        selected = {t['episode_id'] for t in rw.fresh.procedure_evidence(group)['trajectories']}
        for trial in group['trajectories']:
            calls = trial['calls']
            for i, failed in enumerate(calls):
                if rw.fresh.target_outcome(failed) != 'explicit_error':
                    continue
                retry = next((c for c in calls[i+1:] if c['tool_id'] == failed['tool_id']
                              and c['turn_id'] == failed['turn_id']
                              and rw.fresh.target_outcome(c) == 'local_no_reported_error'), None)
                if retry is not None:
                    rows.append({'task_id': group['task_id'], 'episode_id': trial['episode_id'],
                        'tool': failed['tool_id'], 'failed_call': failed['call_index'],
                        'retry_call': retry['call_index'],
                        'outside_summary_representatives': trial['episode_id'] not in selected})
    return {'edges': rows, 'edge_count': len(rows),
            'outside_representative_edges': sum(r['outside_summary_representatives'] for r in rows),
            'semantic_support_verified': False}


def build(source: Path, trials_dir: Path, output: Path):
    source, trials_dir, output = source.resolve(), trials_dir.resolve(), output.resolve()
    if any(output.is_relative_to(p) for p in (source, trials_dir, rw.fresh.TRAIN.parent)):
        raise ValueError('output must be outside protected inputs')
    manifest = json.loads((source / 'manifest.json').read_text())
    freeze = json.loads((source / 'freeze.json').read_text())
    # This path accepts the current development-bank freeze, not legacy archives.
    frozen = {str((ROOT / p).resolve()): h for p, h in freeze['hashes'].items()}
    for name in ('manifest.json', 'memories.json', 'memories_index.json'):
        path = source / name
        if frozen.get(str(path)) != rw.fresh.sha256_file(path):
            raise ValueError('source freeze mismatch: ' + name)
    for name, digest in manifest['outputs_sha256'].items():
        if rw.fresh.sha256_file(source / name) != digest:
            raise ValueError('source manifest mismatch: ' + name)
    tasks, trials, paths = rw.fresh.load_build_inputs(trials_dir)
    records = json.loads((source / 'memories.json').read_text())
    index = json.loads((source / 'memories_index.json').read_text())
    rw.fresh.validate_frozen_bank_index(records, index)
    options, lookup = rw.compile_options(tasks, trials)
    for record in records:
        if record['type_tag'] != 'generic_rule' or record['record_id'] != 'mem_' + rw.fresh.canonical_hash(
                [record['when_to_use'], record['content'], record['tool_tags']])[:16]:
            raise ValueError('expected ordinary bank with content-derived identities')
        for ref in record['metadata']['evidence_refs']:
            if ref['call_sha256'] != rw.fresh.canonical_hash(lookup[(ref['episode_id'], ref['call_index'])]):
                raise ValueError('source evidence differs from supplied TRAIN trials')
    pinned = paths + [rw.fresh.TRAIN, rw.fresh.VAL, Path(__file__), Path(rw.__file__),
        Path(rw.fresh.__file__), ROOT/'tevo/experiment/suite.py'] + list(source.glob('*.json'))
    pins = {str(p): rw.fresh.sha256_file(p) for p in pinned}
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    episode_tasks = {t['episode_id']: t['task_id'] for t in trials}
    derived, decisions = attach(records, options, lookup, tasks, episode_tasks)
    new_index = {new['record_id']: deepcopy(index[old['record_id']]) for old, new in zip(records, derived)}
    rw.fresh.validate_frozen_bank_index(derived, new_index)
    assert len(derived) == len(records) == len(new_index)
    assert all(new['when_to_use'] == old['when_to_use'] and new['content'].startswith(old['content'])
               for old, new in zip(records, derived))
    rw.write_bank(output / 'bank', derived, new_index)
    write_private_json(output/'decisions.json', decisions, exclusive=True)
    write_private_json(output/'relation_options.json', options, exclusive=True)
    write_private_json(output/'repair_inventory.json', repair_inventory(trials, tasks), exclusive=True)
    if any(rw.fresh.sha256_file(Path(p)) != h for p, h in pins.items()):
        raise ValueError('protected input changed during compilation')
    summary = {'kind': 'bounded_checked_source_attachments', 'records': len(derived),
        'training_trials': len(trials), 'decisions': dict(Counter(d['status'] for d in decisions)),
        'relation_kinds': dict(Counter(options[o]['kind'] for d in decisions if d['status']=='attached'
                                     for o in d['options'])),
        'source_bank': str(source), 'inputs_sha256': pins,
        'outputs_sha256': {name: rw.fresh.sha256_file(output/'bank'/name)
                           for name in ('memories.json', 'memories_index.json')},
        'model_requests': 0, 'embedding_requests': 0,
        'keys_vectors_order_preserved': True, 'original_prose_preserved': True,
        'causal_or_general_applicability_proven': False}
    write_private_json(output/'manifest.json', summary, exclusive=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-bank', 'trials', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    summary = build(args.source_bank, args.trials, args.output)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('sha256')}, ensure_ascii=False))
