"""Prompt reconstruction must detect drift, including earlier refreshes."""
import copy
import unittest
from scripts.analyze_frozen_comparison import injected, verify_trial
from tevo.reproducibility import canonical_hash


class FrozenComparisonTest(unittest.TestCase):
    def fixture(self, per_turn, goal='Original task'):
        context = {'text': 'first memory', 'memories': []}
        per = {'turns': [dict(turn=0, text=context['text'], query_text_sha256=canonical_hash(goal)),
                         dict(turn=1, text='second memory', query_text_sha256=canonical_hash(goal+'\nNext'))]} if per_turn else None
        texts = [t['text'] for t in per['turns']] if per else [context['text']]
        history = [{'role':'user','content':injected(goal,texts[-1])}, {'role':'assistant','content':'response'}]
        requests = [{'seed':0,'status':'success','prompt_sha256':canonical_hash([{'role':'user','content':injected(goal,texts[0])}])}]
        if per:
            history += [{'role':'user','content':'Next'}, {'role':'assistant','content':'response 2'}]
            requests += [{'seed':0,'status':'success','prompt_sha256':canonical_hash(history[:3])}]
        logs = [dict(turn=i, query_sha256=canonical_hash(goal.strip() if i==0 else goal+'\nNext'),
                     context_sha256=canonical_hash(text), first_message_content_sha256=canonical_hash(injected(goal,text)),
                     replaced_previous_block=bool(per) and i>0) for i,text in enumerate(texts)]
        row = dict(status='complete', reward=0., base_seed=0,trial_seed=0,
                   sampling={'effective_seed':0,'temperature':.9}, frozen_bank_read_only=True,
                   memory_context_sha256=canonical_hash(context), per_turn_memory=bool(per),
                   per_turn_context_sha256=canonical_hash(per), task_history=history,
                   memory_injection_log=logs, request_metrics=requests)
        return row,goal,context,per

    def test_first_query_whitespace_preserves_raw_prompt_and_later_query(self):
        for mode in (False, True):
            args = self.fixture(mode, ' \nOriginal task \t')
            self.assertEqual(verify_trial(*args, lambda x:x), 2 if mode else 1)
            broken = copy.deepcopy(args)
            broken[0]['memory_injection_log'][0]['query_sha256'] = canonical_hash(args[1])
            with self.assertRaisesRegex(ValueError, 'query drift'):
                verify_trial(*broken, lambda x:x)
            broken = copy.deepcopy(args)
            text = args[3]['turns'][-1]['text'] if mode else args[2]['text']
            broken[0]['task_history'][0]['content'] = injected(args[1].strip(), text)
            with self.assertRaisesRegex(ValueError, 'final memory block drift'):
                verify_trial(*broken, lambda x:x)
            if mode:
                broken = copy.deepcopy(args)
                broken[0]['memory_injection_log'][1]['query_sha256'] = canonical_hash((args[1]+'\nNext').strip())
                with self.assertRaisesRegex(ValueError, 'query drift'):
                    verify_trial(*broken, lambda x:x)

    def test_rebuild_both_modes_and_reject_drift(self):
        for mode in (False,True):
            args = self.fixture(mode)
            self.assertEqual(verify_trial(*args, lambda x:x), 2 if mode else 1)
            for field in ('trial_seed','memory_context_sha256'):
                broken = copy.deepcopy(args)
                broken[0][field] = 'tampered'
                with self.assertRaises(ValueError):
                    verify_trial(*broken, lambda x:x)
            broken=copy.deepcopy(args)
            broken[0]['request_metrics'][0]['prompt_sha256']='tampered'
            with self.assertRaisesRegex(ValueError,'prompt drift'):
                verify_trial(*broken,lambda x:x)

    def test_early_stop_is_retained_and_uninjected_future_is_rejected(self):
        row,goal,context,per = self.fixture(True)
        row['task_history']=row['task_history'][:2]
        row['task_history'][0]['content']=injected(goal,context['text'])
        row['memory_injection_log']=row['memory_injection_log'][:1]
        row['request_metrics']=row['request_metrics'][:1]
        self.assertEqual(verify_trial(row,goal,context,per,lambda x:x),1)
        row['task_history'] += [{'role':'user','content':'Next'}, {'role':'assistant','content':'bad'}]
        row['request_metrics'] += row['request_metrics'][:1]
        with self.assertRaisesRegex(ValueError,'uninjected'):
            verify_trial(row,goal,context,per,lambda x:x)
