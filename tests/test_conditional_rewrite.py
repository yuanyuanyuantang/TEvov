"""Finite source expressions, independent fallback and fixed retrieval identities."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import rewrite_procedural_memory as rw
from scripts.attach_checked_sources import attach


def evidence():
    task = {"id": "t", "tools": [{"function": {"name": "read", "description": "Read current fuel.",
        "parameters": {"properties": {"option": {"type": "string", "enum": ["fuel"]}}},
        "response": {"properties": {"status": {"properties": {"fuel": {"type": "number", "description": "Fuel in gallons."}}}}}}},
        {"function": {"name": "fill", "description": "Tank holds up to 50 gallons.",
        "parameters": {"properties": {"amount": {"type": "number", "description": "Additional fuel in gallons."}}},
        "response": {"properties": {"level": {"type": "number"}}}}}]}
    trials = []
    for index, value in enumerate([10.0, 15.0]):
        trial = {"task_id": "t", "episode_id": f"e{index}", "base_seed": index, "reward": 0,
            "task_history": [
                {"role": "user", "content": "Fill the tank to full."},
                {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "read", "arguments": '{"option":"fuel"}'}}]},
                {"role": "tool", "tool_call_id": "a", "content": json.dumps({"fuel": value})},
                {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "fill", "arguments": json.dumps({"amount": 50 - value})}}]},
                {"role": "tool", "tool_call_id": "b", "content": '{"level":50}'}]}
        trial["replay_payload"] = {"tool_calls": rw.fresh.trace_calls(trial)}
        trials.append(trial)
    metadata = {"evidence_refs": [{"episode_id": "e0", "call_index": 1,
        "call_sha256": rw.fresh.canonical_hash(trials[0]["replay_payload"]["tool_calls"][1])}],
        "parameter_sources_by_episode": {}, "candidate_assessments": [{"task_id": "t"}]}
    record = {"when_to_use": "Filling the current tank", "content": "Read the fuel before filling.",
        "type_tag": "generic_rule", "tool_tags": ["fill"], "source_task": "t", "metadata": metadata}
    record["record_id"] = "mem_" + rw.fresh.canonical_hash([record["when_to_use"], record["content"], record["tool_tags"]])[:16]
    return {"t": task}, trials, record



def proposal(record, options):
    option = next(iter(options.values()))
    source = option["supports"][0]["source"]
    return {"record_id": record["record_id"], "option_ids": [option["option_id"]],
        "replace_span": record["content"], "condition": "the current goal is to fill the tank to full",
        "evidence_refs": [{"episode_id": source["episode_id"], "call_index": source[k]} for k in ("source_call", "target_call")]}


def checked(candidate, record, options, lookup, tasks, refs=None):
    episode_tasks = {episode: "t" for episode, _ in lookup}
    return rw.check_rewrite(candidate, record, options, lookup, tasks, episode_tasks,
        set(lookup) if refs is None else refs)


def copy_evidence():
    tasks, trials, record = evidence()
    for trial in trials:
        calls = trial["replay_payload"]["tool_calls"]
        calls[1]["args"]["amount"] = rw.fresh.parse_output(calls[0]["output"])["fuel"]
    return tasks, trials, record


def test_automatic_attachment_keeps_procedure_and_abstains_on_competing_sources():
    tasks, trials, record = evidence()
    options, lookup = rw.compile_options(tasks, trials)
    before = deepcopy((record, options, lookup))
    new, decisions = attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})
    assert decisions[0]['status'] == 'attached'
    assert new[0]['content'].startswith(record['content'])
    assert 'requested increment' in new[0]['content'] and '50.0 minus' in new[0]['content']
    assert all(new[0]['metadata']['parameter_sources_by_episode'].values())
    assert (record, options, lookup) == before
    competitor = deepcopy(next(iter(options.values())))
    competitor['option_id'] = 'competing'
    options['competing'] = competitor
    unchanged, decisions = attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})
    assert decisions[0]['status'] == 'ambiguous_or_too_many_relations'
    assert unchanged == [record]


def test_automatic_attachment_rejects_post_operation_and_producer_only_anchors():
    tasks, trials, record = evidence()
    options, lookup = rw.compile_options(tasks, trials)
    record['when_to_use'] = 'After filling, verify the result.'
    assert attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})[1][0]['status'] == 'fallback'
    record['metadata']['evidence_refs'][0]['call_index'] = 0
    assert attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})[1][0]['status'] == 'no_in_scope_target_edge'


def test_automatic_copy_requires_named_id_and_available_matching_response():
    tasks, trials, record = copy_evidence()
    record['content'] = 'Use the current booking ID for the intended operation.'
    for task in tasks.values():
        task['tools'][0]['function']['response']['properties']['status']['properties'] = {
            'booking_id': {'type': 'string'}}
        task['tools'][1]['function']['parameters']['properties'] = {'booking_id': {'type': 'string'}}
    for t in trials:
        a, b = t['replay_payload']['tool_calls']
        a['output'] = '{"booking_id":"new_id"}'
        b['args'] = {'booking_id': 'new_id'}
    options, lookup = rw.compile_options(tasks, trials)
    new, decisions = attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})
    assert decisions[0]['status'] == 'attached'
    assert 'matching current response' in new[0]['content'] and 'new_id' not in new[0]['content']
    unrelated = deepcopy(record)
    unrelated.update(when_to_use='After authenticating', content='Use the access token to book.')
    assert attach([unrelated], options, lookup, tasks, {'e0': 't', 'e1': 't'})[0] == [unrelated]
    for t in trials:
        t['replay_payload']['tool_calls'][0]['response_message'] = 10
    options, lookup = rw.compile_options(tasks, trials)
    assert attach([record], options, lookup, tasks, {'e0': 't', 'e1': 't'})[0] == [record]


def cities(equal_returns=False):
    task = {"tools": [{"function": {"name": "lookup", "parameters": {"properties": {"city": {"type": "string"}}},
        "response": {"properties": {"code": {"type": "string", "description": "City code."}}}}},
        {"function": {"name": "distance", "parameters": {"properties": {k: {"type": "string", "description": "City code."} for k in ["cityA", "cityB"]}}}}]}
    trial = {"task_id": "t", "episode_id": "city_episode", "base_seed": 0, "reward": 1,
        "task_history": [{"role": "user", "content": "Find distance between Northcity and Southcity."}]}
    for i,(tool,args,output) in enumerate([
            ("lookup", {"city": "Northcity"}, {"code": "11111"}),
            ("lookup", {"city": "Southcity"}, {"code": "11111" if equal_returns else "22222"}),
            ("distance", {"cityA": "11111", "cityB": "11111" if equal_returns else "22222"}, {"distance": 42})]):
        trial["task_history"].extend([{"role": "assistant", "tool_calls": [{"id": str(i), "function": {"name": tool,"arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": str(i), "content": json.dumps(output)}])
    trial["replay_payload"] = {"tool_calls": rw.fresh.trace_calls(trial)}
    return {"t": task}, [trial]


def test_difference_keeps_multistate_units_and_adjacent_response_checks():
    tasks, trials, record = evidence()
    options, lookup = rw.compile_options(tasks, trials)
    assert len(options) == 1
    option = next(iter(options.values()))
    assert option["kind"] == "state_difference" and {s["state_value"] for s in option["supports"]} == {10, 15}
    assert option["source_path"] == ["fuel"]
    assert not rw.compile_options(tasks, trials[:1])[0]
    bad = deepcopy(tasks)
    bad["t"]["tools"][0]["function"]["response"]["properties"]["status"]["properties"]["fuel"]["description"] = "Fuel in liters."
    assert not rw.compile_options(bad, trials)[0]
    for trial in trials:trial["replay_payload"]["tool_calls"][0]["response_message"] = 4
    assert not rw.compile_options(tasks, trials)[0]


def test_documented_response_suffix_is_not_allowed_to_guess_between_fields():
    schema = {"response": {"properties": {k: {"properties": {"x": {"type":"number"}}} for k in ["a", "b"]}}}
    assert rw.response_property(schema,["x"]) == {}
    assert rw.response_property(schema,["a","x"])["type"] == "number"


def test_same_tool_two_objects_match_per_parameter_and_preserve_private_arguments():
    tasks, trials = cities()
    options, lookup = rw.compile_options(tasks,trials)
    assert {o["parameter"] for o in options.values()} == {"cityA","cityB"}
    supports = {o["parameter"]:o["supports"][0] for o in options.values()}
    assert supports["cityA"]["producer_arguments"] == {"city":"Northcity"}
    assert supports["cityB"]["producer_arguments"] == {"city":"Southcity"}
    assert {s["source"]["source_call"] for s in supports.values()} == {0,1}
    text = rw.render_instruction(list(options.values()),"the task needs codes for the two current cities")
    assert "Northcity" not in text and "11111" not in text and "22222" not in text
    assert "not response order" in text


def test_equal_returns_from_distinct_objects_are_ambiguous():
    tasks,trials = cities(equal_returns=True)
    assert not rw.compile_options(tasks,trials)[0]


@pytest.mark.parametrize("condition", [
    "After placing an order, retrieve its details.",
    "When cancelling an order, verify its details.",
    "Retrieve order details after placing an order.",
])
def test_render_preserves_context_without_inventing_an_if_clause(condition):
    for tasks, trials in (copy_evidence()[:2], evidence()[:2], cities()):
        options, _ = rw.compile_options(tasks, trials)
        text = rw.render_instruction(list(options.values()), condition)
        context, guidance = text.split("\nParameter guidance: ")
        assert context == "Context: " + condition
        assert guidance and "If After" not in text and "If Retrieve" not in text


@pytest.mark.parametrize("mode",["bool","enum","wrong_type"])
def test_bool_enum_and_incompatible_copy_are_not_relations(mode):
    tasks,trials,_ = copy_evidence()
    for trial in trials:
        a,b = trial["replay_payload"]["tool_calls"]
        if mode == "bool":a["output"]='{"fuel":false}';b["args"]["amount"]=False
    src=tasks["t"]["tools"][0]["function"]["response"]["properties"]["status"]["properties"]["fuel"]
    dest=tasks["t"]["tools"][1]["function"]["parameters"]["properties"]["amount"]
    if mode=="enum":dest["enum"]=[10.0,15.0]
    if mode=="wrong_type":src["type"]="string"
    assert not rw.compile_options(tasks,trials)[0]


@pytest.mark.parametrize("mode",["error","unknown","same_batch"])
def test_error_unknown_and_unavailable_response_do_not_support_binding(mode):
    tasks,trials,_ = copy_evidence()
    for trial in trials:
        a,b=trial["replay_payload"]["tool_calls"]
        if mode=="error":b["output"]='{"error":"failed"}'
        if mode=="unknown":b["output"]='null'
        if mode=="same_batch":a["response_message"]=b["assistant_message"]+1
    assert not rw.compile_options(tasks,trials)[0]


def test_real_read_anchor_can_join_fill_but_postcheck_and_name_only_cannot():
    tasks,trials,record = evidence();options,lookup=rw.compile_options(tasks,trials)
    record["metadata"]["evidence_refs"]=[{"episode_id":"e0","call_index":0}]
    assert rw.related_options(record,"t",options,lookup,rw.schemas_for(tasks["t"]))
    record["when_to_use"]="After filling, verify the tank."
    assert not rw.related_options(record,"t",options,lookup,rw.schemas_for(tasks["t"]))
    record["when_to_use"]="Before filling, read the tank."
    # An existing record that cites another episode's call is not linked merely by tool name.
    record["metadata"]["evidence_refs"]=[{"episode_id":"unknown","call_index":0}]
    assert not rw.related_options(record,"t",options,lookup,rw.schemas_for(tasks["t"]))


def test_payload_includes_target_failures_even_when_original_only_cites_reader():
    tasks,trials,record=evidence();options,lookup=rw.compile_options(tasks,trials)
    record["metadata"]["evidence_refs"]=[{"episode_id":"e0","call_index":0}]
    trial=deepcopy(trials[0]);trial["episode_id"]="bad"
    for c in trial["replay_payload"]["tool_calls"]:c["episode_id"]="bad"
    trial["replay_payload"]["tool_calls"][1]["output"]='{"error":"capacity exceeded"}'
    trials.append(trial)
    payload,_,_=rw.make_payload([record],"t",tasks["t"],trials,options,lookup)
    assert any(c["local_outcome"]=="explicit_error" for v in payload["current_task_trajectories"] for c in v["calls"])


def test_exact_sentence_splice_retains_other_steps_and_full_original_metadata():
    tasks,trials,record=evidence();options,lookup=rw.compile_options(tasks,trials)
    record["content"]+=" Keep other prerequisites."
    candidate=proposal(record,options);candidate["replace_span"]="Read the fuel before filling."
    original=deepcopy(record)
    new=checked(candidate,record,options,lookup,tasks)
    assert record==original and new["content"].endswith(" Keep other prerequisites.")
    assert "50.0 minus its fuel" in new["content"] and "10.0" not in new["content"]
    assert new["metadata"]["source_record_metadata"]==original["metadata"]
    assert new["metadata"]["publication_status"]=="pending_training_review"
    assert len(new["metadata"]["new_binding_evidence"])==4
    assert new["when_to_use"]==record["when_to_use"] and new["tool_tags"]==record["tool_tags"]


@pytest.mark.parametrize("mutation",["tool_name","partial_token","two_sentences","history_value","missing_producer_ref","unsupplied_ref","wrong_condition","fake_contract","late_state"])
def test_bad_splices_and_incomplete_evidence_fail_closed(mutation):
    tasks,trials,record=evidence();options,lookup=rw.compile_options(tasks,trials)
    candidate=proposal(record,options);refs=None
    if mutation=="tool_name":candidate["replace_span"]="Read"
    if mutation=="partial_token":record["content"]="Read.tool is used to inspect fuel.";candidate["replace_span"]="Read."
    if mutation=="two_sentences":record["content"]+=" Keep this.";candidate["replace_span"]=record["content"]
    if mutation=="history_value":candidate["condition"]="the current tank should be full at 10.0 gallons"
    if mutation=="missing_producer_ref":candidate["evidence_refs"]=candidate["evidence_refs"][1:]
    if mutation=="unsupplied_ref":refs={("e0",1)}
    if mutation=="wrong_condition":candidate["condition"]="the user wants extra fuel"
    if mutation=="fake_contract":next(iter(options.values()))["contract_quote"]="Capacity is 60 gallons"
    if mutation=="late_state":lookup[("e0",0)]["response_message"]=4
    with pytest.raises((ValueError,KeyError)):
        checked(candidate,record,options,lookup,tasks,refs)


def test_conditional_copy_does_not_require_the_producer_unconditionally():
    tasks,trials,record=copy_evidence();options,lookup=rw.compile_options(tasks,trials)
    candidate=proposal(record,options);candidate["condition"]="a current value needs to be obtained through read"
    new=checked(candidate,record,options,lookup,tasks)
    assert 'if read(option="fuel") is needed' in new["content"] and "{{" not in new["content"]
    assert "'read (" not in new["content"]


def test_ambiguous_response_cannot_be_hidden_by_a_stale_option():
    tasks,trials,record=copy_evidence();options,lookup=rw.compile_options(tasks,trials)
    extra=deepcopy(lookup[("e0",0)]);extra["call_index"]=-1
    lookup[("e0",-1)]=extra
    with pytest.raises(ValueError,match="ambiguous"):
        checked(proposal(record,options),record,options,lookup,tasks)


def test_attachment_keeps_all_steps_and_program_owns_complete_evidence():
    tasks, trials, record = evidence()
    record["content"] += " Verify the final state after filling."
    options, lookup = rw.compile_options(tasks, trials)
    selection = {"record_id": record["record_id"], "option_ids": list(options)}
    new = rw.check_attachment(selection, record, options, lookup, tasks, {"e0": "t", "e1": "t"})
    assert new["content"].startswith(record["content"] + "\n")
    assert "Only to fill to capacity" in new["content"] and "50.0 minus fuel" in new["content"]
    assert "requested increment" in new["content"] and len(new["metadata"]["new_binding_evidence"]) == 4
    assert new["metadata"]["source_record_metadata"] == record["metadata"]
    assert new["metadata"]["publication_status"] == "pending_training_review"
    with pytest.raises(ValueError, match="selection fields"):
        rw.check_attachment({**selection, "condition": "invented"}, record, options, lookup, tasks, {"e0": "t", "e1": "t"})


def test_attachment_input_priority_object_match_and_no_new_topic():
    tasks, trials, record = copy_evidence()
    options, lookup = rw.compile_options(tasks, trials)
    selection = {"record_id": record["record_id"], "option_ids": list(options)}
    new = rw.check_attachment(selection, record, options, lookup, tasks, {"e0": "t", "e1": "t"})
    assert "valid value already given" in new["content"] and "matching current response" in new["content"]
    assert "wait if pending" in new["content"] and "Do not create a new object" in new["content"]
    old_failure = deepcopy(lookup[("e0", 1)])
    old_failure.update(episode_id="unrelated_failure", args={"amount": "amount"})
    lookup[("unrelated_failure", 0)] = old_failure
    # A schema field in deterministic code is not a copied historical value.
    assert rw.check_attachment(selection, record, options, lookup, tasks, {"e0": "t", "e1": "t"})["content"] == new["content"]
    record["tool_tags"] = ["read"]
    with pytest.raises(ValueError, match="outside the original memory"):
        rw.check_attachment(selection, record, options, lookup, tasks, {"e0": "t", "e1": "t"})


def test_two_object_attachment_keeps_given_values_and_current_object_correspondence():
    tasks, trials = cities()
    options, _ = rw.compile_options(tasks, trials)
    text = rw.render_attachment(list(options.values()))
    assert "valid values already supplied" in text and "For missing values" in text
    assert "not response order" in text and "11111" not in text and "Northcity" not in text


def test_bank_vectors_order_permissions_and_exclusive_outputs(tmp_path):
    tasks,trials,record=evidence();options,lookup=rw.compile_options(tasks,trials)
    new=checked(proposal(record,options),record,options,lookup,tasks)
    other=deepcopy(record);other["record_id"]="z_unchanged"
    index={other["record_id"]:{"search_text":other["when_to_use"],"vector":[.1,.2]},new["record_id"]:{"search_text":new["when_to_use"],"vector":[.3,.4]}}
    hashes=rw.write_bank(tmp_path/"bank",[other,new],index)
    stored=json.loads((tmp_path/"bank/memories_index.json").read_text())
    assert stored==index and list(stored)==list(index)
    assert (tmp_path/"bank").stat().st_mode&0o777==0o700
    assert all((tmp_path/"bank"/n).stat().st_mode&0o777==0o600 for n in hashes)
    with pytest.raises(FileExistsError):rw.write_bank(tmp_path/"bank",[other,new],index)


def test_cpu_preparation_and_mock_generation_require_training_review(tmp_path,monkeypatch):
    tasks,trials,record=evidence();options,_=rw.compile_options(tasks,trials)
    source=tmp_path/"source";source.mkdir();data=tmp_path/"trials";data.mkdir()
    monkeypatch.setattr(rw.fresh,"TRAIN",tmp_path/"data/train.jsonl");monkeypatch.setattr(rw.fresh,"VAL",tmp_path/"data/val.jsonl")
    rw.fresh.TRAIN.parent.mkdir();rw.fresh.TRAIN.write_text('train');rw.fresh.VAL.write_text('val')
    paths=[]
    for trial in trials:
        p=data/(trial['episode_id']+'.json');p.write_text(json.dumps(trial));paths.append(p)
    monkeypatch.setattr(rw.fresh,"load_build_inputs",lambda directory:(tasks,trials,paths))
    index={record['record_id']:{'search_text':record['when_to_use'],'vector':[.2,.4]}}
    hashes={'memories.json':rw.write_private_json(source/'memories.json',[record]),'memories_index.json':rw.write_private_json(source/'memories_index.json',index)}
    pins={str(p.resolve()):rw.fresh.sha256_file(p) for p in [rw.fresh.TRAIN,rw.fresh.VAL,*paths]}
    mh=rw.write_private_json(source/'manifest.json',{'outputs_sha256':hashes,'inputs_sha256':pins,'manual_train_review_exclusions':[]})
    vh=rw.write_private_json(source/'verification.json',{'passed':True})
    rw.write_private_json(source/'freeze.json',{'manifest_sha256':mh,'verification_sha256':vh,'outputs_sha256':hashes})
    def forbidden(*args,**kwargs):raise AssertionError('CPU preparation must not make model calls')
    monkeypatch.setattr(rw.fresh,'DeepSeekLLMClient',forbidden)
    audit=rw.offline_rewrite(source,data,tmp_path/'cpu')
    assert audit['related_records']==1 and not audit['actor_admission'] and audit['network_requests']==0
    class Client:
        def __init__(self,settings):assert not settings.extra_body['chat_template_kwargs']['enable_thinking']
        _request_tokenization=forbidden
        _request_with_retry=forbidden
        def chat(self,prompt,user,**kwargs):
            assert 'copy sentences' in prompt and 'exactly record_id and option_ids' in prompt
            candidate = {"record_id": record["record_id"], "option_ids": list(options)}
            return {'raw':{'usage':{'total_tokens':100},'choices':[{'finish_reason':'stop'}]},'content':json.dumps({'memories':[candidate]})}
    monkeypatch.setattr(rw.fresh,'DeepSeekLLMClient',Client)
    def providers(model):
        assert model['enable_thinking'] is False and model['temperature']==0 and model['max_tokens']==4096
        assert model['base_urls']==['http://localhost:1/v1']
        return {'fixture':True}
    monkeypatch.setattr(rw.fresh,'inspect_providers',providers)
    out=tmp_path/'generated'
    result=rw.rewrite(source,data,out,{'name':'Qwen/Qwen3-8B','base_urls':['http://localhost:1/v1']})
    assert result['rewritten_records']==1 and not result['actor_admission']
    assert result['publication_status']=='pending_training_review'
    assert json.loads((out/'baseline_bank/memories.json').read_text())==[record]
    assert list(json.loads((out/'proposed_bank/memories_index.json').read_text()).values())==list(index.values())
    assert json.loads((out/'training_review.json').read_text())['candidates'][0]['decision']=='pending'
    actual=json.loads((out/'generation_settings.json').read_text())
    assert actual['temperature']==0 and actual['enable_thinking'] is False
    assert all(rw.fresh.sha256_file(Path(p))==h for p,h in result['inputs_sha256'].items())
    monkeypatch.setattr(rw.fresh, 'DeepSeekLLMClient', forbidden)
    monkeypatch.setattr(rw.fresh, 'inspect_providers', forbidden)
    # Legacy text/citations are private migration input, not repaired evidence.
    response_path = out / 'responses/t.json'
    response = json.loads(response_path.read_text())
    response['content'] = json.dumps({'memories': [{**proposal(record, options),
        'condition': 'invented historical text', 'evidence_refs': [{'episode_id': 'not-real', 'call_index': 999}]}]})
    response_path.write_text(json.dumps(response))
    migrated = rw.offline_rewrite(source, data, tmp_path/'migrated', out)
    assert migrated['resources']['requests'] == 0 and not migrated['actor_admission']
    m = json.loads((tmp_path/'migrated/proposed_bank/memories.json').read_text())[0]
    assert 'invented historical text' not in m['content']
    assert all(ref['episode_id'] in {'e0', 'e1'} for ref in m['metadata']['new_binding_evidence'])
    assert m['metadata']['legacy_proposal_private_not_injected']['condition'] == 'invented historical text'
    assert json.loads((tmp_path/'migrated/selection_migration.json').read_text())['new_prompt_not_sent']
