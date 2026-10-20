"""Bounded train-only parameter relations; no actor, retrieval, or bank writes.

The three environment adapters supply a fixed local prefix and an independent
oracle. Proposers see recorded responses/contracts, not the oracle. References
name a current operation and JSON field path, never a historical call index.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_output(value):
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None


def errored(value):
    return value is None or isinstance(value, dict) and bool(value.get("error"))


def leaves(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, path + (index,))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if not isinstance(value, float) or math.isfinite(value):
            yield path, value


def at_path(value, path):
    for part in path:
        value = value[part]
    return value


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)
    return type(a) is type(b) and a == b


def scope(tool, goal):
    """Frozen lexical scope, deliberately narrower than general intent parsing."""
    lower = goal.lower()
    if tool == "fillFuelTank":
        full = re.search(r"(?:tank.*\bfull\b|completely full|add it to full|fill (?:up |out )?(?:my |the |vehicle.?s )?(?:fuel )?tank)", lower)
        explicit_quantity = re.search(r"\d+(?:\.\d+)?\s*(?:gallons?|liters?)", lower)
        return ("VehicleControlAPI", "fuelAmount", "fill_to_full") if full and not explicit_quantity else None
    if tool == "place_order" and re.search(r"(?:prevailing|going|present|current).*market (?:rate|price)", lower):
        return "TradingBot", "price", "current_market_price"
    if tool in {"cancel_order", "get_order_details"}:
        return "TradingBot", "order_id", "current_created_order"
    if tool in {"cancel_booking", "retrieve_invoice"} and "insurance" not in lower:
        return "TravelAPI", "booking_id", "current_created_booking"
    return None


def build_units(tasks, trials):
    """Identity/scope first; outcomes only determine available fitting evidence.

    Each episode contributes its last executed non-error target observation.
    Units with only failed targets or missing reads remain in the denominator.
    Existing-object tasks without any creation in recorded prefixes are outside
    this declared new-object scope, counted separately before generation.
    """
    groups, excluded = {}, Counter()
    for filename, trial in trials:
        calls = trial["replay_payload"]["tool_calls"]
        for index, call in enumerate(calls):
            goal = call.get("goal_snapshot") or call.get("state", {}).get("user_goal", "")
            selected = scope(call["tool_id"], goal)
            if not selected:
                continue
            domain, parameter, condition = selected
            if parameter not in call["args"]:
                continue
            key = (trial["task_id"], call.get("turn_id"), call["tool_id"], parameter, goal)
            group = groups.setdefault(key, {"domain": domain, "condition": condition, "observations": [], "all_refs": []})
            group["all_refs"].append({"file": filename, "episode_id": trial["episode_id"], "base_seed": trial["base_seed"], "call_index": index, "call_sha256": digest(call)})
            group["observations"].append({"episode_id": trial["episode_id"], "call_index": index, "target": call, "prior": calls[:index]})
    units = []
    for key, group in sorted(groups.items(), key=lambda pair: canonical(pair[0])):
        task_id, turn, target_tool, parameter, goal = key
        creator = "book_flight" if group["domain"] == "TravelAPI" else "place_order"
        if parameter.endswith("_id") and not any(c["tool_id"] == creator for o in group["observations"] for c in o["prior"]):
            excluded["existing_object_without_recorded_creation"] += 1
            continue
        raw_observations = group.pop("observations")
        by_episode = {}
        for observation in raw_observations:
            if not errored(parse_output(observation["target"].get("output"))) and observation["target"].get("execution_status", "executed") == "executed":
                by_episode[observation["episode_id"]] = observation
        observations = list(by_episode.values())
        schemas = {t["function"]["name"]: t["function"] for t in tasks[task_id]["tools"]}
        constants = []
        if parameter == "fuelAmount":
            match = re.search(r"hold up to (\d+(?:\.\d+)?) gallons", schemas[target_tool]["description"])
            if match:
                constants = [{"value": float(match[1]), "quote": match[0], "contract_sha256": digest(schemas[target_tool])}]
        # Only the public contract can introduce a constant; observations cannot.
        unit = {**group, "unit_id": digest(key)[:20], "task_id": task_id, "turn_id": turn,
                "goal": goal, "target_tool": target_tool, "parameter": parameter,
                "observations": observations, "context_observation": observations[0] if observations else raw_observations[0], "constants": constants,
                "initial_config": copy.deepcopy(tasks[task_id]["extra"]["initial_config"]),
                "schemas": schemas}
        unit["prefix"], unit["target_args"] = fixed_prefix(unit)
        unit["variations"] = {batch: [variation(unit, batch, i) for i in range(3)] for batch in ("A", "B")}
        units.append(unit)
    if len(units) > 166:
        raise ValueError("scope exceeds frozen 166-unit cap")
    return units, dict(excluded)


def fixed_prefix(unit):
    first = unit["context_observation"]
    args = copy.deepcopy(first["target"]["args"])
    domain = unit["domain"]
    if domain == "VehicleControlAPI":
        return [{"tool": "displayCarStatus", "args": {"option": "fuel"}}], args
    if unit["parameter"] == "price":
        return [{"tool": "get_stock_info", "args": {"symbol": args["symbol"]}}], args
    creator = "book_flight" if domain == "TravelAPI" else "place_order"
    creators = [c for c in first["prior"] if c["tool_id"] == creator and not errored(parse_output(c.get("output")))]
    if len(creators) != 1:
        return [], args
    prefix = [{"tool": creator, "args": copy.deepcopy(creators[0]["args"])}]
    reader = "retrieve_invoice" if domain == "TravelAPI" else "get_order_details"
    if unit["target_tool"] != reader and any(c["tool_id"] == reader for c in first["prior"]):
        prefix.append({"tool": reader, "args": {**({"access_token": prefix[0]["args"]["access_token"]} if domain == "TravelAPI" else {}), unit["parameter"]: "$created_object"}})
    return prefix, args


def variation(unit, batch, i):
    """A/B fixed before candidate generation; no oracle/candidate-dependent values."""
    config = {unit["domain"]: copy.deepcopy(unit["initial_config"].get(unit["domain"], {}))}
    state = config[unit["domain"]]
    if unit["domain"] == "VehicleControlAPI":
        state["fuelLevel"] = ({"A": [6.25, 21.75, 44.5], "B": [13.25, 27.0, 46.25]}[batch])[i]
    elif unit["domain"] == "TravelAPI":
        state["random_seed"] = 2026101301 + i + (3 if batch == "B" else 0)
    else:
        state["order_counter"] = ({"A": [31401, 31403, 31407], "B": [42001, 42003, 42007]}[batch])[i]
        if unit["parameter"] == "price":
            factor = ({"A": [1.11, 1.37, 1.71], "B": [1.23, 1.53, 1.93]}[batch])[i]
            for stock in state.get("stocks", {}).values():
                stock["price"] = round(stock["price"] * factor, 2)
    return {"batch": batch, "variation_id": f"{batch}{i}", "initial_config": config}


def prior_values(unit, observation, producer):
    calls = [c for c in observation["prior"] if c["tool_id"] == producer and not errored(parse_output(c.get("output")))]
    if producer == "displayCarStatus":
        calls = [c for c in calls if c["args"].get("option") == "fuel"]
    if producer == "get_stock_info":
        calls = [c for c in calls if c["args"].get("symbol") == observation["target"]["args"].get("symbol")]
    if len(calls) != 1:
        raise ValueError("producer absent or non-unique in recorded object scope")
    return parse_output(calls[0].get("output"))


def references(unit):
    refs = {}
    for step in unit["prefix"]:
        for observation in unit["observations"]:
            try:
                value = prior_values(unit, observation, step["tool"])
            except ValueError:
                continue
            for path, _ in leaves(value):
                ref = {"producer": step["tool"], "path": list(path)}
                refs[canonical(ref)] = ref
    return [refs[key] for key in sorted(refs)]


def candidate_valid(unit, candidate):
    if not isinstance(candidate, dict) or candidate.get("kind") not in {"copy", "difference"}:
        return False
    allowed = {"kind", "producer", "path"} | ({"constant"} if candidate["kind"] == "difference" else set())
    if set(candidate) != allowed or {"producer": candidate["producer"], "path": candidate["path"]} not in references(unit):
        return False
    if candidate["kind"] == "difference" and (isinstance(candidate["constant"], bool) or not any(same(candidate["constant"], c["value"]) for c in unit["constants"])):
        return False
    return True


def instantiate(candidate, outputs):
    value = at_path(outputs[candidate["producer"]], candidate["path"])
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("non-scalar parameter")
    if candidate["kind"] == "difference":
        if not isinstance(value, (int, float)):
            raise ValueError("difference operand is not numeric")
        value = candidate["constant"] - value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite parameter")
    return value


def fits(unit, candidate):
    if not candidate_valid(unit, candidate) or not unit["observations"]:
        return False
    for observation in unit["observations"]:
        try:
            value = instantiate(candidate, {candidate["producer"]: prior_values(unit, observation, candidate["producer"])})
            if not parameter_type(unit, value) or not same(value, observation["target"]["args"][unit["parameter"]]):
                return False
        except (ValueError, KeyError, TypeError, IndexError):
            return False
    return True


def parameter_type(unit, value):
    expected = unit["schemas"][unit["target_tool"]]["parameters"]["properties"][unit["parameter"]]["type"]
    if isinstance(value, bool):
        return False
    if expected in {"number", "float"}:
        return isinstance(value, (int, float)) and math.isfinite(value)
    if expected in {"integer", "int"}:
        return isinstance(value, int)
    return expected == "string" and isinstance(value, str)


def rank(candidate):
    return (candidate["kind"] != "copy", canonical(candidate))


def pool(unit, proposer, proposed=None):
    universe = [{"kind": "copy", **ref} for ref in references(unit)]
    universe += [{"kind": "difference", **ref, "constant": c["value"]} for ref in references(unit) for c in unit["constants"]]
    if proposer == "G3":
        universe = proposed if isinstance(proposed, list) and len(proposed) <= 3 else []
    result = []
    for candidate in universe:
        if not fits(unit, candidate):
            continue
        if proposer == "G1":
            tokens = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", unit["parameter"]).lower()))
            field = str(candidate["path"][-1])
            field_tokens = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", field).lower()))
            if not tokens & field_tokens:
                continue
            # A direct copy must not have an indistinguishable alternative source.
            if candidate["kind"] == "copy":
                ambiguous = False
                for observation in unit["observations"]:
                    target = observation["target"]["args"][unit["parameter"]]
                    goal_numbers = re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", unit["goal"])
                    defaults = [v.get("default") for v in unit["schemas"][unit["target_tool"]]["parameters"]["properties"].values() if "default" in v]
                    if str(target) in goal_numbers or any(same(target, v) for v in defaults):
                        ambiguous = True
                    for prior in observation["prior"]:
                        same_role = prior["tool_id"] == candidate["producer"]
                        if candidate["producer"] == "get_stock_info":
                            same_role &= prior["args"].get("symbol") == observation["target"]["args"].get("symbol")
                        if candidate["producer"] == "displayCarStatus":
                            same_role &= prior["args"].get("option") == "fuel"
                        if same_role:
                            continue
                        if any(same(target, v) for _, v in leaves(parse_output(prior.get("output")))):
                            ambiguous = True
                if ambiguous:
                    continue
        if canonical(candidate) not in {canonical(c) for c in result}:
            result.append(candidate)
    return sorted(result, key=rank)[:3]


def selection(candidates, results):
    if not candidates:
        return {"status": "no_candidate", "candidate": None}
    if any(r["status"] == "infrastructure_error" for rows in results for r in rows):
        return {"status": "infrastructure_error", "candidate": None}
    survivors = [c for c, rows in zip(candidates, results) if len(rows) == 3 and all(r["status"] == "valid" for r in rows)]
    if not survivors:
        return {"status": "refuted", "candidate": None}
    if len({canonical(c) for c in survivors}) > 1:
        return {"status": "unresolved", "candidate": None}
    return {"status": "bounded_checked", "candidate": survivors[0]}


def load_executor(root):
    name = "_offline_bfcl_session"
    spec = importlib.util.spec_from_file_location(name, root / "tevo/experiment/bfcl_session_executor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.BFCLSessionExecutor


class LocalChecks:
    """Each attempt includes initialization/prefix, even when it fails early."""
    def __init__(self, executor, budget=10000, previous=12):
        self.executor, self.budget = executor, budget
        self.attempts, self.dispatches, self.public_calls = previous, 0, 0
        self.rows = []

    def run(self, unit, variation, candidate=None):
        if self.attempts >= self.budget:
            raise RuntimeError("local attempt budget exhausted")
        self.attempts += 1
        started = time.perf_counter()
        session = self.executor("multi_turn_offline_" + unit["unit_id"], copy.deepcopy(variation["initial_config"]), [unit["domain"]])
        row = {"unit_id": unit["unit_id"], "variation": variation["variation_id"], "candidate": candidate,
               "status": "infrastructure_error", "explicit_calls": 0, "public_calls": 0}
        try:
            if not unit["prefix"]:
                raise ValueError("no unique recorded creator / fitting observation")
            outputs, instances = {}, {}
            # Count nested public API calls as well as explicit dispatches.
            session.start()
            runtime = sys.modules[session.__class__.__module__]._bfcl_multi_turn_utils
            instance = vars(runtime)[session._instance_names()[0]]
            for name in dir(instance):
                method = getattr(instance, name)
                if not name.startswith("_") and callable(method):
                    def counted(_self, *args, _method=method, **kwargs):
                        row["public_calls"] += 1
                        return _method(*args, **kwargs)
                    setattr(instance, name, MethodType(counted, instance))
            def dispatch(tool, args):
                if tool not in unit["schemas"] or not re.fullmatch(r"[A-Za-z_]\w*", tool):
                    raise ValueError("non-contract tool")
                if any(not re.fullmatch(r"[A-Za-z_]\w*", key) for key in args):
                    raise ValueError("unsafe keyword")
                call = tool + "(" + ",".join(f"{key}={value!r}" for key, value in sorted(args.items())) + ")"
                self.dispatches += 1
                row["explicit_calls"] += 1
                result, _ = session.execute(call)
                parsed = parse_output(result[0]) if result else None
                return parsed
            created = None
            balance_before = copy.deepcopy(getattr(instance, "credit_card_list", {}))
            for step in unit["prefix"]:
                args = {k: (created if v == "$created_object" else v) for k, v in step["args"].items()}
                output = dispatch(step["tool"], args)
                if errored(output):
                    raise ValueError("fixed producer prefix failed: " + str(output))
                outputs[step["tool"]] = output
                if step["tool"] == "book_flight":
                    created = output["booking_id"]
                elif step["tool"] == "place_order":
                    created = output["order_id"]
            if candidate is None:
                row["status"] = "construction_ok"
            else:
                try:
                    value = instantiate(candidate, outputs)
                    if not parameter_type(unit, value):
                        raise ValueError("candidate argument violates target parameter type")
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    row.update(status="refuted", reason="candidate_instantiation: " + str(exc))
                else:
                    args = {**unit["target_args"], unit["parameter"]: value}
                    output = dispatch(unit["target_tool"], args)
                    valid = not errored(output)
                    if unit["domain"] == "VehicleControlAPI":
                        maximum = vars(sys.modules[instance.__class__.__module__])["MAX_FUEL_LEVEL"]
                        valid &= same(instance.fuelLevel, maximum)
                    elif unit["domain"] == "TravelAPI":
                        valid &= same(value, created)
                        if unit["target_tool"] == "cancel_booking":
                            valid &= created not in instance.booking_record and instance.credit_card_list == balance_before
                        else:
                            valid &= output.get("invoice", {}).get("booking_id") == created
                    elif unit["parameter"] == "price":
                        order = instance.orders[instance.order_counter - 1]
                        valid &= same(order["price"], instance.stocks[args["symbol"]]["price"]) and order["symbol"] == args["symbol"] and same(order["amount"], args["amount"])
                    else:
                        valid &= same(value, created)
                        if unit["target_tool"] == "cancel_order":
                            valid &= instance.orders[created]["status"] == "Cancelled"
                        else:
                            valid &= same(output.get("id"), created) and output.get("symbol") == instance.orders[created]["symbol"]
                    row.update(status="valid" if valid else "refuted", argument=value, output=output)
        except Exception as exc:
            row.update(status="infrastructure_error", reason=type(exc).__name__ + ": " + str(exc))
        finally:
            session.close()
        row["seconds"] = time.perf_counter() - started
        self.public_calls += row["public_calls"]
        self.rows.append(row)
        return row


def metrics(units, choices, scores):
    published = [u for u in units if choices[u["unit_id"]]["candidate"] is not None]
    failed = sum(any(r["status"] == "refuted" for r in scores[u["unit_id"]]) for u in published)
    errors = [u["unit_id"] for u in published if len(scores[u["unit_id"]]) != 3 or any(r["status"] == "infrastructure_error" for r in scores[u["unit_id"]])]
    n, p = len(units), len(published)
    return {"N": n, "P": p, "F": failed, "C": p / n if n else 0,
            "E": failed / p if p and not errors else None,
            "Q": (p - failed) / n if n and not errors else None,
            "abstention_rate": 1 - p / n if n else 1,
            "abstention_reasons": dict(Counter(choices[u["unit_id"]]["status"] for u in units if choices[u["unit_id"]]["candidate"] is None)),
            "published_domains": sorted({u["domain"] for u in published}), "scoring_incomplete": errors}


def decide(summary):
    eligible = [g for g in ("G1", "G2", "G3") if summary[g + "+V"]["C"] >= .30 and summary[g + "+V"]["E"] is not None and len(summary[g + "+V"]["published_domains"]) >= 3]
    simple = [g for g in eligible if g != "G3"]
    selected = None
    for g in simple:
        if selected is None:
            selected = g
            continue
        a, b = summary[g + "+V"], summary[selected + "+V"]
        if b["E"] - a["E"] >= .02 or abs(a["E"] - b["E"]) < .02 and a["Q"] - b["Q"] >= .02:
            selected = g
        # Inseparable shared accounting makes wall time non-comparable: prefer G1.
    if "G3" in eligible:
        if selected is None:
            selected = "G3"
        else:
            a, b = summary["G3+V"], summary[selected + "+V"]
            if b["C"] - a["C"] <= .10 and (b["E"] - a["E"] >= .05 or a["Q"] - b["Q"] >= .10 and a["E"] - b["E"] <= .02):
                selected = "G3"
    gate = False
    reason = "no verified proposer meets coverage / three-domain floor"
    if selected:
        v, v0 = summary[selected + "+V"], summary[selected + "+V0"]
        if v0["E"] is None:
            reason = "selected V0 has undefined/incomplete error rate"
        elif v0["E"] < .05:
            reason = "selected V0 error below 5%; validation necessity not demonstrated"
        else:
            gate = v0["E"] - v["E"] >= .05 and v0["C"] - v["C"] <= .10
            reason = "validation gate passed" if gate else "selected proposer's own validation gate failed"
    return {"selected_proposer": selected, "validation_gate": gate, "reason": reason,
            "formal_actor_evaluation_started": False, "literature_increment_established": False,
            "next": "stop method upgrade" if not gate else "review concrete novelty difference before any formal evaluation"}
