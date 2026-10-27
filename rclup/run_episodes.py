"""Run closed-loop HoosierHelp episodes under the fan-out topology (paper App. G) and save records.

Examples:
  python -m rclup.run_episodes --controller voi_det --limit 6 --out smoke/voi_det
  python -m rclup.run_episodes --controller voi_det --variant atomic --intake-policy atomic --max-turns 8 \
      --behaviors normal,rambling,impatience,self_contradictory,unsupported_request --limit 240 --seed 0 --out eval/seed0/voi_det

Runs resume: episodes already present in <out>/records.jsonl are skipped (see scripts/strip_errors.py).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
import traceback
from pathlib import Path

from .backbone import VLLMClient
from .common import DATA, RUNS, append_jsonl, read_jsonl, write_json
from .cost import BudgetExceeded, CostLedger
from .workflow import (ATDController, DisagreementController, FixedOrderController, GreedyController, HoosierWorkflow, LLMVoIController,
                       MaxUncertaintyController, MyopicVOIController, Pi0Controller, ProCoTController, SAGEController, VOIController)

CONTROLLERS = {
    "voi_det": lambda seed: VOIController(),            # ours: deterministic value of information (paper G.5/G.6 defaults)
    "voi_myopic": lambda seed: MyopicVOIController(),   # one-step VOI
    "voi_llm": lambda seed: LLMVoIController(),         # VOI with LLM-estimated beliefs
    "disagreement_ask": lambda seed: DisagreementController(),
    "max_uncertainty": lambda seed: MaxUncertaintyController(),
    "fixed_order": lambda seed: FixedOrderController(),
    "sage": lambda seed: SAGEController(),
    "atd": lambda seed: ATDController(),
    "procot": lambda seed: ProCoTController(),
    "text_mas": lambda seed: GreedyController(),        # no clarification: greedy hand-off
    "pi0": lambda seed: Pi0Controller(seed),            # stochastic commitment reference
}
BEHAVIORS = "normal,rambling,impatience,self_contradictory,unsupported_request"


def load_roots(split: str, limit: int | None, seed: int = 0, behaviors: tuple[str, ...] = ("normal",)) -> list[dict]:
    path = DATA / "hoosier" / ("eval_specs.jsonl" if split == "test" else "train_specs.jsonl")
    if not path.exists():
        raise SystemExit(f"{path} not found: build the user specs first (python -m rclup.data.hoosier_build)")
    rows = [r for r in read_jsonl(path) if r["split"] == split]
    rows = [dict(r, behavior=b) for r in rows for b in behaviors]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def make_env(root: dict, ledger: CostLedger, max_turns: int | None = None):
    from .envs.hoosier_env import HoosierEnv, MAX_AGENT_TURNS
    return HoosierEnv(root, root.get("behavior", "normal"), ledger, max_agent_turns=max_turns or MAX_AGENT_TURNS)


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    out = {"n": len(records), "errors": len(records) - len(ok)}
    if not ok:
        return out
    out["end_to_end_match"] = statistics.mean(r["score"]["end_to_end_match"] for r in ok)
    out["user_questions_avg"] = statistics.mean(r["score"]["user_questions"] for r in ok)
    out["items_used_avg"] = statistics.mean(r.get("items_used", 0) for r in ok)
    out["tool_all_match"] = statistics.mean(bool(r["score"]["all_match"]) for r in ok)
    out["resource_exact_match"] = statistics.mean(bool(r["score"]["resource_exact_match"]) for r in ok)
    out["explicit_final"] = statistics.mean(bool(r["score"]["explicit_final_action"]) for r in ok)
    out["agent_turns_avg"] = statistics.mean(r["score"]["agent_turns"] for r in ok)
    by = collections.defaultdict(list)
    for r in ok:
        by[f"{r['case_type']}|{r['constraint_profile']}"].append(r["score"]["end_to_end_match"])
    out["by_case"] = {k: statistics.mean(v) for k, v in by.items()}
    by = collections.defaultdict(list)
    for r in ok:
        by[r.get("behavior", "")].append(r["score"]["end_to_end_match"])
    out["by_behavior"] = {k: statistics.mean(v) for k, v in by.items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=sorted(CONTROLLERS), default="voi_det")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--limit", type=int, default=8, help="episodes (0 = all specs x behaviours)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--behaviors", default="normal", help=f"comma list of simulated-user behaviours; the paper protocol uses {BEHAVIORS}")
    p.add_argument("--items-budget", type=int, default=2)
    p.add_argument("--variant", default="atomic", choices=["atomic", "native"],
                   help="intake-question protocol: atomic = ordinary intake questions are budgeted controller decisions; "
                        "native = the benchmark's fixed multi-item intake questions (official HoosierHelp protocol)")
    p.add_argument("--intake-policy", default="atomic", choices=["atomic", "native"],
                   help="grouping of ordinary intake questions: atomic = one field per question; native = the benchmark's three groups")
    p.add_argument("--max-turns", type=int, default=None, help="HoosierHelp agent-turn cap (native 8)")
    p.add_argument("--modules-off", default="", help="component switches: comma list of modules to switch off: relaxation, det_planner, user_direct, prune, "
                   "fallback, contradiction, candidates")
    p.add_argument("--omega-max", type=int, default=None, help="joint interpretations retained (paper G.2: 16)")
    p.add_argument("--s-draws", type=int, default=None, help="stochastic proposals per invocation (paper G.2: 4 + greedy)")
    p.add_argument("--k-max", type=int, default=None, help="candidates kept per packet (paper G.2: 3)")
    p.add_argument("--p-fallback", type=float, default=None, help="VOI fallback-acceptance prior (paper G.5: 0.5)")
    p.add_argument("--voi-depth", type=int, default=None, help="VOI lookahead depth H (paper G.5: 2)")
    p.add_argument("--c-item", type=float, default=None, help="per-question cost (paper G.6: 0.04)")
    p.add_argument("--c-fallback", type=float, default=None, help="fallback-question cost (paper G.6: 0.06)")
    p.add_argument("--max-calls-episode", type=int, default=None, help="backbone-call budget per episode (paper G.7: 180; 0 = unlimited)")
    p.add_argument("--out", required=True, help="output directory (relative paths are under RCLUP_RUNS)")
    p.add_argument("--ledger", default=str(RUNS / "ledger.json"), help="shared OpenAI cost ledger")
    p.add_argument("--cap-usd", type=float, default=40.0, help="hard cap on the ledger TOTAL (all runs sharing --ledger)")
    p.add_argument("--workers", type=int, default=4, help="concurrent episodes")
    a = p.parse_args()

    out = Path(a.out) if Path(a.out).is_absolute() else RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(Path(a.ledger), a.cap_usd, label="eval")
    bb = VLLMClient()
    roots = load_roots(a.split, a.limit, a.seed, tuple(a.behaviors.split(",")))
    records_path = out / "records.jsonl"
    done = {r["root_id"] + "|" + r.get("behavior", "") for r in read_jsonl(records_path)} if records_path.exists() else set()
    start = time.monotonic()
    usd0 = ledger.usd

    def one(root):
        key = root["root_id"] + "|" + root.get("behavior", "")
        if key in done:
            return None
        env = make_env(root, ledger, a.max_turns)
        wf = HoosierWorkflow(bb, CONTROLLERS[a.controller](a.seed), seed=a.seed, items_budget=a.items_budget)
        wf.set_topology("fanout")
        wf.set_variant(a.variant)
        if a.intake_policy == "atomic":
            wf.intake_groups = tuple((f,) for f in ("location", "schedule", "intake_methods", "available_documents", "eligibility"))
        for m in [x for x in a.modules_off.split(",") if x]:
            if m == "relaxation": wf.use_relaxation = False
            elif m == "det_planner": wf.llm_planner = True
            elif m == "user_direct": wf.user_direct = False
            elif m == "prune": wf.prune_on_answer = False
            elif m == "fallback": wf.fallback_clarification = False
            elif m == "contradiction": wf.contradiction_clarification = False
            elif m == "candidates": wf.s_draws = 0
            else: raise SystemExit(f"unknown module {m}")
        if a.s_draws is not None: wf.s_draws = a.s_draws
        if a.k_max is not None: wf.k_max = a.k_max
        if a.p_fallback is not None: wf.p_fallback_override = a.p_fallback
        if a.voi_depth is not None: wf.controller.depth = a.voi_depth
        if a.c_item is not None: wf.controller.c_item = a.c_item
        if a.c_fallback is not None: wf.controller.c_fallback = a.c_fallback
        if a.omega_max is not None: wf.omega_max = a.omega_max
        if a.max_calls_episode is not None: wf.max_calls_episode = a.max_calls_episode or None
        try:
            rec = wf.run(env)
        except BudgetExceeded:
            raise
        except Exception as e:  # keep failures as records (unresolved = failure)
            rec = env.record() if env.messages else {"root_id": root["root_id"], "score": {"end_to_end_match": False, "user_questions": 0}}
            rec.update({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:], "controller": a.controller,
                        "trace": wf.trace.events})
        rec["behavior"] = root.get("behavior", "")
        rec["topology"] = "fanout"
        rec["gen_usage"] = bb.usage_dict()
        return rec

    from concurrent.futures import ThreadPoolExecutor, as_completed
    records = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(one, r) for r in roots]
        for f in as_completed(futures):
            try:
                rec = f.result()
            except BudgetExceeded as e:
                print("STOP:", e)
                break
            if rec is None:
                continue
            records.append(rec)
            append_jsonl(records_path, rec)
            s = rec["score"]
            print(f"[{len(records)}/{len(roots)}] {rec['root_id']} ok={s.get('end_to_end_match')} uq={s.get('user_questions')} "
                  f"items={rec.get('items_used')} err={'error' in rec} ${ledger.usd - usd0:.3f}", flush=True)
    all_records = read_jsonl(records_path)
    summary = summarize(all_records)
    summary.update({"controller": a.controller, "split": a.split, "items_budget": a.items_budget, "variant": a.variant,
                    "max_turns": a.max_turns, "intake_policy": a.intake_policy, "modules_off": a.modules_off, "topology": "fanout",
                    "behaviors": a.behaviors, "seed": a.seed,
                    "s_draws": a.s_draws, "k_max": a.k_max, "p_fallback": a.p_fallback, "voi_depth": a.voi_depth, "c_item": a.c_item,
                    "c_fallback": a.c_fallback, "omega_max": a.omega_max, "max_calls_episode": a.max_calls_episode,
                    "wall_seconds": time.monotonic() - start, "openai_usd_this_run": ledger.usd - usd0,
                    "openai_usd_ledger_total": ledger.usd, "gen_usage": bb.usage_dict()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
