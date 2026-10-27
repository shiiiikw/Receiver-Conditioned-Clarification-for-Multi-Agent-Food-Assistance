"""Episode driver: role invocations, candidate packets, controllers and branching (fan-out topology, paper App. G).

The workflow is the public planner shared by every method: search, final
recommendation and fallback repair are fixed rules. Methods differ only in the
receiver-side controller that chooses between committing a candidate and asking
one atomic clarification. Episodes can be forked at a receiver decision point
and resumed with a forced first decision.
"""
from __future__ import annotations

import collections
import copy
import itertools
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .backbone import VLLMClient
from .packets import Packet, Candidate, build_packet, UNRESOLVED_KEY
from . import roles as R

S_DRAWS = 8
PROPOSAL_MAX_TOKENS = 512
T_FB = 5   # turns the workflow's fallback path can need after the last ordinary question: search, relaxation question,
           # follow-up for a vague answer, re-search, final (a design constant of the workflow, not tuned on any split)
Q_MAX = 6


@dataclass
class Question:
    qid: str
    target_role: str
    scope: dict            # {"need_id","field"} or {"status"}
    a: str                 # candidate key (or "__residual__")
    b: str
    text: str
    items: int = 1
    tier: int = 0          # 0 disagreement, 1 unknown, 2 confirmation
    footprint: tuple = ()  # dependency footprint R_t(q): receivers reachable from the targeted premise (G.3)


@dataclass
class Decision:
    kind: str              # "commit" | "ask"
    candidate: int | None = None
    question: Question | None = None
    scores: dict | None = None


class Controller:
    """Receiver-side action rule. Baselines never ask."""
    name = "base"

    def decide(self, ctx: "ReceiverContext") -> Decision:
        raise NotImplementedError


class GreedyController(Controller):
    name = "text_mas"

    def decide(self, ctx):
        return Decision("commit", ctx.greedy_index())


class Pi0Controller(Controller):
    """Reference continuation pi_0: stochastic backbone commitment, no learned asks."""
    name = "pi0"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def decide(self, ctx):
        cands = [c for c in ctx.candidates if not c.is_unresolved]
        weights = [max(c.count, 0) for c in cands]
        if sum(weights) == 0:
            return Decision("commit", ctx.greedy_index())
        pick = self.rng.choices(range(len(cands)), weights=weights, k=1)[0]
        return Decision("commit", ctx.candidates.index(cands[pick]))


class DisagreementController(Controller):
    """Heuristic control: ask the first menu question while budget remains."""
    name = "disagreement_ask"

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            return Decision("ask", question=ctx.menu[0])
        return Decision("commit", ctx.greedy_index())


def _question_masses(ctx: "ReceiverContext", q: "Question") -> tuple[float, float]:
    """Support mass on the two sides of a menu question (residual side = unlisted + unknown mass)."""
    pk = ctx.incoming[0]
    wa = sum(c.weight for c in pk.candidates if c.key == q.a)
    if q.b == "__residual__":
        wb = pk.residual_weight + sum(c.weight for c in pk.candidates if not c.is_unresolved and c.key != q.a
                                      and ctx.aux.get("field_value", {}).get((c.key, q.qid)) == "unknown")
    else:
        wb = sum(c.weight for c in pk.candidates if c.key == q.b)
    return wa, wb


class MaxUncertaintyController(Controller):
    """Heuristic control: ask the question whose minority side carries the most support (largest disagreement mass)."""
    name = "max_uncertainty"

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            scored = sorted(ctx.menu, key=lambda q: -min(_question_masses(ctx, q)))
            return Decision("ask", question=scored[0], scores={"masses": [_question_masses(ctx, q) for q in ctx.menu]})
        return Decision("commit", ctx.greedy_index())


class FixedOrderController(Controller):
    """Heuristic control: ask open questions in the task's fixed field order (ignores disagreement tiers)."""
    name = "fixed_order"

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            order = ctx.aux.get("field_order", {})
            scored = sorted(ctx.menu, key=lambda q: order.get(q.qid, 10 ** 6))
            return Decision("ask", question=scored[0])
        return Decision("commit", ctx.greedy_index())


def voi_values(ctx: "ReceiverContext") -> list[float]:
    """Decision-relevance value of information per menu question (deterministic, no learning). Group questions (budgeted intake):
    train-prior gain minus the budget cost of a turn; other questions: support mass of alternatives that both differ on the
    asked field and change the downstream outcome signature (plus residual mass when the greedy value is unknown)."""
    sig = ctx.aux.get("signature", {})
    fv = ctx.aux.get("field_value", {})
    pk = ctx.incoming[0]
    greedy = pk.greedy
    gains = []
    for q in ctx.menu:
        if q.scope.get("pairs") is not None:
            cost = ctx.aux.get("p_fallback", 0.0) if (ctx.external_left - 1) < T_FB else 0.0
            gains.append(ctx.aux.get("prior", {}).get(q.qid, 0.0) - cost)
            continue
        g_sig = sig.get(greedy.key)
        g_val = fv.get((greedy.key, q.qid))
        value = 0.0
        for c in pk.candidates:
            if c.is_unresolved or c.key == greedy.key:
                continue
            if fv.get((c.key, q.qid)) != g_val and sig.get(c.key) != g_sig:
                value += c.weight
        if g_val == "unknown":
            value += pk.residual_weight
        gains.append(value)
    return gains


def voi_lookahead(ctx: "ReceiverContext", depth: int = 1, c_item: float = 0.0, c_fallback: float | None = None) -> list[float]:
    """Generalisation of voi_values: H-step lookahead (depth) and a constant per-question cost (c_item).
    depth=1, c_item=0 reproduces voi_values exactly. Deeper steps add the expected value of the best follow-up question in
    the branch where the person's answer agrees with the greedy interpretation (the only branch in which the current
    program is kept; the other branches are already credited by the one-step term, so nothing is double counted). The
    residual mass is credited at most once along a path. Group (unknown-field) questions do not split the candidate set;
    their budget cost follows the remaining turns at that depth. No model calls: the lookahead runs on the packet only."""
    sig = ctx.aux.get("signature", {})
    fv = ctx.aux.get("field_value", {})
    prior = ctx.aux.get("prior", {})
    pf = ctx.aux.get("p_fallback", 0.0)
    pk = ctx.incoming[0]
    greedy = pk.greedy
    g_sig = sig.get(greedy.key)
    cands = [c for c in pk.candidates if not c.is_unresolved and c.key != greedy.key]

    def one_step(q, cset, residual, ext_left):
        if q.scope.get("pairs") is not None:
            cost = pf if (ext_left - 1) < T_FB else 0.0
            return prior.get(q.qid, 0.0) - cost - (c_item if c_fallback is None else c_fallback), cset, residual
        g_val = fv.get((greedy.key, q.qid))
        value = sum(c.weight for c in cset if fv.get((c.key, q.qid)) != g_val and sig.get(c.key) != g_sig)
        used_residual = g_val == "unknown"
        if used_residual:
            value += residual
        # continuation branch: alternatives that answered like the greedy interpretation
        agree = [c for c in cset if fv.get((c.key, q.qid)) == g_val]
        return value - c_item, agree, (0.0 if used_residual else residual)

    def best(menu, cset, residual, ext_left, items_left, h):
        if h <= 0 or not menu or ext_left <= 2:
            return 0.0
        top = 0.0
        for q in menu:
            if q.items > items_left:
                continue
            g, nxt, res = one_step(q, cset, residual, ext_left)
            if h > 1:
                g += max(0.0, best([r for r in menu if r.qid != q.qid], nxt, res, ext_left - 1, items_left - q.items, h - 1))
            top = max(top, g)
        return top

    gains = []
    for q in ctx.menu:
        g, nxt, res = one_step(q, cands, pk.residual_weight, ctx.external_left)
        if depth > 1:
            g += max(0.0, best([r for r in ctx.menu if r.qid != q.qid], nxt, res, ctx.external_left - 1, ctx.items_left - q.items, depth - 1))
        gains.append(g)
    return gains


class VOIController(Controller):
    """Decision-relevance value of information (deterministic, no learning): a question is worth asking only if the
    interpretations it separates lead to different downstream outcomes (search results / rule verdicts). Value = support
    mass of the alternatives that both differ on the asked field and change the outcome relative to the greedy one.
    Parameters (set by run_episodes): depth H (default 1) and per-question cost c_item (default 0)."""
    name = "voi_det"
    depth = 1
    c_item = 0.0
    c_fallback = None      # cost of a fallback-acceptance (group) question (G.6); None = c_item

    def decide(self, ctx):
        gains = (voi_values(ctx) if (self.depth == 1 and self.c_item == 0.0 and self.c_fallback is None)
                 else voi_lookahead(ctx, self.depth, self.c_item, self.c_fallback))
        if ctx.menu and ctx.items_left > 0 and gains and max(gains) > 0:
            i = max(range(len(gains)), key=lambda k: gains[k])
            return Decision("ask", question=ctx.menu[i], scores={"voi": gains})
        return Decision("commit", ctx.greedy_index(), scores={"voi": gains})


class MyopicVOIController(Controller):
    """Literature-style myopic value of information (no learning, no budget model): ask the question with the largest
    probability of changing the decision while any is positive -- for an unknown field, the train prior that it is a real
    constraint; for a disagreement, the support mass of alternatives that change the outcome. Never reasons about the
    remaining turn budget."""
    name = "voi_myopic"

    def decide(self, ctx):
        gains = []
        for q in ctx.menu:
            if q.scope.get("pairs") is not None:
                gains.append(ctx.aux.get("prior", {}).get(q.qid, 0.0))
            else:
                sig, fv, pk = ctx.aux.get("signature", {}), ctx.aux.get("field_value", {}), ctx.incoming[0]
                g_sig, g_val = sig.get(pk.greedy.key), fv.get((pk.greedy.key, q.qid))
                v = sum(c.weight for c in pk.candidates if not c.is_unresolved and c.key != pk.greedy.key
                        and fv.get((c.key, q.qid)) != g_val and sig.get(c.key) != g_sig)
                gains.append(v + (pk.residual_weight if g_val == "unknown" else 0.0))
        if ctx.menu and ctx.items_left > 0 and gains and max(gains) > 0:
            i = max(range(len(gains)), key=lambda k: gains[k])
            return Decision("ask", question=ctx.menu[i], scores={"voi_myopic": gains})
        return Decision("commit", ctx.greedy_index(), scores={"voi_myopic": gains})


class LLMVoIController(Controller):
    """Value-of-information rule of Dong et al. (ACL 2026, 'Value of Information: A Framework for Human-Agent
    Communication'), transplanted to the receiver's ask-or-act decision. Following the paper: (i) the LLM states its belief
    over the hypotheses (here the sender's interpretations) as verbalised probabilities; (ii) for each closed candidate
    question the LLM predicts the person's answer distribution (for a value-A-vs-value-B question the answer is implied by
    the hypothesis; for an unknown-field question the LLM gives P(the person states a requirement)); (iii) utility is
    task-given and graded (one unit per correctly constrained search field, like the paper's flight reward); (iv) VoI(q) = E_y[V(b_y)] - V(b) and the
    question with the largest VoI - c is asked while that is positive, else the agent acts on argmax expected utility.
    The per-question cost c is a constant (no reasoning about the remaining turn budget)."""
    name = "voi_llm"

    def __init__(self, cost: float | None = None):
        self.cost = float(os.environ.get("RCLUP_VOI_LLM_COST", "0.1")) if cost is None else float(cost)

    @staticmethod
    def _belief(llm, conv: str, cands: list) -> dict:
        prior = {c.key: c.weight for c in cands}
        z = sum(prior.values()) or 1.0
        prior = {k: v / z for k, v in prior.items()}
        if llm is None or len(cands) < 2:
            return prior
        listing = "\n\n".join(f"Interpretation {i + 1}:\n{c.text}" for i, c in enumerate(cands))
        prompt = [{"role": "system", "content": "You are the planner of an Indiana 211 resource-search team. Several structured interpretations "
                   "of what the person needs were proposed. Read the conversation and state your belief, as probabilities summing to 1, that each "
                   "interpretation is the correct reading of the person's constraints. Output ONLY compact JSON: {\"probabilities\": [p1, p2, ...]}."},
                  {"role": "user", "content": "Conversation so far:\n" + conv + "\n\n" + listing}]
        obj = R.parse_json_object(llm(prompt, max_tokens=128, purpose="voi_llm_belief")) or {}
        probs = obj.get("probabilities")
        if not isinstance(probs, list) or len(probs) != len(cands):
            return prior
        try:
            vals = [max(0.0, float(x)) for x in probs]
        except (TypeError, ValueError):
            return prior
        z = sum(vals)
        if z <= 0:
            return prior
        return {c.key: v / z for c, v in zip(cands, vals)}

    @staticmethod
    def _answer_probs(llm, conv: str, qs: list) -> dict:
        if not qs:
            return {}
        if llm is None:
            return {q.qid: 0.5 for q in qs}
        listing = "\n".join(f"Q{i + 1}: {q.text}" for i, q in enumerate(qs))
        prompt = [{"role": "system", "content": "You are the planner of an Indiana 211 resource-search team. For each question below, predict the "
                   "person's answer: give the probability (0 to 1) that, if asked, the person would state a concrete requirement that changes the "
                   "search constraints (as opposed to having no requirement or confirming what is already assumed). Base it on the conversation. "
                   "Output ONLY compact JSON: {\"Q1\": p, \"Q2\": p, ...}."},
                  {"role": "user", "content": "Conversation so far:\n" + conv + "\n\nQuestions:\n" + listing}]
        obj = R.parse_json_object(llm(prompt, max_tokens=128, purpose="voi_llm_answers")) or {}
        out = {}
        for i, q in enumerate(qs):
            try:
                v = float(obj.get(f"Q{i + 1}", 0.5))
            except (TypeError, ValueError):
                v = 0.5
            out[q.qid] = min(1.0, max(0.0, v))
        return out

    def decide(self, ctx):
        pk = ctx.incoming[0]
        cands = [c for c in pk.candidates if not c.is_unresolved]
        if not ctx.menu or ctx.items_left <= 0 or not cands:
            return Decision("commit", ctx.greedy_index())
        fv = ctx.aux.get("field_value", {})
        llm, conv = ctx.aux.get("llm"), ctx.aux.get("conversation", "")
        b = self._belief(llm, conv, cands)
        open_qs = [q for q in ctx.menu if q.b == "__residual__"]
        h = self._answer_probs(llm, conv, open_qs)
        # Graded, task-given utility as in the paper's flight domain: one unit per search field that matches the person's
        # true constraint. The one-step VoI of a question is then the probability that its answer corrects that field:
        #   unknown field  -> P(person states a requirement)   (acting now assumes no constraint)
        #   A-vs-B field   -> 1 - belief mass of the majority value (acting now takes the majority reading)
        gains = []
        for q in ctx.menu:
            if q.b == "__residual__":
                gains.append(h[q.qid])
            else:
                by_value = collections.Counter()
                for c in cands:
                    by_value[fv.get((c.key, q.qid))] += b.get(c.key, 0.0)
                gains.append(1.0 - (max(by_value.values()) if by_value else 0.0))
        net = [g - self.cost for g in gains]
        scores = {"voi_llm": gains, "belief": [round(b.get(c.key, 0.0), 3) for c in cands], "p_answer": h, "cost": self.cost}
        if max(net) > 0:
            i = max(range(len(net)), key=lambda k: net[k])
            return Decision("ask", question=ctx.menu[i], scores=scores)
        # act on the maximum-belief interpretation
        best = max(cands, key=lambda c: b.get(c.key, 0.0))
        idx = ctx.greedy_index()
        if not best.greedy:
            try:
                progs = [R.intake_to_search_args(HoosierWorkflow._assume_none(n)) for n in best.artifact["needs"]]
                key = json.dumps(progs, sort_keys=True)
                for j, pc in enumerate(ctx.candidates):
                    if not pc.is_unresolved and json.dumps([p["arguments"] for p in pc.artifact.get("programs", [])], sort_keys=True) == key:
                        idx = j
                        break
            except Exception:
                pass
        return Decision("commit", idx, scores=scores)


def _commit_index(ctx: "ReceiverContext", best) -> int:
    """Planner-candidate index whose programs are the image of intake interpretation `best` (greedy index as fallback)."""
    idx = ctx.greedy_index()
    if best is None or best.greedy:
        return idx
    try:
        progs = [R.intake_to_search_args(HoosierWorkflow._assume_none(n)) for n in best.artifact["needs"]]
        key = json.dumps(progs, sort_keys=True)
        for j, pc in enumerate(ctx.candidates):
            if not pc.is_unresolved and json.dumps([p["arguments"] for p in pc.artifact.get("programs", [])], sort_keys=True) == key:
                return j
    except Exception:
        pass
    return idx


def _question_aspects(q: "Question") -> list[tuple[str, str]]:
    if q.scope.get("pairs") is not None:
        return [tuple(p) for p in q.scope["pairs"]]
    return [(q.scope.get("need_id", "*"), q.scope.get("field", ""))]


def _cand_value(c, need_id: str, field: str):
    need = next((n for n in c.artifact["needs"] if n["need_id"] == need_id), None)
    return None if need is None else need.get(field)


class SAGEController(Controller):
    """SAGE-Agent (Suri et al., Findings of ACL 2026, 'Structured Uncertainty guided Clarification for LLM Agents'):
    structured belief over tool-call candidates, pi_c = prod_j p(theta_j) with p = 1 for a specified parameter and
    1/|D_j| for an unspecified one (epsilon for a continuous domain); a question targeting aspects A(q) is scored by the
    expected value of perfect information EVPI(q) = E_r[max_c pi_c(t|q,r)] - max_c pi_c(t) (perfect resolution multiplies
    pi_c by |D_a| on the aspects where c is unspecified; an A-vs-B answer eliminates the other side) minus the redundancy cost
    lambda * sum_a n_a; the agent executes when max_c pi_c >= tau_exec or when the best net score < alpha * max_c pi_c.
    Paper constants: lambda = 0.5, alpha = 0.1, epsilon = 1e-4; tau_exec = 1 (execute once fully specified)."""
    name = "sage"
    LAMBDA, ALPHA, TAU_EXEC, EPS = 0.5, 0.1, 1.0, 1e-4

    def __init__(self):
        self.n_asked = collections.Counter()

    @staticmethod
    def _viability(c, domains: dict) -> float:
        p = 1.0
        for n in c.artifact["needs"]:
            if n.get("service_categories") in (R.UNKNOWN, R.EMPTY):
                p /= domains.get("service_categories", 21)
            for f in R.FIELDS:
                if n.get(f) in (R.UNKNOWN, R.EMPTY):
                    p /= domains.get(f, 10.0)
        return p

    def decide(self, ctx):
        pk = ctx.incoming[0]
        cands = [c for c in pk.candidates if not c.is_unresolved]
        if not ctx.menu or ctx.items_left <= 0 or not cands:
            return Decision("commit", ctx.greedy_index())
        domains, fv = ctx.aux.get("domain_sizes", {}), ctx.aux.get("field_value", {})
        pi = {c.key: self._viability(c, domains) for c in cands}
        pmax = max(pi.values())
        best = max(cands, key=lambda c: pi[c.key])
        if pmax >= self.TAU_EXEC:
            return Decision("commit", _commit_index(ctx, best), scores={"sage_pi": [round(pi[c.key], 6) for c in cands]})
        scores = []
        for q in ctx.menu:
            aspects = _question_aspects(q)
            if q.b == "__residual__":
                upd = {}
                for c in cands:
                    v = pi[c.key]
                    for nid, f in aspects:
                        if _cand_value(c, nid, f) in (R.UNKNOWN, R.EMPTY):
                            v *= domains.get(f, 10.0)
                    upd[c.key] = v
                evpi = max(upd.values()) - pmax
            else:
                sides = collections.defaultdict(list)
                for c in cands:
                    sides[fv.get((c.key, q.qid))].append(c.key)
                tot = sum(pi.values()) or 1.0
                evpi = sum((sum(pi[k] for k in ks) / tot) * max(pi[k] for k in ks) for ks in sides.values()) - pmax
            scores.append(evpi - self.LAMBDA * sum(self.n_asked[a] for a in aspects))
        i = max(range(len(scores)), key=lambda k: scores[k])
        sc = {"sage_score": [round(x, 6) for x in scores], "sage_pmax": pmax}
        if scores[i] < self.ALPHA * pmax:
            return Decision("commit", _commit_index(ctx, best), scores=sc)
        for a in _question_aspects(ctx.menu[i]):
            self.n_asked[a] += 1
        return Decision("ask", question=ctx.menu[i], scores=sc)


class ATDController(Controller):
    """Active Task Disambiguation (Kobalczyk et al., ICLR 2025): the LLM samples N complete candidate solutions compatible
    with the conversation (here: complete specifications of the person's constraints, every unknown field filled with a
    plausible value or 'none'); each candidate question is scored by its expected information gain over that solution set,
    estimated as the entropy of the answer partition (EIG(q) = -sum_k p_k log p_k with p_k the fraction of solutions whose
    implied answer is a_k); the question with the largest EIG - c is asked while that is positive (c constant;
    the original method fixes a question budget and has no budget-aware stopping)."""
    name = "atd"
    N_SOLUTIONS = 8

    def __init__(self, cost: float | None = None):
        self.cost = float(os.environ.get("RCLUP_ATD_COST", "0.5")) if cost is None else float(cost)

    @staticmethod
    def _solutions(sample, conv: str, greedy) -> list[dict]:
        if sample is None:
            return []
        prompt = [{"role": "system", "content": "You are the planner of an Indiana 211 resource-search team. Below is the current structured reading of the "
                   "person's needs; some fields are unknown. Propose ONE complete, plausible specification consistent with the conversation: keep "
                   "every known value and fill every unknown field with a concrete plausible value or \"none\" (no constraint). Field formats: "
                   "location {\"counties\": [...], \"cities\": [...], \"zipcodes\": [...]}; schedule {\"day\": \"mon\"..\"sun\", \"time\": \"HH:MM\"} or "
                   "{\"requires_24_hours\": true}; intake_methods / available_documents / eligibility: arrays. Output ONLY compact JSON: "
                   "{\"needs\": [{\"need_id\": \"need-1\", \"location\": ..., \"schedule\": ..., \"intake_methods\": [...], "
                   "\"available_documents\": [...], \"eligibility\": [...]}]}"},
                  {"role": "user", "content": "Conversation so far:\n" + conv + "\n\nCurrent reading:\n" + greedy.text}]
        out = []
        for text in sample(prompt, n=ATDController.N_SOLUTIONS, max_tokens=320, purpose="atd_solutions"):
            obj = R.parse_json_object(text) or {}
            needs = obj.get("needs")
            if not isinstance(needs, list):
                continue
            sol = {}
            for n in needs:
                if not isinstance(n, dict):
                    continue
                nid = str(n.get("need_id", "need-1"))
                sol[nid] = {f: json.dumps(R._norm_field(f, n.get(f, R.UNKNOWN)), sort_keys=True) for f in R.FIELDS}
            if sol:
                out.append(sol)
        return out

    def decide(self, ctx):
        pk = ctx.incoming[0]
        cands = [c for c in pk.candidates if not c.is_unresolved]
        if not ctx.menu or ctx.items_left <= 0 or not cands:
            return Decision("commit", ctx.greedy_index())
        sols = self._solutions(ctx.aux.get("sample"), ctx.aux.get("conversation", ""), pk.greedy)
        gains = []
        for q in ctx.menu:
            aspects = _question_aspects(q)
            answers = collections.Counter()
            for sol in sols:
                ans = tuple(sol.get(nid, {}).get(f, "?") for nid, f in aspects)
                answers[ans] += 1
            n = sum(answers.values())
            gains.append(-sum((k / n) * math.log(k / n) for k in answers.values()) if n else 0.0)
        sc = {"atd_eig": [round(g, 4) for g in gains], "n_solutions": len(sols), "cost": self.cost}
        i = max(range(len(gains)), key=lambda k: gains[k])
        if gains[i] - self.cost > 0:
            return Decision("ask", question=ctx.menu[i], scores=sc)
        return Decision("commit", ctx.greedy_index(), scores=sc)


class ProCoTController(Controller):
    """ProCoT (Deng et al., Findings of EMNLP 2023, 'Prompting and Evaluating LLMs for Proactive Dialogues'): proactive
    chain-of-thought prompting -- given the task background and the conversation, the LLM first analyses whether the
    current request is ambiguous/underspecified for the next action, then selects a dialogue act ('Ask a Clarification
    Question' or 'Directly Search') and the question. No belief model, no cost model, no budget reasoning."""
    name = "procot"

    def decide(self, ctx):
        pk = ctx.incoming[0]
        llm = ctx.aux.get("llm")
        if not ctx.menu or ctx.items_left <= 0 or llm is None:
            return Decision("commit", ctx.greedy_index())
        listing = "\n".join(f"[{i + 1}] {q.text}" for i, q in enumerate(ctx.menu))
        prompt = [{"role": "system", "content": "You are the planner of an Indiana 211 resource-search team. Task background: the team must run "
                   "search_resources with the person's service category, location, schedule, intake method, documents and eligibility "
                   "constraints (empty when the person has no requirement) and then recommend a resource; the whole conversation is capped at "
                   f"8 agent turns and {ctx.external_left} remain. Given the task background and the conversation history, first analyse whether "
                   "the current reading of the person's needs is ambiguous or incomplete for searching, and then select the appropriate action: "
                   "\"Directly Search\" with the current reading, or \"Ask a Clarification Question\" (one of the listed questions). Write a brief "
                   "analysis, then a final line with ONLY JSON: {\"act\": \"search\" | \"ask\", \"question\": <number or null>}."},
                  {"role": "user", "content": "Conversation so far:\n" + ctx.aux.get("conversation", "") + "\n\nCurrent reading:\n" + pk.greedy.text
                   + "\n\nClarification questions available:\n" + listing}]
        text = llm(prompt, max_tokens=320, purpose="procot")
        obj = None
        for m in re.finditer(r"\{[^{}]*\}", text):
            try:
                obj = json.loads(m.group(0))
            except ValueError:
                continue
        obj = obj or {}
        act = str(obj.get("act", "")).lower()
        sc = {"procot_act": act, "procot_analysis": text[:600]}
        if act.startswith("ask"):
            try:
                k = int(obj.get("question") or 1) - 1
            except (TypeError, ValueError):
                k = 0
            k = min(max(k, 0), len(ctx.menu) - 1)
            return Decision("ask", question=ctx.menu[k], scores=sc)
        return Decision("commit", ctx.greedy_index(), scores=sc)


class RoundRobinMenuController(Controller):
    """Reference policy: round-robin over the menu up to the budget, then pi_0 commit."""
    name = "round_robin_menu"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.k = 0

    def decide(self, ctx):
        if ctx.menu and ctx.items_left > 0:
            q = ctx.menu[self.k % len(ctx.menu)]
            self.k += 1
            return Decision("ask", question=q)
        return Pi0Controller.decide(self, ctx)


class ForcedController(Controller):
    """Branch executor: forced first decision, then a continuation policy (pi_0 by default; the fixed-order rule for the
    budgeted intake menu, so that 'ask q' means 'ask q and keep following the default rule', and 'commit' means 'stop asking now')."""
    name = "forced_then_pi0"

    def __init__(self, forced: Decision, seed: int, inner: "Controller | None" = None):
        self.forced = forced
        self.inner = inner if inner is not None else Pi0Controller(seed)
        self.used = False

    def decide(self, ctx):
        if not self.used:
            self.used = True
            return self.forced
        return self.inner.decide(ctx)


@dataclass
class ReceiverContext:
    role: str
    incoming: list[Packet]
    candidates: list[Candidate]
    menu: list[Question]
    items_left: int
    external_left: int
    task_state_text: str
    aux: dict = field(default_factory=dict)   # workflow-provided deterministic side information (shared by all controllers)

    def greedy_index(self) -> int:
        for i, c in enumerate(self.candidates):
            if c.greedy:
                return i
        return 0


@dataclass
class Invocation:
    inv_id: str
    role: str
    version: int
    messages: list[dict]
    tools: list[dict] | None
    packet: Packet
    selected: int
    purpose: str


@dataclass
class Trace:
    events: list[dict] = field(default_factory=list)

    def add(self, kind: str, **kw):
        self.events.append({"kind": kind, **kw})


class Workflow:
    """Base workflow with sender/receiver invocations; task subclasses define roles and rules."""
    dataset = "base"
    roles: dict[str, R.RoleSpec]

    def __init__(self, backbone: VLLMClient, controller: Controller, *, seed: int = 0, items_budget: int = 2,
                 prefix_hook: Callable | None = None):
        self.bb = backbone
        self.controller = controller
        self.seed = seed
        self.items_budget = items_budget
        self.prefix_hook = prefix_hook
        self.versions: dict[str, int] = {}
        self.invocations: list[Invocation] = []
        self.trace = Trace()
        self.items_used = 0
        self.rng = random.Random(seed)
        self.asked: set = set()
        self.decision_points = 0
        self.calls_episode = 0          # backbone calls (samples) this episode / this step; limited under the fan-out topology (G.7)
        self.calls_step = 0
        self.max_calls_episode: int | None = None
        self.max_calls_step: int | None = None
        self.greedy_counts = False       # G.2 support rule: cluster size / total samples (greedy counted, no pseudocount)

    def fork(self, controller: Controller, seed: int | None = None) -> "Workflow":
        """Copy mutable state for a branch; the backbone and the packets are shared."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.controller = controller
        clone.seed = self.seed if seed is None else seed
        clone.versions = dict(self.versions)
        clone.invocations = list(self.invocations)
        clone.trace = Trace()
        clone.items_used = self.items_used
        clone.rng = random.Random(clone.seed * 7919 + self.decision_points)
        clone.asked = set(self.asked)
        clone.prefix_hook = None
        if hasattr(self, "fallback_overrides"):
            clone.fallback_overrides = dict(self.fallback_overrides)
        if hasattr(self, "asked_count"):
            clone.asked_count = collections.Counter(self.asked_count)
        return clone

    # ----- helpers -----
    def next_version(self, role: str) -> int:
        self.versions[role] = self.versions.get(role, 0) + 1
        return self.versions[role]

    def invoke(self, role: str, messages: list[dict], *, tools=None, ancestors=(), sources=(), unresolved_text=None,
               purpose: str = "") -> Invocation:
        spec = self.roles[role]
        version = self.next_version(role)
        inv_id = f"{role}#{version}"
        s_draws = getattr(self, "s_draws", S_DRAWS)   # 0: greedy-only handoff (plain message-passing baselines)
        if s_draws and self.max_calls_episode is not None and (self.calls_episode + s_draws + 1 > self.max_calls_episode or
                                                              self.calls_step + s_draws + 1 > (self.max_calls_step or 10 ** 9)):
            self.trace.add("budget_degraded", inv_id=inv_id, role=role, calls_episode=self.calls_episode, calls_step=self.calls_step)
            s_draws = 0   # inference budget exhausted (G.7): greedy-only invocation instead of dropping the step
        self.calls_episode += s_draws + 1
        self.calls_step += s_draws + 1
        max_tokens = getattr(spec, "max_tokens", None) or PROPOSAL_MAX_TOKENS   # receivers emit per-candidate decisions and need a longer output budget
        if s_draws == 0:
            samples = self.bb.chat(messages, n=1, temperature=0.0, max_tokens=max_tokens, tools=tools, purpose=f"{self.dataset}:{role}:greedy")
        else:
            samples = self.bb.propose(messages, s=s_draws, max_tokens=max_tokens, seed=self.seed * 1000 + version,
                                      tools=tools, purpose=f"{self.dataset}:{role}")
        kmax_kw = {"k_max": self.k_max} if getattr(self, "k_max", None) else {}   # candidates kept per packet (K)
        if self.greedy_counts:
            kmax_kw.update(alpha=0.0, greedy_counts=True)
        packet = build_packet(inv_id, version, role, samples, spec.parse, spec.canon, spec.render,
                              ancestors=ancestors, sources=sources, unresolved_text=unresolved_text, **kmax_kw)
        if not any(not c.is_unresolved for c in packet.candidates):
            # All draws invalid: one retry with a different seed (logged); otherwise fail the invocation.
            samples = self.bb.propose(messages, s=max(s_draws, 1), max_tokens=max_tokens, seed=self.seed * 1000 + version + 500,
                                     tools=tools, purpose=f"{self.dataset}:{role}:retry")
            packet = build_packet(inv_id, version, role, samples, spec.parse, spec.canon, spec.render,
                                  ancestors=ancestors, sources=sources, unresolved_text=unresolved_text, **kmax_kw)
            packet.extra["retried"] = True
            if not any(not c.is_unresolved for c in packet.candidates):
                self.trace.add("invocation_failed", inv_id=inv_id, role=role, examples=[x.text[:1500] for x in samples[:3]])
                raise RuntimeError(f"No valid proposals for {inv_id}")
        invalid_examples = [x.text[:900] for x in samples[1:] if spec.parse(x.text) is None][:2]
        inv = Invocation(inv_id, role, version, messages, tools, packet, 0, purpose)
        self.invocations.append(inv)
        self.trace.add("invocation", inv_id=inv_id, role=role, version=version, purpose=purpose,
                       packet=packet.public(), n_samples=len(samples), invalid=packet.invalid_count,
                       invalid_examples=invalid_examples, finish_reasons=[x.finish_reason for x in samples])
        return inv

    def decide(self, ctx: ReceiverContext, inv: Invocation | None = None, snapshot: dict | None = None) -> Decision:
        self.decision_points += 1
        if self.prefix_hook is not None and snapshot is not None:
            self.prefix_hook(self, ctx, snapshot)
        ctx.aux.setdefault("llm", self.greedy_text)
        ctx.aux.setdefault("sample", lambda messages, n=8, max_tokens=256, purpose="aux": [
            x.text.strip() for x in self._count_calls(n) or self.bb.chat(messages, n=n, temperature=0.7, top_p=0.9, max_tokens=max_tokens,
                                                 seed=self.seed * 1000 + self.decision_points, purpose=f"{self.dataset}:{purpose}")])
        d = self.controller.decide(ctx)
        self.trace.add("decision", role=ctx.role, decision=d.kind, candidate=d.candidate,
                       question=None if d.question is None else d.question.__dict__, scores=d.scores,
                       menu=[q.__dict__ for q in ctx.menu], items_left=ctx.items_left)
        return d

    def _count_calls(self, n: int) -> None:
        self.calls_episode += n
        self.calls_step += n

    def greedy_text(self, messages, max_tokens=256, purpose="aux") -> str:
        self._count_calls(1)
        return self.bb.chat(messages, n=1, temperature=0.0, max_tokens=max_tokens, purpose=f"{self.dataset}:{purpose}")[0].text.strip()

    def record_extra(self, rec: dict, outcome: dict, cycles: int) -> dict:
        rec.update(outcome, controller=self.controller.name, items_used=self.items_used, cycles=cycles,
                   trace=self.trace.events, invocations=[{"inv_id": i.inv_id, "role": i.role, "selected": i.selected} for i in self.invocations])
        return rec


# ==========================================================================
# HoosierHelp
# ==========================================================================
class HoosierWorkflow(Workflow):
    dataset = "hoosier"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from .envs.hoosier_env import native_tool_schemas, native_agent_prompt, service_categories
        from .common import ensure_vendor_on_path
        ensure_vendor_on_path()
        from tools.tool_protocol import qwen_tool_schema
        self.schemas = native_tool_schemas()
        self.search_schema = [s for s in self.schemas if s["name"] == "search_resources"]
        final_schema = [s for s in self.schemas if s["name"] == "final_recommendation"][0]
        self.roles = R.hoosier_roles(qwen_tool_schema(self.search_schema[0]), qwen_tool_schema(final_schema))
        self.native_prompt = native_agent_prompt()
        self.categories = service_categories()
        from tools.curated_categories import SERVICE_CATEGORY_DESCRIPTIONS
        self.category_help = "\n".join(f"- {c}: {SERVICE_CATEGORY_DESCRIPTIONS.get(c, '')}" for c in self.categories)
        self.llm_planner = False  # deterministic planner mapping by default (an LLM planner is optional)
        self.confirmation_questions = False  # confirmation questions off: evidence-resolved confirmations carry no new information
        self.contradiction_clarification = True  # ask the person directly (the simulated user follows the documented protocol, see npc.py)
        self.asked_count = collections.Counter()   # (need_id, field) -> times the person was asked for it
        self.prune_on_answer = False  # candidate-indexed clarification: an evidence-resolved answer prunes the sender's candidates instead of re-proposing
        self.user_direct = True  # disagreement/unknown questions go to the person directly (the evidence is ambiguous by construction)
        self.use_relaxation = True  # counterfactual relaxation diagnosis after empty results
        self.intake_groups = R.INTAKE_GROUPS   # question grouping for ordinary intake (three low-burden messages by default)
        self.skip_fields: tuple = ()           # ordinary-intake fields never asked (treated as no constraint)
        # fan-out state (paper App. G)
        self.receiver_cache: dict = {}         # role -> (projection key, Invocation): a receiver re-runs only when its input changed (G.7)
        self.omega_max = 4                     # joint interpretations retained (set to 16 by set_topology, G.2)
        self.max_repair_rounds: int | None = None   # at most 2 verification/repair rounds under fan-out (G.7)
        self.fallback_norm_fields = ("location", "schedule", "intake_methods")   # fields a fallback reply may change (fan-out adds documents/eligibility)

    def set_variant(self, variant: str) -> None:
        """Intake-question protocol.  "native": the benchmark's ordinary intake questions are fixed rules (three low-burden
        multi-item messages) and the controller decides clarifications only.  "atomic": ordinary intake questions enter the
        controller's menu as group questions ("ask the person about <fields> for <needs>") and committing treats unknown
        fields as no constraint, so under the turn budget every question is the controller's decision."""
        if variant not in ("native", "atomic"):
            raise ValueError(f"unknown variant {variant!r} (native | atomic)")
        self.use_relaxation, self.llm_planner, self.confirmation_questions, self.user_direct, self.prune_on_answer = True, False, False, True, True
        self.fallback_clarification = True
        self.budget_questions = variant == "atomic"
        self.priors = self._group_priors() if self.budget_questions else {}
        self.last_fallback_request = None
        self.fallback_overrides = {}
        from .envs.hoosier_env import _Native
        index = _Native.load()["index"]
        self.known_counties = {c.upper() for r in index.resources for c in (r.counties or ())}
        self.known_cities = {str(r.city).upper() for r in index.resources if getattr(r, "city", None)}

    _PRIORS = None

    def _group_priors(self) -> dict:
        """Priors for the budget-aware VOI rule, estimated once from the training specs (never from the test split):
        P(a need has a constraint in each intake question group) and P(the spec needs a fallback search)."""
        if HoosierWorkflow._PRIORS is None:
            from .common import DATA, read_jsonl, ensure_vendor_on_path
            ensure_vendor_on_path()
            from eval import runner
            rows = [r for r in read_jsonl(DATA / "hoosier" / "train_specs.jsonl") if r.get("split") != "test"]   # all train partitions
            keys = {"location": ("counties", "cities", "zipcodes"), "schedule": ("schedule",), "intake_methods": ("intake_methods",),
                    "available_documents": ("available_documents",), "eligibility": ("eligibility",)}
            hits, n = collections.Counter(), 0
            for r in rows:
                for e in runner.expected_tool_calls(r):
                    n += 1
                    for f, ks in keys.items():
                        if any(e.get(k) for k in ks):
                            hits[f] += 1
            HoosierWorkflow._PRIORS = {"field": {f: hits[f] / max(1, n) for f in keys},
                                       "fallback": sum(r.get("constraint_profile") == "fallback_required" for r in rows) / max(1, len(rows)), "n_needs": n}
        return HoosierWorkflow._PRIORS

    @staticmethod
    def _assume_none(need: dict) -> dict:
        return {k: (R.NONE if (k in R.FIELDS and v in (R.UNKNOWN, R.EMPTY)) else v) for k, v in need.items()}

    # ----- prompts -----
    DENIAL_RE = re.compile(r",?\s*\b(but|however|although|though)\b\s+(I|we)\s+(cannot|can't|can not|am not able to|won't|do not|don't)\b[^.?!;]*", re.I)

    def normalize_contradiction(self, text: str) -> str:
        """Task module: the benchmark's self-contradictory user states the real fact and denies it in one sentence
        ('X works for me, but I cannot do X'); the denial clause is dropped so every agent reads the stated fact."""
        return self.DENIAL_RE.sub("", text) if getattr(self, "contradiction_normalizer", True) else text

    def _conversation_text(self, env) -> str:
        lines, k = [], 0
        for m in env.messages:
            if m["role"] == "user" and not m["content"].startswith("Tool result for search_resources call"):
                k += 1
                lines.append(f"[person {k}] {self.normalize_contradiction(m['content'])}")
            elif m["role"] == "assistant" and "<tool_call>" not in m["content"]:
                lines.append(f"[agent] {m['content']}")
            elif m["role"] == "user":
                lines.append("[system] " + m["content"].split("\n")[0] + (" (no resources matched)" if "No resources matched" in m["content"] else " (resources returned)"))
        return "\n".join(lines)

    def _person_text(self, env) -> str:
        return " ".join(m["content"] for m in env.messages if m["role"] == "user"
                        and not m["content"].startswith("Tool result for search_resources call")).lower()

    def intake_messages(self, env) -> list[dict]:
        system = self.roles["intake"].task + "\n\nAllowed service_categories values (choose the closest):\n" + self.category_help
        user = "Conversation so far:\n" + self._conversation_text(env)
        note = getattr(self, "last_fallback_request", None)
        if note:
            user += (f"\n\nNote: the last search returned no resources and the person was asked whether their {note} could change. "
                     "Apply their latest answer to the affected need(s): a new value replaces the old one; 'I can change it', 'if needed', "
                     "'flexible' or 'no longer needed' means set that field to \"none\"; 'cannot change' keeps the old value.")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def planner_messages(self, opening: str, packet: Packet, replies: list[dict]) -> list[dict]:
        system = self.roles["planner"].task
        user = f"Public opening from the person: {opening}\n\n{packet.readable()}"
        if replies:
            user += "\n\nClarification replies from the intake agent:\n" + "\n".join(
                f"Q: {r['question']}\nA: {r['answer']} (sources: person turns {r.get('sources')}; resolved={r.get('resolved')})" for r in replies)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def retriever_messages(self, opening: str, planner_packet: Packet, selected: Candidate, results: list[dict], diagnosis: dict | None = None) -> list[dict]:
        user = (f"Public opening from the person: {opening}\n\n{planner_packet.readable()}\n\nExecuted program: {selected.text}\n\n"
                "Search results (JSON):\n" + json.dumps(results, ensure_ascii=False))
        if diagnosis:
            user += ("\n\nRelaxation diagnosis (resources returned per need if these constraints were dropped): "
                     + json.dumps(diagnosis["report"]) + f"\nSmallest relaxations that make every need return resources: {diagnosis['minimal'][:3] or 'none'}")
        return [{"role": "system", "content": self.roles["retriever"].task}, {"role": "user", "content": user}]

    def recover_location(self, replies_text: str) -> dict | None:
        """Source-grounded literal recovery: county/ZIP/city mentioned by the person, validated against the index."""
        import re as _re
        found = {}
        replies_text = replies_text.lower()   # callers pass raw replies too; city patterns are lower-case
        counties = [m.group(1).strip().upper() for m in _re.finditer(r"([A-Za-z][A-Za-z.' ]{1,30}?)\s+county\b", replies_text, flags=_re.I)]
        counties = [c.split()[-1] if c.split()[-1] in self.known_counties else c for c in counties]
        counties = sorted({c for c in counties if c in self.known_counties})
        zips = sorted(set(_re.findall(r"\b(\d{5})\b", replies_text)))
        if counties:
            found["counties"] = counties
        if zips:
            found["zipcodes"] = zips
        if not found:
            cities = sorted({c for c in self.known_cities if len(c) > 3 and _re.search(r"\b" + _re.escape(c.lower()) + r"\b", replies_text)})
            cities = [c for c in cities if not any(o != c and c in o for o in cities)]   # longest match: 'North Vernon' beats 'Vernon'
            if len(cities) == 1:
                found["cities"] = cities
        return found or None

    CATEGORY_KEYWORDS = {
        "Material Goods": (r"\bhousehold", r"\bmaterial", r"\bgoods\b", r"\bitems\b", r"\bfurniture", r"\bclothes", r"\bdiapers"),
        "Tax Help": (r"\btax",), "Legal and Court Help": (r"\blegal", r"\bcourt", r"\blawyer"),
        "Financial Assistance and Benefits": (r"\bfinancial", r"\bbenefit", r"\bmoney", r"\bbills?\b", r"\bpaying\b"),
        "Public Safety": (r"\bsafety", r"\bpolice"), "Transportation": (r"\btransport", r"\bride\b", r"\bbus\b"),
        "Substance Use Services": (r"\bsubstance", r"\brehab", r"\bdrug", r"\balcohol"),
        "Employment and Job Training": (r"\bjobs?\b", r"\bemployment", r"\btraining"),
        "Pet and Animal Services": (r"\bpets?\b", r"\bdog", r"\bcat\b", r"\banimal", r"\bvet\b"),
        "Pregnancy and Reproductive Health": (r"\bpregnan", r"\breproductive", r"\bprenatal"),
        "Utility Assistance": (r"\butilit", r"\belectric", r"\bgas bill", r"\bwater bill"),
        "Family and Caregiver Services": (r"\bcaregiver", r"\bfamily (support|services|help)"),
        "Medical Support Services": (r"\bmedical", r"\bdoctor"), "Disability and Rehabilitation": (r"\bdisabilit", r"\brehabilitation"),
        "Community and Recreation": (r"\bcommunity", r"\brecreation", r"\bactivities"),
        "Housing and Shelter": (r"\bhousing", r"\bshelter", r"\bplace to stay", r"\bsomewhere to stay"),
        "Disaster and Environmental Services": (r"\bdisaster", r"\benvironment", r"\bstorm", r"\bflood"),
        "Food Assistance": (r"\bfood\b", r"\bgroceries", r"\bhungry"), "Health Care": (r"\bhealth ?care", r"\bclinic"),
    }
    VALUE_LITERALS = {
        "intake_methods": {"call": (r"\bcall", r"\bphone"), "email": (r"\be-?mail",), "mail": (r"(?<![e-])\bmail\b", r"\bpostal"), "online": (r"\bonline", r"\bwebsite"),
                           "text": (r"\btext",), "walk_in": (r"\bwalk", r"\bin[- ]person"), "appointment": (r"\bappointment",)},
        "day": {"mon": (r"\bmon(day)?s?\b",), "tue": (r"\btue(s|sday)?s?\b",), "wed": (r"\bwed(nesday)?s?\b",), "thu": (r"\bthu(rs|rsday)?s?\b",),
                "fri": (r"\bfri(day)?s?\b",), "sat": (r"\bsat(urday)?s?\b",), "sun": (r"\bsun(day)?s?\b",)},
    }

    def _user_sentences(self, env) -> list[str]:
        out = []
        for m in env.messages:
            if m["role"] != "user" or m["content"].startswith("Tool result for search_resources call"):
                continue
            for sent in re.split(r"(?<=[.!?;])\s+", self.normalize_contradiction(m["content"])):
                # clause level: 'For the family support, I need Thursday, and for the legal help, ...'
                out += [x.strip().lower() for x in re.split(r",?\s+(?:and |but |while )?(?=for (?:the |my |both )?\w)", sent) if x.strip()]
        return out

    def ground_category(self, env, needs: list[dict]) -> bool:
        """Task module: the service category must be supported by the person's words. If the chosen category's keywords
        never occur but exactly one other category's keywords do (e.g. 'substance use' read as Disability and
        Rehabilitation), switch to that category. Categories already taken by the other need are excluded."""
        text = " ".join(self._user_sentences(env))
        first = {}
        for cat, kws in self.CATEGORY_KEYWORDS.items():
            pos = [m.start() for k in kws for m in [re.search(k, text)] if m]
            if pos:
                first[cat] = min(pos)
        taken = {n["service_categories"][0] for n in needs if isinstance(n["service_categories"], list) and n["service_categories"] and n["service_categories"][0] in first}
        changed = False
        for n in needs:   # needs are numbered in order of mention, so unsupported needs take unclaimed categories by first mention
            cat = n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else None
            if cat is None or cat in first:
                continue
            cands = sorted((pos, c) for c, pos in first.items() if c not in taken)
            if cands:
                self.trace.add("category_grounding", need_id=n["need_id"], old=cat, new=cands[0][1])
                n["service_categories"] = [cands[0][1]]
                taken.add(cands[0][1])
                changed = True
        return changed

    @staticmethod
    def _time_patterns(hhmm: str) -> tuple:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm or "")
        if not m:
            return ()
        h, mm = int(m.group(1)), m.group(2)
        forms = {f"{h}:{mm}", f"{h % 12 or 12}:{mm}", f"{h:02d}:{mm}"}
        if mm == "00":
            forms |= {f"{h % 12 or 12}\s*(am|pm|o'clock)", f"{h}\s*(am|pm)"}
        return tuple(r"\b" + f + r"\b" for f in forms)

    def ground_binding(self, env, needs: list[dict]) -> bool:
        """Task module (composite needs): a concrete intake-method / weekday value attached to need A must be supported by a
        sentence that is about need A (or about both / no particular need); if every sentence that mentions the value is
        about need B only, the value was bound to the wrong need and is dropped from A."""
        sents = self._user_sentences(env)
        kws = []
        for n in needs:
            cat = n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else ""
            kws.append(self.CATEGORY_KEYWORDS.get(cat, ()))
        if not all(kws):
            return False
        generic = re.compile(r"\b(both|either|each|all of|neither)\b")
        changed = False
        for i, n in enumerate(needs):
            mine, other = kws[i], kws[1 - i]
            checks = []
            if isinstance(n.get("intake_methods"), list):
                for v in n["intake_methods"]:
                    pats = self.VALUE_LITERALS["intake_methods"].get(v)
                    if pats:
                        checks.append(("intake_methods", pats))
            if isinstance(n.get("schedule"), dict) and n["schedule"].get("day") in self.VALUE_LITERALS["day"]:
                checks.append(("schedule", self.VALUE_LITERALS["day"][n["schedule"]["day"]]))
            if isinstance(n.get("schedule"), dict):
                for tk in ("start_time", "end_time", "time"):
                    pats = self._time_patterns(n["schedule"].get(tk, ""))
                    if pats:
                        checks.append(("schedule_time", pats))
            for f, pats in checks:
                hits = [s for s in sents if any(re.search(p, s) for p in pats)]
                if not hits:
                    continue
                if all(any(re.search(k, s) for k in other) and not any(re.search(k, s) for k in mine) and not generic.search(s) for s in hits):
                    if f == "schedule_time":
                        # the window belongs to the other need: keep the day, drop the times
                        if not isinstance(n["schedule"], dict):
                            continue
                        n["schedule"] = {k: v for k, v in n["schedule"].items() if k not in ("start_time", "end_time", "time")} or R.NONE
                        self.trace.add("binding_grounding", need_id=n["need_id"], field="schedule_time", dropped=True)
                        changed = True
                        continue
                    was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                    n[f] = R.NONE if was_asked else R.UNKNOWN
                    changed = True
                    self.trace.add("binding_grounding", need_id=n["need_id"], field=f, dropped=True)
        return changed

    # ----- task-specific module: provenance checks on intake candidates -----
    def check_intake_packet(self, env, packet: Packet) -> None:
        """(a) an empty container is 'none' only if the field was asked, else 'unknown'; a 'none' claim for an
        unasked field becomes 'unknown'; (b) location literals must appear in the person's own words."""
        text = self._person_text(env)
        replies_text = " ".join(m["content"] for m in env.messages[1:] if m["role"] == "user"
                                and not m["content"].startswith("Tool result for search_resources call")).lower()
        spec = self.roles["intake"]
        merged: dict[str, Candidate] = {}
        for c in packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for n in art["needs"]:
                for f in R.FIELDS:
                    was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                    if n[f] == R.EMPTY:
                        n[f] = R.NONE if was_asked else R.UNKNOWN
                        changed = True
                    if n[f] == R.UNKNOWN and self.asked_count.get((n["need_id"], f), 0) + self.asked_count.get(("*", f), 0) >= 2:
                        n[f] = R.NONE   # asked twice and still nothing concrete: treat as no constraint instead of asking a third time
                        changed = True
                    elif n[f] == R.NONE and not was_asked and f not in self.skip_fields:
                        n[f] = R.UNKNOWN
                        changed = True
                loc = n["location"]
                if isinstance(loc, dict):
                    kept = {}
                    for k, vals in loc.items():
                        ok = [v for v in vals if re.search(r"\b" + re.escape(v.lower()) + r"\b", text)]
                        if ok:
                            kept[k] = ok
                    if kept != loc:
                        n["location"] = kept if kept else (self.recover_location(replies_text) or R.UNKNOWN)
                        changed = True
                elif loc == R.UNKNOWN and ((n["need_id"], "location") in self.asked or ("*", "location") in self.asked):
                    rec = self.recover_location(replies_text)
                    if rec:
                        n["location"] = rec
                        changed = True
                # documents / eligibility literals must be traceable to the person's replies (not the need statement)
                for f in ("available_documents", "eligibility"):
                    if isinstance(n[f], list):
                        ok = [v for v in n[f] if any(k in replies_text for k in R.FIELD_VALUE_LEXICON.get(v, [v.replace("_", " ")]))]
                        if ok != n[f]:
                            was_asked = (n["need_id"], f) in self.asked or ("*", f) in self.asked
                            n[f] = ok if ok else (R.NONE if was_asked else R.UNKNOWN)
                            changed = True
                sch = n["schedule"]
                if isinstance(sch, dict):
                    cleaned = {k: v for k, v in sch.items() if k not in ("time", "start_time", "end_time") or R.time_supported(v, text)}
                    if cleaned.get("requires_24_hours") and not re.search(r"\b24\b|twenty[- ]four|around the clock|any time of day", text):
                        cleaned = {k: v for k, v in cleaned.items() if k != "requires_24_hours"}
                    # a single time that is actually the start of a stated window becomes that window
                    if "time" in cleaned:
                        for a, b in R.time_ranges(replies_text):
                            if cleaned["time"] == a:
                                cleaned = {k: v for k, v in cleaned.items() if k != "time"} | {"start_time": a, "end_time": b}
                                break
                    if cleaned != sch:
                        n["schedule"] = cleaned if cleaned else R.UNKNOWN
                        changed = True
                for f in self.skip_fields:   # never asked (applied after grounding): assume no constraint
                    if n[f] in (R.UNKNOWN, R.EMPTY):
                        n[f] = R.NONE
                        changed = True
                # fallback resolution overrides (ordinary repair, applied to every candidate)
                for (need_id, f), action in (self.fallback_overrides or {}).items():
                    if need_id in (n["need_id"], "*") and action is not None:
                        if n[f] != action:
                            n[f] = action
                            changed = True
            changed = self.ground_category(env, art["needs"]) or changed
            if len(art["needs"]) == 2:
                changed = self.ground_binding(env, art["needs"]) or changed
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
                m.sources = sorted(set(m.sources) | set(c.sources))
                if c.rep_logprob > m.rep_logprob:
                    m.rep_text, m.rep_logprob = c.rep_text, c.rep_logprob
            else:
                merged[c.key] = c
        packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))

    # ----- fan-out topology (paper App. G): intake -> {access, schedule, logistics} receivers -> deterministic joint planner -----
    topology = "chain"

    def set_topology(self, topology: str) -> None:
        """fanout (paper App. G): intake -> {access, schedule, logistics} receivers over the intake candidates plus domain
        evidence -> deterministic joint planner -> retriever, with the paper's parameters as defaults.  "chain" is the plain
        intake -> planner -> retriever pipeline the fan-out roles are built on; the released driver always selects fanout."""
        self.topology = topology
        if topology == "fanout":
            from . import fanout2 as F2
            self.roles.update(F2.receiver_roles())
            self.s_draws, self.k_max, self.greedy_counts = F2.S_DRAWS_FANOUT, F2.K_MAX_FANOUT, True   # G.2
            self.omega_max = F2.OMEGA_MAX                                                              # G.2
            self.max_calls_step, self.max_calls_episode = F2.MAX_CALLS_STEP, F2.MAX_CALLS_EPISODE      # G.7
            self.max_repair_rounds = F2.MAX_REPAIR_ROUNDS                                              # G.7
            self.p_fallback_override = F2.P_ACCEPT                                                     # G.5
            self.fallback_norm_fields = ("location", "schedule", "intake_methods", "available_documents", "eligibility")   # G.4 ladder
            if isinstance(self.controller, VOIController):                                             # G.5 / G.6
                self.controller.depth, self.controller.c_item, self.controller.c_fallback = F2.VOI_DEPTH, F2.C_ITEM, F2.C_FALLBACK

    def sender_invocation(self, env) -> Invocation:
        """Sender side of a cycle: under fanout, fanout2.sender_invocation (intake -> receivers -> joint packet)."""
        if self.topology == "fanout":
            from . import fanout2 as F2
            return F2.sender_invocation(self, env)
        inv = self.invoke("intake", self.intake_messages(env), tools=self.roles["intake"].tools,
                          sources=list(range(1, len(env.user_replies) + 2)), purpose="intake")
        self.check_intake_packet(env, inv.packet)
        self.trace.add("intake_checked", inv_id=inv.inv_id, packet=inv.packet.public())
        return inv

    def planner_packet_from_intake(self, intake_packet: Packet, version: int) -> Packet:
        """Deterministic planner: each resolved intake interpretation maps to exactly one search program, so the
        planner's candidates are the images of the intake candidates (support inherited), plus UNRESOLVED."""
        spec = self.roles["planner"]
        groups: dict[str, dict] = {}
        for c in intake_packet.candidates:
            if c.is_unresolved:
                continue
            programs = []
            for n in c.artifact["needs"]:
                args = R.intake_to_search_args(self._assume_none(n) if self.budget_questions else n)
                if args is None:
                    programs = []
                    break
                programs.append({"need_id": n["need_id"], "arguments": args})
            if not programs:
                continue
            try:
                key, norm = spec.canon({"programs": programs})
            except ValueError:
                continue
            g = groups.setdefault(key, {"artifact": norm, "count": 0, "weight": 0.0, "greedy": False, "sources": set(), "src": c})
            g["count"] += c.count
            g["weight"] += c.weight
            g["greedy"] = g["greedy"] or c.greedy
            g["sources"] |= set(c.sources)
        cands = [Candidate(key=k, artifact=g["artifact"], text=spec.render(g["artifact"]), count=g["count"], weight=g["weight"],
                           greedy=g["greedy"], rep_text=g["src"].rep_text, rep_logprob=g["src"].rep_logprob, sources=sorted(g["sources"]))
                 for k, g in groups.items()]
        cands.sort(key=lambda c: (not c.greedy, -c.count))
        if not any(c.greedy for c in cands) and cands:
            cands[0].greedy = True
        if not (self.budget_questions and cands):
            # budgeted intake: unknown fields are already mapped to 'no constraint', so every interpretation has a program; an UNRESOLVED
            # option would re-enter the fixed intake rule and act as a hidden 'ask everything' action
            cands.append(Candidate(key=UNRESOLVED_KEY, artifact={"unresolved": True},
                                   text="UNRESOLVED: the evidence does not support a search program yet.", is_unresolved=True))
        total = sum(c.weight for c in cands if not c.is_unresolved)
        return Packet(origin=f"planner#{version}", version=version, sender_role="planner", candidates=cands,
                      residual_weight=max(0.0, 1.0 - total), residual_count=intake_packet.residual_count,
                      invalid_count=0, n_stochastic=intake_packet.n_stochastic, ancestors=[intake_packet.origin], sources=list(intake_packet.sources))

    def check_planner_packet(self, intake_packet: Packet, planner_packet: Packet) -> None:
        """Programs may only use location literals and schedules that some intake interpretation states (no invention)."""
        allowed = {"counties": set(), "cities": set(), "zipcodes": set()}
        schedules = []
        for c in intake_packet.candidates:
            if c.is_unresolved:
                continue
            for n in c.artifact["needs"]:
                loc = n["location"]
                if isinstance(loc, dict):
                    for k, vals in loc.items():
                        allowed[k].update(v.upper() for v in vals)
                if isinstance(n["schedule"], dict):
                    schedules.append(n["schedule"])
        spec = self.roles["planner"]
        merged: dict[str, Candidate] = {}
        for c in planner_packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for prog in art["programs"]:
                a = prog["arguments"]
                for k in ("counties", "cities", "zipcodes"):
                    kept = [v for v in a[k] if v.upper() in allowed[k]]
                    if kept != a[k]:
                        a[k] = kept
                        changed = True
                if a["schedule"] and a["schedule"] not in schedules:
                    a["schedule"] = {}
                    changed = True
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
                if c.rep_logprob > m.rep_logprob:
                    m.rep_text, m.rep_logprob = c.rep_text, c.rep_logprob
            else:
                merged[c.key] = c
        planner_packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))

    def planner_aux(self, env, packet: Packet, menu: list[Question]) -> dict:
        """Deterministic side information for heuristic controllers: per interpretation, the search outcome signature
        (resource ids per need, or 'unresolved'), the value each interpretation assigns to every menu field, and the
        task's fixed field order."""
        ind, index = env.n["indiana211"], env.n["index"]
        # offline analysis calls this with self=None: budget mode is recognisable from the menu (group questions)
        budget = getattr(self, "budget_questions", False) or any(q.scope.get("pairs") is not None for q in menu)
        priors = (getattr(self, "priors", None) or HoosierWorkflow._group_priors(None)) if budget else {}
        signature, field_value = {}, {}
        for c in packet.candidates:
            if c.is_unresolved:
                continue
            sig = []
            for n in c.artifact["needs"]:
                args = R.intake_to_search_args(HoosierWorkflow._assume_none(n) if budget else n)
                if args is None:
                    sig.append("unresolved")
                else:
                    res = ind.execute_search_resources(index, args, limit=env.tool_limit).get("resources") or []
                    sig.append(tuple(sorted(r.get("resource_id", r.get("id", str(r))) if isinstance(r, dict) else str(r) for r in res)))
            signature[c.key] = tuple(sig)
            for q in menu:
                if q.scope.get("pairs") is not None:   # group question: unknown if any of its (need, field) pairs is unknown here
                    vals = [n.get(f) for nid, f in q.scope["pairs"] for n in c.artifact["needs"] if n["need_id"] == nid]
                    field_value[(c.key, q.qid)] = "unknown" if any(v in (R.UNKNOWN, R.EMPTY) for v in vals) else json.dumps(vals, sort_keys=True)
                    continue
                need = next((n for n in c.artifact["needs"] if n["need_id"] == q.scope.get("need_id")), None)
                v = need.get(q.scope["field"]) if need is not None else "missing"
                field_value[(c.key, q.qid)] = "unknown" if v in (R.UNKNOWN, R.EMPTY) else json.dumps(v, sort_keys=True)
        if getattr(self, "topology", "chain") == "fanout":
            # G.3: a decision signature is (admissible decisions, remaining obligations); an interpretation that leaves a
            # receiver's obligation open differs from one that does not even when the search outcome is the same
            obl = packet.extra.get("obligations", {})
            signature = {k: (v, tuple(obl.get(k, ()))) for k, v in signature.items()}
        fields = ("service_categories",) + R.FIELDS
        order = {q.qid: (("", q.scope["group"]) if q.scope.get("pairs") is not None else
                         (q.scope.get("need_id", ""), fields.index(q.scope["field"]) if q.scope.get("field") in fields else 99)) for q in menu}
        rank = {qid: i for i, (qid, _) in enumerate(sorted(order.items(), key=lambda kv: kv[1]))}
        prior = {}
        if budget:
            pf = priors.get("field", {})
            for q in menu:
                if q.scope.get("pairs") is not None:
                    prior[q.qid] = max(pf.get(f, 0.0) for _, f in q.scope["pairs"])   # P(some asked field is a real constraint)
        return {"signature": signature, "field_value": field_value, "field_order": rank, "prior": prior,
                "p_fallback": (self.p_fallback_override if (self is not None and getattr(self, "p_fallback_override", None) is not None)
                               else priors.get("fallback", 0.0)), "turns_left": env.turns_left,
                "conversation": self._conversation_text(env) if self is not None else "",
                "domain_sizes": {"service_categories": len(index.service_categories), "schedule": 1.0 / SAGEController.EPS,
                                 "location": (len(self.known_counties) + len(self.known_cities)) if self is not None else 200,
                                 "intake_methods": 7, "available_documents": 8, "eligibility": 11}}

    # ----- task-specific module: atomic question menu over (need, field) distinctions -----
    def planner_menu(self, packet: Packet, asked: set[str], env=None) -> list[Question]:
        """Atomic (need, field) questions: candidate-vs-candidate where interpretations disagree, candidate-vs-unknown
        where a field is unresolved, and candidate-vs-residual confirmations of agreed concrete values. Ordered by
        tier (disagreement, unknown, confirmation) then field order; capped at Q_MAX."""
        cands = [c for c in packet.candidates if not c.is_unresolved]
        if not cands:
            return []
        tiers = {0: [], 1: [], 2: []}
        need_ids = sorted({n["need_id"] for c in cands for n in c.artifact["needs"]})
        for f in ("service_categories",) + R.FIELDS:
            for need_id in need_ids:
                qid = f"{need_id}:{f}"
                if qid in asked:
                    continue
                values = {}
                for c in cands:
                    need = next((n for n in c.artifact["needs"] if n["need_id"] == need_id), None)
                    v = json.dumps(need[f], sort_keys=True) if need is not None else "missing"
                    values.setdefault(v, []).append(c)
                has_unknown = any(json.loads(v) == R.UNKNOWN for v in values if v != "missing")
                ordered = sorted(values.items(), key=lambda kv: -sum(c.weight for c in kv[1]))
                a_val, a_c = ordered[0]
                label = R.FIELD_LABEL.get(f, f)
                names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                         for c in cands for n in c.artifact["needs"]}
                who = names.get(need_id, need_id)
                if len(values) >= 2:
                    b_val, b_c = ordered[1]
                    text = (f"For the {who} need, which {label} applies: {a_val} or {b_val}? "
                            "Answer from what the person actually said; if they never said, ask them.")
                    tiers[0].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, b_c[0].key, text, tier=0))
                elif has_unknown:
                    text = (f"For the {who} need, what is the person's {label}? It is currently unknown. "
                            "Answer from what the person said; if they never said, ask them.")
                    tiers[1].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, "__residual__", text, tier=1))
                elif f != "service_categories":
                    text = (f"For the {who} need, is {label} = {a_val} exactly what the person said (including whether they have no constraint)? "
                            "Quote the person's words; if the value is not supported, correct it or ask them.")
                    tiers[2].append(Question(qid, "intake", {"need_id": need_id, "field": f}, a_c[0].key, "__residual__", text, tier=2))
        menu = tiers[0] + ([] if self.budget_questions else tiers[1]) + (tiers[2] if self.confirmation_questions else [])
        if self.budget_questions:
            menu = self.group_questions(packet, asked) + menu
        if env is not None and self.contradiction_clarification:
            menu = self.contradiction_questions(env, cands, asked) + menu
        if self.topology == "fanout":
            from . import fanout2 as F2
            for q in menu:
                q.footprint = F2.question_footprint(q)
            menu = [q for q in menu if len(q.footprint) > 2]   # G.3 prefilter: a premise no receiver depends on is not asked
        return menu[:Q_MAX]

    def group_questions(self, packet: Packet, asked: set[str]) -> list[Question]:
        """Budgeted-intake menu items: one question per ordinary-intake group still unknown in the greedy interpretation (asked to the
        person in one low-burden message); each group may be asked at most twice. Costs a turn, not a budget item."""
        g = packet.greedy.artifact
        names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                 for n in g["needs"]}
        out = []
        for gi, group in enumerate(self.intake_groups):
            pairs = [(n["need_id"], f) for n in g["needs"] for f in group if n[f] in (R.UNKNOWN, R.EMPTY) and f not in self.skip_fields]
            if not pairs:
                continue
            k = sum(1 for q in asked if q.startswith(f"*:group{gi}:"))
            if k >= 2:
                continue
            labels = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in pairs}))
            needs = " and ".join(sorted({names[nid] for nid, _ in pairs}))
            text = f"Ask the person about their {labels} for {needs} (currently unknown; committing assumes no constraint)."
            out.append(Question(f"*:group{gi}:{k}", "intake", {"need_id": "*", "field": f"group{gi}", "group": gi, "pairs": pairs},
                                packet.greedy.key, "__residual__", text, items=0, tier=1))
        return out

    CONTRA_RE = re.compile(r"\b(but|however|although|though)\b[^.?!]{0,80}\b(can't|cannot|can not|not able|won't|don't|do not|isn't|is not|doesn't|does not|unavailable)\b", re.I)
    FIELD_WORDS = {"location": ("county", "city", "zip", "area", "town", "located", "location", "go to", "travel", "live"),
                   "schedule": ("day", "time", "morning", "afternoon", "evening", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                                "24 hours", "weekday", "weekend", "schedule", "am", "pm", "o'clock"),
                   "intake_methods": ("intake", "phone", "call", "email", "mail", "online", "walk", "in person", "in-person", "text", "appointment", "apply")}

    def contradiction_questions(self, env, cands: list[Candidate], asked: set[str]) -> list[Question]:
        """Task module: a person's turn that states a value and negates it in the same breath ('X works for me, but I
        cannot do X') is a genuine ambiguity the samples do not expose; surface it as a disagreement question."""
        out = []
        need_ids = sorted({n["need_id"] for c in cands for n in c.artifact["needs"]})
        names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                 for c in cands for n in c.artifact["needs"]}
        for m in env.messages:
            if m["role"] != "user":
                continue
            for sent in re.split(r"(?<=[.!?])\s+", m["content"]):
                if not self.CONTRA_RE.search(sent):
                    continue
                low = sent.lower()
                fields = [f for f, words in self.FIELD_WORDS.items() if any(w in low for w in words)]
                if not fields:
                    continue
                f = fields[0]
                targets = [nid for nid in need_ids if names[nid].lower().split()[0] in low] or need_ids[:1]
                for nid in targets:
                    qid = f"{nid}:{f}:contradiction"
                    if qid in asked or f"{nid}:{f}" in asked or any(q.qid == qid for q in out):
                        continue
                    greedy = next((c for c in cands if c.greedy), cands[0])
                    label = R.FIELD_LABEL.get(f, f)
                    text = (f"For the {names[nid]} need, the person said something conflicting about their {label} (\"{sent.strip()[:160]}\"). "
                            f"Which {label} actually applies, or do they have no requirement?")
                    out.append(Question(qid, "intake", {"need_id": nid, "field": f}, greedy.key, "__residual__", text, tier=0))
        return out

    # ----- ordinary intake rule -----
    def ask_user_for_missing(self, env, artifact: dict) -> bool:
        missing = [(n, f) for (n, f) in R.intake_missing(artifact) if f not in self.skip_fields]
        if not missing:
            return False
        first_field = missing[0][1]
        group = next((g for g in self.intake_groups if first_field in g), (first_field,))
        targets = [(n, f) for (n, f) in missing if f in group]
        names = {}
        for n in artifact["needs"]:
            cats = n["service_categories"]
            names[n["need_id"]] = (cats[0] if isinstance(cats, list) and cats else "the service need")
        self._ask_missing_targets(env, names, targets)
        return True

    def _ask_missing_targets(self, env, names: dict, targets: list[tuple[str, str]], kind_detail: str = "ordinary_intake") -> str:
        fields_text = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in targets}))
        needs_text = " and ".join(sorted({names.get(n, n) for n, _ in targets}))
        prompt = [{"role": "system", "content": self.native_prompt},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                   f"\n\nWrite the next assistant message only (no tool call): ask the person, in one short low-burden message, "
                   f"about their {fields_text} for {needs_text}. Ask only these facts; do not repeat already answered facts. "
                   "Do not list example options or suggest values; ask whether they have a requirement or no preference. "
                   "If there are two needs, ask them to answer for each need separately."}]
        text = self.greedy_text(prompt, purpose="intake_question")
        wanted = {R.FIELD_LABEL.get(f, f) for _, f in targets}
        keyword = {"location": ("where", "location", "city", "county", "zip", "area"), "schedule": ("schedule", "day", "time", "when"),
                   "intake method": ("intake", "apply", "contact", "reach", "method"), "available documents": ("document",),
                   "eligibility": ("eligib", "qualif", "requirement")}
        for label in sorted(wanted):
            if not any(k in text.lower() for k in keyword.get(label, (label,))):
                text = text.rstrip() + f" Also, do you have any {label} requirements, or no preference?"
        reply = env.ask_user(text)
        for n, f in targets:
            self.asked.add((n, f))
            self.asked_count[(n, f)] += 1
        self.trace.add("user_question", kind_detail=kind_detail, fields=fields_text, text=text, reply=reply)
        return reply

    def diagnose_relaxation(self, env, programs: list[dict]) -> dict:
        """Counterfactual searches per need: which single relaxable constraint (location/schedule/intake), or pair,
        makes that need's program return resources. Deterministic; no agent turn consumed."""
        if getattr(self, "topology", "chain") == "fanout":
            from . import fanout2 as F2
            return F2.diagnose_relaxation(self, env, programs)
        import itertools
        ind, index = env.n["indiana211"], env.n["index"]
        fields = ("location", "schedule", "intake")
        keys = {"location": ("counties", "cities", "zipcodes"), "schedule": ("schedule",), "intake": ("intake_methods",)}
        report, per_need = {}, []
        for i, prog in enumerate(programs):
            found = []
            if ind.execute_search_resources(index, prog, limit=env.tool_limit).get("resources"):
                per_need.append([])
                continue
            for size in (1, 2, 3):
                for combo in itertools.combinations(fields, size):
                    args = copy.deepcopy(prog)
                    for f in combo:
                        for k in keys[f]:
                            args[k] = {} if k == "schedule" else []
                    n = len(ind.execute_search_resources(index, args, limit=env.tool_limit).get("resources") or [])
                    report[f"need-{i + 1}:{'+'.join(combo)}"] = n
                    if n > 0:
                        found.append("+".join(combo))
                if found:
                    break
            per_need.append(found)
        # union of the per-need minimal relaxations (every alternative, so the person can pick the one they can change)
        union = []
        for found in per_need:
            for combo in found:
                for f in combo.split("+"):
                    if f not in union:
                        union.append(f)
        minimal = ["+".join(union)] if union else []
        return {"report": report, "per_need": per_need, "minimal": minimal}

    FIELD_WORD = {"location": "location", "schedule": "schedule", "intake": "intake method", "documents": "available documents", "eligibility": "eligibility"}
    FIELD_KEY = {"intake": "intake_methods", "documents": "available_documents"}   # diagnosis word -> intake field

    def apply_overrides(self, packet: Packet, overrides: dict) -> None:
        """Deterministic repair: apply (need_id, field) -> value overrides to every intake interpretation in place."""
        spec = self.roles["intake"]
        merged: dict[str, Candidate] = {}
        for c in packet.candidates:
            if c.is_unresolved:
                merged[c.key] = c
                continue
            art = copy.deepcopy(c.artifact)
            changed = False
            for n in art["needs"]:
                for (need_id, f), action in overrides.items():
                    if need_id in (n["need_id"], "*") and action is not None and n.get(f) != action:
                        n[f] = action
                        changed = True
            if changed:
                key, norm = spec.canon(art)
                c.key, c.artifact, c.text = key, norm, spec.render(norm)
            if c.key in merged:
                m = merged[c.key]
                m.count += c.count
                m.weight += c.weight
                m.greedy = m.greedy or c.greedy
            else:
                merged[c.key] = c
        packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))
        packet.extra["overrides_applied"] = packet.extra.get("overrides_applied", []) + [f"{k[0]}|{k[1]}" for k in overrides]

    def fallback_round(self, env, inv: Invocation, empty: list[tuple[dict, dict]], diagnosis: dict | None, st: dict) -> bool:
        """One clarification round for every unmatched need at once: name each need and the constraints whose
        relaxation would make it match (from the counterfactual diagnosis), interpret the reply per need, and apply the
        resulting overrides to the intake packet deterministically. Returns True when some constraint changed."""
        names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                 for n in inv.packet.greedy.artifact["needs"]}
        asked = st.setdefault("fb_asked", {})
        parts, asked_ids, fields_asked = [], [], []
        for i, (p, _) in enumerate(empty):
            nid = p["need_id"]
            alts = (diagnosis["per_need"][i] if diagnosis and i < len(diagnosis.get("per_need", [])) else []) or ["location", "schedule", "intake"]
            fields = []
            for combo in alts:
                for f in combo.split("+"):
                    if f not in fields and f not in asked.get(nid, []):
                        fields.append(f)
            if not fields:
                continue
            asked.setdefault(nid, []).extend(fields)
            asked_ids.append(nid)
            fields_asked += fields
            parts.append(f"{names.get(nid, nid)} (your {' or '.join(self.FIELD_WORD[f] for f in fields)})")
        if not parts:
            return False
        text = ("I could not find any resources for " + "; ".join(parts) +
                ". Could any of those constraints be changed, and if so, to what? Please answer for each need.")
        reply = env.ask_user(text)
        self.last_fallback_request = ", ".join(sorted({self.FIELD_WORD[f] for f in fields_asked}))
        allowed = {(nid, self.FIELD_KEY.get(f, f)) for nid in asked_ids for f in asked.get(nid, [])}
        st["repair_rounds"] = st.get("repair_rounds", 0) + 1
        overrides = self.resolve_fallback_reply(env, text, reply, asked_ids, allowed)
        adjustable = [a for a in getattr(self, "_adjustable", []) if a not in overrides]
        self.trace.add("user_question", kind_detail="fallback_repair", request=text, text=text, reply=reply)
        self.trace.add("fallback_resolution", request=text, overrides={f"{k[0]}|{k[1]}": v for k, v in overrides.items()}, adjustable=adjustable)
        if adjustable and env.turns_left > 2:
            labels = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in adjustable}))
            text2 = (f"Thanks. Which specific {labels} would work for you instead (or is any value fine)?"
                     + (" Please answer for each need." if len(names) > 1 else ""))
            reply2 = env.ask_user(text2)
            more = self.resolve_fallback_reply(env, text2, reply2, asked_ids, set(adjustable))
            more = {k: v for k, v in more.items() if k in adjustable}
            for key in adjustable:
                if key not in more:
                    more[key] = R.NONE
            overrides.update(more)
            self.trace.add("user_question", kind_detail="fallback_followup", fields=labels, text=text2, reply=reply2)
            self.trace.add("fallback_resolution", request=labels, overrides={f"{k[0]}|{k[1]}": v for k, v in overrides.items()})
        for f in fields_asked:
            self.asked.add(("*", self.FIELD_KEY.get(f, f)))
        if overrides:
            self.apply_overrides(inv.packet, overrides)
            st["executed_programs"] = []
        return bool(overrides)

    def ask_user_fallback(self, env, request: str, need_name: str | None = None) -> None:
        # deterministic wording: name the need (composite cases) and ask only whether the constraint can change
        what = f"for {need_name}" if need_name else "matching your request"
        text = (f"I could not find any resources {what} with your current {request}. "
                f"Could your {request} {'for ' + need_name + ' ' if need_name else ''}be changed, and if so, to what?")
        reply = env.ask_user(text)
        self.last_fallback_request = request
        self.fallback_overrides = self.resolve_fallback_reply(env, text, reply)
        self.trace.add("fallback_resolution", request=request, overrides={f"{k[0]}|{k[1]}": v for k, v in self.fallback_overrides.items()},
                       adjustable=list(getattr(self, "_adjustable", [])))
        # "I can change it" without a value: ask which value works (ordinary repair), while turns allow search + final.
        adjustable = [a for a in getattr(self, "_adjustable", []) if a not in self.fallback_overrides]
        if adjustable and env.turns_left > 2:
            labels = ", ".join(sorted({R.FIELD_LABEL.get(f, f) for _, f in adjustable}))
            prompt2 = [{"role": "system", "content": self.native_prompt},
                       {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                        f"\n\nThe person said they can adjust their {labels}. Write the next assistant message only (no tool call): ask which specific {labels} "
                        "would work for them instead (or whether any value is fine), in one short message; answer per need if there are two needs."}]
            text2 = self.greedy_text(prompt2, purpose="fallback_followup")
            reply2 = env.ask_user(text2)
            more = self.resolve_fallback_reply(env, text2, reply2)
            more = {k: v for k, v in more.items() if k in adjustable}  # the follow-up concerns only the adjustable fields
            for key in adjustable:
                if key not in more and key in getattr(self, "_adjustable", []):
                    more[key] = R.NONE  # still vague after a direct follow-up: treat as no constraint
            self.fallback_overrides.update(more)
            self.trace.add("user_question", kind_detail="fallback_followup", fields=labels, text=text2, reply=reply2)
            self.trace.add("fallback_resolution", request=labels, overrides={f"{k[0]}|{k[1]}": v for k, v in self.fallback_overrides.items()})
        for f in ("location", "schedule", "intake_methods"):
            if f.split("_")[0] in request:
                self.asked.add(("*", f))
        self.trace.add("user_question", kind_detail="fallback_repair", request=request, text=text, reply=reply)

    def resolve_fallback_reply(self, env, question_text: str, reply: str, asked_ids: list[str] | None = None,
                               allowed: set | None = None) -> dict:
        """Ordinary repair: interpret the person's answer to a fallback question as per-need field actions.
        `allowed` = {(need_id, field)} that this question put on the table; other fields keep their values
        (a person saying 'I still need Thursday morning' must not have that window re-parsed)."""
        needs = [n for inv in reversed(self.invocations) if inv.role == "intake" for n in inv.packet.candidates[inv.selected].artifact["needs"]][:2]
        if asked_ids:
            needs = [n for n in needs if n["need_id"] in asked_ids] or needs
        need_desc = "; ".join(f"{n['need_id']} = {n['service_categories']}" for n in needs) or "need-1"
        fb_fields = self.fallback_norm_fields
        extra_fmt = ("; available_documents array from [birth_certificate, insurance_card, lease, photo_id, proof_of_address, proof_of_income, social_security, utility_bill]; "
                     "eligibility array from [disability, family, homeless, low_income, medicaid, pregnant, resident, senior, uninsured, veteran, youth]") if len(fb_fields) > 3 else ""
        example = "{" + ", ".join(f"\"{f}\": ..." for f in fb_fields) + "}"
        prompt = [{"role": "system", "content": "You interpret a person's reply to a resource-search agent that asked whether a constraint could be relaxed after an empty search. "
                   f"For each need and each of the fields {', '.join(fb_fields)} decide: \"keep\" (not mentioned or cannot change), \"none\" (the person explicitly says any value is fine / they have no preference / no longer need that constraint), "
                   "\"adjustable\" (the person says they can change or adjust it but does not name a new specific value), or a new concrete value in the search format "
                   "(location {\"counties\":[...]}/{\"cities\":[...]}/{\"zipcodes\":[...]}; schedule {\"day\":\"mon\"..\"sun\",\"time\"/\"start_time\"/\"end_time\":\"HH:MM\"} or {\"requires_24_hours\":true}; "
                   f"intake_methods array from [appointment, call, email, mail, online, text, walk_in]{extra_fmt}). A statement about one need does not apply to the other need. "
                   f"Output ONLY compact JSON: {{\"need-1\": {example}, \"need-2\": {{...}}}}"},
                  {"role": "user", "content": f"Needs: {need_desc}\nAgent asked: {question_text}\nPerson replied: {reply}"}]
        raw = self.greedy_text(prompt, purpose="fallback_resolve")
        obj = R.parse_json_object(raw) or {}
        self.trace.add("fallback_resolve_raw", raw=raw[:600], reply=reply[:400])
        overrides, adjustable = {}, []
        ranges = R.time_ranges(reply)
        lex = {"location": ("location", "county", "city", "zip", "area", "town", "place", "move", "travel", "elsewhere", "nearby"),
               "schedule": ("schedule", "day", "time", "hour", "week", "morning", "afternoon", "evening", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "24"),
               "intake_methods": ("intake", "method", "call", "phone", "email", "mail", "online", "walk", "person", "appointment", "text", "apply", "contact"),
               "available_documents": ("document", "paperwork", "photo id", "proof", "certificate", "lease", "insurance", "social security", "utility bill"),
               "eligibility": ("eligib", "qualif", "veteran", "senior", "income", "disab", "homeless", "pregnan", "medicaid", "uninsured", "resident", "youth", "family")}
        mentioned = {f for f, words in lex.items() if any(w in (question_text + " " + reply).lower() for w in words)}
        for n in needs:
            entry = obj.get(n["need_id"]) if isinstance(obj.get(n["need_id"]), dict) else {}
            for f in fb_fields:
                action = entry.get(f, "keep")
                tok = re.compile(r"^\W*(keep|none|adjustable|flexible|can change|any)\W*$", re.I)
                if isinstance(action, list):
                    # malformed/control tokens inside a value list ("adjustable", "keep}}") are not values
                    flags = [str(x).strip().lower() for x in action if isinstance(x, str) and tok.match(str(x))]
                    action = [x for x in action if not (isinstance(x, str) and tok.match(str(x)))]
                    if not action:
                        action = "adjustable" if any(fl.startswith(("adjust", "flex", "can")) for fl in flags) else ("none" if any(fl.startswith(("none", "any")) for fl in flags) else "keep")
                if isinstance(action, str) and action.strip().lower().strip("}\"' ") == "keep":
                    continue
                if f not in mentioned:
                    continue  # a field neither asked about nor mentioned cannot change
                if allowed is not None and (n["need_id"], f) not in allowed:
                    continue  # not on the table in this question
                flex = re.search(r"\b(or something else|or anything else|anything else (works|is fine)|whatever works|any (method|way|location|time|day) (works|is fine)|or whatever|doesn.t matter|don.t mind)\b", reply.lower())
                if flex and f in ("intake_methods", "location", "schedule") and f in mentioned:
                    overrides[(n["need_id"], f)] = R.NONE
                    continue
                if f == "location":
                    lit = self.recover_location(reply)
                    if lit and lit != n.get("location"):
                        overrides[(n["need_id"], f)] = lit   # a literal in the person's words beats the model's reading
                        continue
                if isinstance(action, str) and action.strip().lower().strip("}\"' ") in ("adjustable", "can change", "flexible"):
                    adjustable.append((n["need_id"], f))
                    continue
                norm = R._norm_field(f, action)
                if norm in (R.UNKNOWN, R.EMPTY):
                    continue
                if f == "location" and isinstance(norm, dict) and norm.get("counties"):
                    norm["counties"] = [c.upper().replace(" COUNTY", "").strip() for c in norm["counties"]]
                if f == "location" and isinstance(norm, dict):
                    # every location literal must appear in the person's reply (no ZIP/city/county from model knowledge)
                    low = reply.lower()
                    grounded = {k: [v for v in vals if re.search(r"\b" + re.escape(str(v).lower()) + r"\b", low)] for k, vals in norm.items()}
                    grounded = {k: v for k, v in grounded.items() if v}
                    if grounded != norm:
                        self.trace.add("fallback_location_ungrounded", need_id=n["need_id"], proposed=norm, kept=grounded)
                        if not grounded:
                            adjustable.append((n["need_id"], f))   # the person said they can move but named nothing usable: ask which place
                            continue
                        norm = grounded
                if f == "schedule" and isinstance(norm, dict) and norm.get("day"):
                    days = [d for d, pats in self.VALUE_LITERALS["day"].items() if any(re.search(p, reply.lower()) for p in pats)]
                    if norm["day"] not in days:
                        if len(days) == 1:
                            norm["day"] = days[0]          # the model misread the weekday; the person named exactly one
                        else:
                            norm = {k: v for k, v in norm.items() if k != "day"}
                            if not norm:
                                continue
                if f == "schedule" and isinstance(norm, dict):
                    # a single time that starts a stated window becomes that window
                    if "time" in norm:
                        for a, b in ranges:
                            if norm["time"] == a:
                                norm = {k: v for k, v in norm.items() if k != "time"} | {"start_time": a, "end_time": b}
                                break
                    # time literals must be traceable to the person's words; repair a window from the reply when possible
                    times = [norm.get(k) for k in ("time", "start_time", "end_time") if norm.get(k)]
                    if any(not R.time_supported(t, reply) for t in times):
                        if len(ranges) == 1 and ("start_time" in norm or "end_time" in norm or "time" in norm):
                            norm = {k: v for k, v in norm.items() if k not in ("time", "start_time", "end_time")} | {"start_time": ranges[0][0], "end_time": ranges[0][1]}
                        else:
                            norm = {k: v for k, v in norm.items() if k not in ("time", "start_time", "end_time")}
                            if not norm:
                                continue
                overrides[(n["need_id"], f)] = norm
        # 'nearby / anywhere / any location' without a new literal = no location constraint
        if "location" in mentioned and re.search(r"\b(nearby|anywhere|any (location|area|place|county|city|where)|surrounding|other (areas|counties|places|cities)|wherever)\b", reply.lower()):
            new_lit = self.recover_location(reply)
            for n in needs:
                key = (n["need_id"], "location")
                if key not in overrides or overrides[key] == R.NONE or new_lit is None or overrides[key] == new_lit:
                    prev = n.get("location")
                    if new_lit is None or new_lit == prev:
                        overrides[key] = R.NONE
                        adjustable[:] = [a for a in adjustable if a != key]
        self._adjustable = adjustable
        return overrides

    # ----- sender answers a clarification (source-backed) -----
    def intake_answer(self, env, question: Question) -> dict:
        two_valued = question.b != "__residual__"
        schema = ("{\"answer\": \"...\", \"resolved\": true|false, \"choice\": \"A\"|\"B\"|\"neither\", \"sources\": [person turn indices]}"
                  if two_valued else "{\"answer\": \"...\", \"resolved\": true|false, \"sources\": [person turn indices]}")
        extra = (" The question offers two values: A is the first value stated, B the second; set choice to the one the person's words "
                 "support, or \"neither\"." if two_valued else "")
        prompt = [{"role": "system", "content": "You are the intake agent. Answer the planner's clarification strictly from what the person said. "
                   f"Output ONLY compact JSON: {schema}. Set resolved=false if the conversation does not settle it.{extra}"},
                  {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) + "\n\nQuestion: " + question.text}]
        obj = R.parse_json_object(self.greedy_text(prompt, purpose="intake_answer")) or {}
        choice = str(obj.get("choice") or "").strip().upper()[:1] if two_valued else ""
        return {"question": question.text, "qid": question.qid, "answer": str(obj.get("answer") or ""),
                "resolved": bool(obj.get("resolved")), "sources": obj.get("sources") or [], "choice": choice if choice in ("A", "B") else ""}

    def prune_intake_packet(self, packet: Packet, question: Question, choice: str) -> int:
        """Candidate-indexed clarification: keep only the interpretations that agree with the chosen value on the asked
        (need, field); support mass of the kept side is renormalized to the mass the two sides had. Returns kept count
        (0 = nothing to do, caller falls back to re-proposal)."""
        chosen_key = question.a if choice == "A" else question.b
        ref = next((c for c in packet.candidates if c.key == chosen_key and not c.is_unresolved), None)
        if ref is None:
            return 0
        need_id, f = question.scope.get("need_id"), question.scope.get("field")
        need = next((n for n in ref.artifact["needs"] if n["need_id"] == need_id), None)
        if need is None:
            return 0
        target = json.dumps(need.get(f), sort_keys=True)
        keep, drop = [], []
        for c in packet.candidates:
            if c.is_unresolved:
                keep.append(c)
                continue
            n = next((x for x in c.artifact["needs"] if x["need_id"] == need_id), None)
            (keep if n is not None and json.dumps(n.get(f), sort_keys=True) == target else drop).append(c)
        if not drop:
            return len(keep)
        total = sum(c.weight for c in packet.candidates if not c.is_unresolved)
        kept_mass = sum(c.weight for c in keep if not c.is_unresolved)
        for c in keep:
            if not c.is_unresolved and kept_mass > 0:
                c.weight = c.weight * total / kept_mass
        if not any(c.greedy for c in keep):
            best = max((c for c in keep if not c.is_unresolved), key=lambda c: (c.count, c.weight))
            best.greedy = True
        packet.candidates = keep
        packet.extra["pruned_by"] = packet.extra.get("pruned_by", []) + [question.qid]
        return len(keep)

    def intake_ask_user_for(self, env, question: Question) -> str:
        if question.scope.get("pairs") is not None:   # group question: the ordinary low-burden intake message for that group
            intake = next((i for i in reversed(self.invocations) if i.role == "intake"), None)
            g = intake.packet.greedy.artifact if intake is not None else {"needs": []}
            names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                     for n in g.get("needs", [])}
            return self._ask_missing_targets(env, names, [tuple(p) for p in question.scope["pairs"]], kind_detail="budget_intake")
        if question.qid.endswith(":contradiction"):
            # deterministic wording: point at the conflicting statement and ask for the actual fact
            m = re.search(r'\("(.*?)"\)\.', question.text)
            quoted = m.group(1) if m else ""
            label = R.FIELD_LABEL.get(question.scope.get("field", ""), question.scope.get("field", ""))
            m2 = re.search(r"For the (.+?) need,", question.text)
            need = m2.group(1) if m2 else ""
            text = (f"I want to clarify and confirm your {label}{' for ' + need if need else ''}: "
                    f"what {label} can you actually use, or do you have no requirement?")
        else:
            prompt = [{"role": "system", "content": self.native_prompt},
                      {"role": "user", "content": "Conversation so far:\n" + self._conversation_text(env) +
                       f"\n\nWrite the next assistant message only (no tool call): ask the person one short, specific question to settle this: {question.text} "
                       "Do not suggest values; ask whether they have a requirement or no preference."}]
            text = self.greedy_text(prompt, purpose="clarify_user")
        reply = env.ask_user(text)
        self.asked.add((question.scope.get("need_id", "*"), question.scope.get("field", "")))
        self.trace.add("user_question", kind_detail="clarification", qid=question.qid, text=text, reply=reply)
        return reply

    def _pi0_pick(self, packet: Packet) -> Candidate:
        cands = [c for c in packet.candidates if not c.is_unresolved]
        w = [c.count for c in cands]
        return self.rng.choices(cands, weights=w, k=1)[0] if sum(w) else packet.greedy

    # ----- episode -----
    def run(self, env, *, max_cycles: int = 12, resume: dict | None = None) -> dict:
        if resume is None:
            opening = env.reset()
            st = {"replies": [], "asked": set(), "executed_programs": [], "fallback_fields_asked": [], "found": {}}
        else:
            opening = env.messages[0]["content"]
            st = resume["st"]
        cycles = 0
        outcome = {"terminated": "unresolved"}
        pruned = None   # intake invocation whose candidate set an evidence-resolved answer just pruned (no re-proposal)
        while cycles < max_cycles and not env.done:
            cycles += 1
            self.calls_step = 0
            if resume is not None and cycles == 1:
                inv, pinv, menu = resume["inv"], resume["pinv"], resume["menu"]
                selected_intake = inv.packet.candidates[inv.selected]
            else:
                if pruned is not None:
                    inv, pruned = pruned, None
                    selected_intake = inv.packet.greedy
                    inv.selected = inv.packet.candidates.index(selected_intake)
                    self.trace.add("intake_checked", inv_id=inv.inv_id, packet=inv.packet.public(), pruned=True)
                else:
                    # --- sender: intake -> {access, schedule, logistics} receivers -> joint packet (fanout2.sender_invocation) ---
                    inv = self.sender_invocation(env)
                    self.last_fallback_request = None
                    self.fallback_overrides = {}
                    selected_intake = self._pi0_pick(inv.packet) if isinstance(self.controller, Pi0Controller) else inv.packet.greedy
                    inv.selected = inv.packet.candidates.index(selected_intake)
                    # ordinary rule: collect missing facts before searching (reserve 2 turns for search + final)
                    if not self.budget_questions and R.intake_missing(selected_intake.artifact) and env.turns_left > 2:
                        if self.ask_user_for_missing(env, selected_intake.artifact):
                            continue
                    if env.turns_left < 2:
                        break
                # --- receiver: planner ---
                if self.llm_planner:
                    pinv = self.invoke("planner", self.planner_messages(opening, inv.packet, st["replies"]), tools=self.roles["planner"].tools,
                                       ancestors=[inv.inv_id], unresolved_text="UNRESOLVED: the evidence does not support a search program yet.", purpose="planner")
                    self.check_planner_packet(inv.packet, pinv.packet)
                else:
                    version = self.next_version("planner")
                    pkt = self.planner_packet_from_intake(inv.packet, version)
                    if not any(not c.is_unresolved for c in pkt.candidates):
                        # no fully resolved interpretation: fall back to the backbone planner
                        pinv = self.invoke("planner", self.planner_messages(opening, inv.packet, st["replies"]), tools=self.roles["planner"].tools,
                                           ancestors=[inv.inv_id], unresolved_text="UNRESOLVED: the evidence does not support a search program yet.", purpose="planner")
                        self.check_planner_packet(inv.packet, pinv.packet)
                    else:
                        pinv = Invocation(f"planner#{version}", "planner", version, self.planner_messages(opening, inv.packet, st["replies"]),
                                          self.roles["planner"].tools, pkt, 0, "planner_deterministic")
                        for c in pkt.candidates:   # representative text of a deterministic candidate is its program text
                            c.rep_text = c.text if not c.is_unresolved else ""
                        self.invocations.append(pinv)
                        self.trace.add("invocation", inv_id=pinv.inv_id, role="planner", version=version, purpose="planner_deterministic",
                                       packet=pkt.public(), n_samples=0, invalid=0, invalid_examples=[], finish_reasons=[])
                self.trace.add("planner_checked", inv_id=pinv.inv_id, packet=pinv.packet.public())
                menu = self.planner_menu(inv.packet, st["asked"], env) if env.turns_left > 2 else []
            ctx = ReceiverContext("planner", [inv.packet], pinv.packet.candidates, menu,
                                  self.items_budget - self.items_used, env.turns_left, opening)
            ctx.aux = self.planner_aux(env, inv.packet, menu)
            snapshot = {"inv": inv, "pinv": pinv, "menu": menu, "st": copy.deepcopy(st), "sender_invs": [inv]}
            if resume is not None and cycles == 1:
                d = resume["forced"]
                self.trace.add("decision", role="planner", decision=d.kind, candidate=d.candidate, forced=True,
                               question=None if d.question is None else d.question.__dict__, menu=[q.__dict__ for q in menu])
                self.controller = ForcedController(d, self.seed, getattr(self.controller, "inner", None))   # keep the branch continuation policy
                self.controller.used = True
            else:
                d = self.decide(ctx, pinv, snapshot)
            if d.kind == "ask" and d.question is not None:
                self.items_used += d.question.items
                st["asked"].add(d.question.qid)
                if d.question.qid.endswith(":contradiction"):
                    st["asked"].add(d.question.qid.rsplit(":", 1)[0])
                if self.user_direct and d.question.tier in (0, 1) and env.turns_left > 2:
                    ans = {"question": d.question.text, "qid": d.question.qid, "answer": "", "resolved": True, "sources": []}
                    ans["user_reply"] = self.intake_ask_user_for(env, d.question)
                else:
                    ans = self.intake_answer(env, d.question)
                    if not ans["resolved"] and env.turns_left > 2:
                        ans["user_reply"] = self.intake_ask_user_for(env, d.question)
                        ans["resolved"] = True
                st["replies"].append(ans)
                self.trace.add("clarification", **ans)
                if self.prune_on_answer and "user_reply" not in ans:
                    # evidence-only answer: no new information entered the conversation, so do not re-sample the intake
                    # (re-proposal noise); prune the candidate set when the answer picks a side, else just re-decide
                    kept = self.prune_intake_packet(inv.packet, d.question, ans["choice"]) if ans.get("resolved") and ans.get("choice") in ("A", "B") else 0
                    pruned = inv
                    self.trace.add("candidate_pruned", qid=d.question.qid, choice=ans.get("choice", ""), kept=kept)
                continue  # intake re-proposes with the user's reply (new evidence), or re-decides on the same/pruned packet
            chosen = pinv.packet.candidates[d.candidate]
            pinv.selected = d.candidate
            if chosen.is_unresolved:
                if R.intake_missing(selected_intake.artifact) and env.turns_left > 2:
                    self.ask_user_for_missing(env, selected_intake.artifact)
                    continue
                break
            progs = chosen.artifact["programs"]
            found = st.setdefault("found", {})          # need_id -> {"arguments", "result"} for needs already matched
            todo = [p for p in progs if p["need_id"] not in found]
            names = {n["need_id"]: (n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else n["need_id"])
                     for n in selected_intake.artifact["needs"]}
            if todo:
                programs = [p["arguments"] for p in todo]
                prog_key = json.dumps(programs, sort_keys=True)
                if prog_key in st["executed_programs"]:
                    break
                st["executed_programs"].append(prog_key)
                results_new = env.search(programs)
                if env.done:
                    break
                for p, r in zip(todo, results_new):
                    if r.get("resources"):
                        found[p["need_id"]] = {"arguments": p["arguments"], "result": r}
                empty = [(p, r) for p, r in zip(todo, results_new) if not r.get("resources")]
                diagnosis = self.diagnose_relaxation(env, [p["arguments"] for p, _ in empty]) if (self.use_relaxation and empty) else None
                if diagnosis:
                    diagnosis["need_ids"] = [p["need_id"] for p, _ in empty]
                    self.trace.add("relaxation_diagnosis", **diagnosis)
                # unmatched needs: clarification rounds (one question covering every unmatched need) until something
                # changes or no relaxation is left; the repaired packet is re-planned without re-sampling the intake
                changed = False
                while (empty and env.turns_left > 2 and not changed and self.fallback_clarification and getattr(self, "combined_fallback", True)
                       and st.get("repair_rounds", 0) < (self.max_repair_rounds or 10 ** 6)):
                    if not self.fallback_round(env, inv, empty, diagnosis, st):
                        break
                    changed = True
                if changed:
                    pruned = inv
                    continue
                results = [found[p["need_id"]]["result"] if p["need_id"] in found else {"resources": []} for p in progs]
            else:
                results = [found[p["need_id"]]["result"] for p in progs]
                diagnosis = None
            # --- receiver: retriever ---
            rinv = self.invoke("retriever", self.retriever_messages(opening, pinv.packet, chosen, results, diagnosis), tools=self.roles["retriever"].tools,
                               ancestors=[pinv.inv_id], purpose="retriever")
            rctx = ReceiverContext("retriever", [pinv.packet], rinv.packet.candidates, [], 0, env.turns_left, opening)
            rd = self.controller.decide(rctx)
            self.trace.add("decision", role="retriever", decision=rd.kind, candidate=rd.candidate, menu=[])
            rchosen = rinv.packet.candidates[rd.candidate]
            rinv.selected = rd.candidate
            if rchosen.artifact.get("empty") and any(r.get("resources") for r in results) and (env.turns_left <= 2 or st.get("fb_asked")):
                ids = [r["resources"][0].get("resource_id") or r["resources"][0].get("id") for r in results if r.get("resources") and isinstance(r["resources"][0], dict)]
                ids = [i for i in ids if i]
                if ids:
                    env.finish(ids, "Here are the resources I could find for your needs.")
                    outcome = {"terminated": "final_partial"}
                    break
            if rchosen.artifact.get("empty") or any(not r.get("resources") for r in results):
                if env.turns_left > 2 and not st.get("fb_asked") and self.fallback_clarification and st.get("repair_rounds", 0) < (self.max_repair_rounds or 10 ** 6):
                    st["repair_rounds"] = st.get("repair_rounds", 0) + 1
                    req = rchosen.artifact.get("fallback_request", "location, schedule or intake")
                    if diagnosis and diagnosis["minimal"]:
                        # ask about every constraint in the smallest relaxation set not yet asked; else the retriever's choice
                        best = next((m for m in diagnosis["minimal"] if not all(f in st["fallback_fields_asked"] for f in m.split("+"))), None)
                        if best:
                            req = " and ".join(self.FIELD_WORD.get(f, f) for f in best.split("+"))
                    for f in ("location", "schedule", "intake"):
                        if f in req:
                            st["fallback_fields_asked"].append(f)
                    first_empty = next((p["need_id"] for p, r in zip(progs, results) if not r.get("resources")), None)
                    self.ask_user_fallback(env, req, names.get(first_empty) if (len(progs) > 1 and first_empty) else None)
                    st["replies"], st["asked"] = [], set()
                    continue
                break
            # the final recommendation must cover every need: merge ids the retriever picked with the matched results it was given
            env.finish(rchosen.artifact["resource_ids"], rchosen.artifact.get("message", ""))
            outcome = {"terminated": "final"}
        return self.record_extra(env.record(), outcome, cycles)
