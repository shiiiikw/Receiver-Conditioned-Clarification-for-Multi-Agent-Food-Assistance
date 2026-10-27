"""Fan-out topology (paper Appendix G, "Receiver-Conditioned Clarification for Multi-Agent Food Assistance"):

    intake -> three parallel receivers {access, schedule, logistics} -> deterministic joint planner -> retriever

* The intake agent reads the complete dialogue and produces candidate interpretations with premise identities
  ((need_id, field) pairs).  (G.1)
* Each receiver reads ONLY the intake candidates projected to its own fields plus domain evidence from the index
  (county names, index cities matching the stated literals, the search schedule format, the field vocabularies);
  it never sees the dialogue.  It emits, per intake candidate, its domain decision (the normalized search value of
  each of its fields) and its obligations (facts still required before that decision is supported).  Decisions
  are validated against hard constraints (no value may be invented or dropped relative to the intake candidate;
  literals are normalized to index names), so the admissible decision A_adm is deterministic given the candidate
  while the obligations O_j carry the receiver's domain judgement.  (G.1, G.3)
* The joint interpretation set Omega is the product of the intake candidates and the receiver candidates,
  weighted by products of supports; at most OMEGA_MAX = 16 interpretations are retained and renormalized.  (G.2)
* Every question carries its dependency footprint R_t(q): the receivers reachable from the targeted premise.  (G.3)
* Counterfactual relaxation diagnosis evaluates adjustments in the fixed order time-window < location < intake
  method < document requirement < eligibility, at most 6 adjustments per step, minimal recovery first.  (G.4)
  (The index has no radius search, so "distance expansion" is realised as location relaxation.)
* Budgets: 60 backbone calls per step, 180 per episode; at most 2 repair rounds.  (G.7)

Everything here is deterministic code plus frozen-backbone invocations; nothing is learned.
"""
from __future__ import annotations

import copy
import json
import re

from typing import TYPE_CHECKING

from . import roles as R
from .common import canonical
from .packets import Packet, Candidate, UNRESOLVED_KEY

if TYPE_CHECKING:
    from .workflow import Invocation

RECEIVER_SCOPE = {"access": ("intake_methods", "available_documents", "eligibility"),
                  "schedule": ("schedule",),
                  "logistics": ("location",)}
RECEIVERS = ("access", "schedule", "logistics")
FIELD_OWNER = {f: r for r, fs in RECEIVER_SCOPE.items() for f in fs}
OMEGA_MAX = 16
S_DRAWS_FANOUT, K_MAX_FANOUT = 4, 3          # G.2: one greedy + four random samples, K = 3 candidates
MAX_CALLS_STEP, MAX_CALLS_EPISODE = 60, 180  # G.7
RECEIVER_MAX_TOKENS = 1536                   # receiver tool calls enumerate per-candidate decisions and need a longer output budget
MAX_REPAIR_ROUNDS = 2                        # G.7
MAX_ADJUSTMENTS, MAX_FALLBACK_QUESTIONS = 6, 3   # G.4
C_ITEM, C_FALLBACK, P_ACCEPT, VOI_DEPTH = 0.04, 0.06, 0.5, 2   # G.5, G.6
RELAXATION_ORDER = ("schedule_time", "schedule", "location", "intake", "documents", "eligibility")   # G.4 ladder

RECEIVER_TOOLS = [R.tool("report_domain_decision", "Report, for every intake candidate, your domain decision and obligations.", {
    "decisions": {"type": "array", "description": "One entry per intake candidate (all of them).", "items": {"type": "object", "properties": {
        "candidate": {"type": "integer", "description": "Candidate number as listed."},
        "needs": {"type": "array", "items": {"type": "object", "properties": {
            "need_id": {"type": "string", "enum": ["need-1", "need-2"]},
            "location": {"description": "\"unknown\" | \"none\" | {\"counties\": [...], \"cities\": [...], \"zipcodes\": [...]}"},
            "schedule": {"description": "\"unknown\" | \"none\" | {\"day\": ..., \"time\"/\"start_time\"/\"end_time\": \"HH:MM\"} | {\"requires_24_hours\": true}"},
            "intake_methods": {"description": "\"unknown\" | \"none\" | array"},
            "available_documents": {"description": "\"unknown\" | \"none\" | array"},
            "eligibility": {"description": "\"unknown\" | \"none\" | array"}}, "required": ["need_id"]}},
        "obligations": {"type": "array", "description": "Facts still required before your decision for this candidate is supported.",
                        "items": {"type": "object", "properties": {"need_id": {"type": "string"}, "field": {"type": "string"},
                                                                   "reason": {"type": "string"}}, "required": ["need_id", "field"]}}},
        "required": ["candidate", "needs", "obligations"]}}}, ["decisions"])]

_RECEIVER_COMMON = """
You do NOT see the conversation with the person. You receive the intake agent's numbered candidate interpretations of the person's need(s), restricted to your fields, with their empirical support, plus domain evidence from the Indiana 211 index. For EVERY candidate, call report_domain_decision exactly once with (a) the search value of each of your fields per need, normalized to the index vocabulary, and (b) your obligations: the facts still required before your decision for that candidate is supported by evidence.
Hard rules: a candidate value "unknown" stays "unknown" and is an obligation; "none" (no constraint) stays "none"; a concrete value keeps its meaning exactly: never drop it, never extend it, never replace it with a value the candidate does not state. Never infer a county from a city or ZIP code. Keep need ids as given. Fields that are not yours are omitted."""

RECEIVER_TASKS = {
    "logistics": """You are the LOGISTICS receiver (location and travel) in a resource-navigation team (intake -> access / schedule / logistics receivers -> joint planner -> retriever).
Your field: location = {"counties": [...]} and/or {"cities": [...]} and/or {"zipcodes": [...]}. Domain evidence below lists the index's county names and the index cities that match the candidates' city literals. Normalize county literals to index names ("Lake County" -> "LAKE"); keep cities and ZIP codes as stated. Obligations: location unknown; a stated literal that matches nothing in the index (the search would return nothing for it).""" + _RECEIVER_COMMON,
    "schedule": """You are the SCHEDULE receiver (opening and service hours) in a resource-navigation team (intake -> access / schedule / logistics receivers -> joint planner -> retriever).
Your field: schedule = {"day": "mon".."sun"} optionally with "time": "HH:MM" or "start_time"/"end_time", or {"requires_24_hours": true}. Domain evidence below gives the search format. Obligations: schedule unknown; a stated time that cannot be matched to opening hours (for example a time without a day).""" + _RECEIVER_COMMON,
    "access": """You are the ACCESS AND ELIGIBILITY receiver (eligibility, documents, intake method) in a resource-navigation team (intake -> access / schedule / logistics receivers -> joint planner -> retriever).
Your fields: intake_methods from [appointment, call, email, mail, online, text, walk_in]; available_documents from [birth_certificate, insurance_card, lease, photo_id, proof_of_address, proof_of_income, social_security, utility_bill]; eligibility from [disability, family, homeless, low_income, medicaid, pregnant, resident, senior, uninsured, veteran, youth]. Domain evidence below gives the vocabularies. Obligations: any of your fields unknown.""" + _RECEIVER_COMMON,
}

SCHEDULE_EVIDENCE = ("Search schedule format: {\"day\": one of mon,tue,wed,thu,fri,sat,sun} optionally with \"time\": \"HH:MM\" (24h) or a window "
                     "\"start_time\"/\"end_time\"; {\"requires_24_hours\": true} for round-the-clock services. A resource matches when it is open "
                     "on that day at that time / throughout that window.")
ACCESS_EVIDENCE = ("Index vocabularies. intake_methods: appointment, call, email, mail, online, text, walk_in. available_documents: birth_certificate, "
                   "insurance_card, lease, photo_id, proof_of_address, proof_of_income, social_security, utility_bill. eligibility: disability, family, "
                   "homeless, low_income, medicaid, pregnant, resident, senior, uninsured, veteran, youth. A resource matches a documents/eligibility filter "
                   "when it accepts those documents / serves that group; an intake_methods filter when it offers one of the listed methods.")


def parse_receiver(text: str):
    for name, args in R.parse_tool_calls(text):
        if name in ("report_domain_decision", "__bare__") and isinstance(args.get("decisions"), list) and args["decisions"]:
            ok = any(isinstance(d, dict) and str(d.get("candidate", "")).strip().lstrip("-").isdigit() for d in args["decisions"])
            if ok:
                return args
    return None


def make_receiver_canon(scope: tuple[str, ...]):
    def canon(obj: dict) -> tuple[str, dict]:
        decs: dict[int, dict] = {}
        for d in obj.get("decisions") or []:
            if not isinstance(d, dict):
                continue
            try:
                ci = int(str(d.get("candidate")).strip())
            except (TypeError, ValueError):
                continue
            needs: dict[str, dict] = {}
            for n in d.get("needs") or []:
                if not isinstance(n, dict) or not isinstance(n.get("need_id"), str):
                    continue
                needs[n["need_id"]] = {f: R._norm_field(f, n.get(f, R.UNKNOWN)) for f in scope}
            obl = set()
            for o in d.get("obligations") or []:
                if isinstance(o, dict) and o.get("field") in scope and isinstance(o.get("need_id"), str):
                    obl.add((o["need_id"], o["field"]))
            for nid, vals in needs.items():
                for f, v in vals.items():
                    if v in (R.UNKNOWN, R.EMPTY):
                        obl.add((nid, f))
            decs[ci] = {"needs": needs, "obligations": sorted(obl)}
        if not decs:
            decs = {1: {"needs": {}, "obligations": []}}
        norm = {"decisions": {str(k): v for k, v in sorted(decs.items())}}
        return canonical(norm), norm
    return canon


def render_receiver(a: dict) -> str:
    lines = []
    for ci, d in a["decisions"].items():
        parts = []
        for nid, vals in d["needs"].items():
            parts.append(f"{nid}: " + ", ".join(f"{f}={json.dumps(v) if not isinstance(v, str) else v}" for f, v in vals.items()))
        obl = ", ".join(f"{nid}:{f}" for nid, f in d["obligations"]) or "none"
        lines.append(f"candidate {ci}: " + ("; ".join(parts) or "no needs") + f" | obligations: {obl}")
    return "\n".join(lines)


def receiver_roles() -> dict[str, R.RoleSpec]:
    return {r: R.RoleSpec(r, RECEIVER_TASKS[r], parse_receiver, make_receiver_canon(RECEIVER_SCOPE[r]), render_receiver, RECEIVER_TOOLS,
                          max_tokens=RECEIVER_MAX_TOKENS)
            for r in RECEIVERS}


# ----- projections and hard constraints -----
def _county_norm(v: str) -> str:
    return re.sub(r"\s+COUNTY$", "", str(v).strip().upper())


def project_need(need: dict, scope: tuple[str, ...]) -> dict:
    out = {"need_id": need["need_id"], "service_categories": copy.deepcopy(need["service_categories"])}
    for f in scope:
        out[f] = copy.deepcopy(need[f])
    return out


def projection_key(intake_packet: Packet, scope: tuple[str, ...]) -> str:
    """Identity of a receiver's input: the resolved intake candidates projected to its fields (what it would read)."""
    rows = [[project_need(n, scope) for n in c.artifact["needs"]] for c in intake_packet.candidates if not c.is_unresolved]
    return canonical(rows)


def det_decision(value, field: str, known_counties: set[str]):
    """Deterministic admissible decision for one candidate value: index normalization without any change of meaning."""
    if value in (R.UNKNOWN, R.EMPTY):
        return R.UNKNOWN
    if value == R.NONE:
        return R.NONE
    if field == "location" and isinstance(value, dict):
        out = {}
        for k, vals in value.items():
            if k == "counties":
                fixed = [_county_norm(v) for v in vals]
                fixed = [v for v in fixed if v in known_counties] or fixed
                out[k] = sorted(set(fixed))
            else:
                out[k] = sorted(set(str(v).strip().upper() for v in vals))
        return out
    return copy.deepcopy(value)


def _same(a, b, field: str) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if field == "location":
            return {k: sorted(set(_county_norm(x) if k == "counties" else str(x).upper() for x in v)) for k, v in a.items()} == \
                   {k: sorted(set(_county_norm(x) if k == "counties" else str(x).upper() for x in v)) for k, v in b.items()}
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return sorted(set(map(str, a))) == sorted(set(map(str, b)))
    return a == b


def receiver_messages(wf, role: str, intake_packet: Packet) -> list[dict]:
    scope = RECEIVER_SCOPE[role]
    cands = [c for c in intake_packet.candidates if not c.is_unresolved]
    lines = []
    for i, c in enumerate(cands, 1):
        parts = []
        for n in c.artifact["needs"]:
            cat = n["service_categories"][0] if isinstance(n["service_categories"], list) and n["service_categories"] else "unknown category"
            vals = ", ".join(f"{f}={json.dumps(n[f]) if not isinstance(n[f], str) else n[f]}" for f in scope)
            parts.append(f"{n['need_id']} ({cat}): {vals}")
        lines.append(f"Candidate {i}{' [greedy]' if c.greedy else ''}, support {c.weight:.2f}: " + "; ".join(parts))
    if role == "logistics":
        lits = set()
        for c in cands:
            for n in c.artifact["needs"]:
                if isinstance(n["location"], dict):
                    for v in n["location"].get("cities", []):
                        lits.add(str(v).upper())
        matches = sorted({city for city in wf.known_cities if any(l in city or city in l for l in lits)})[:12]
        evidence = ("Index county names: " + ", ".join(sorted(wf.known_counties)) +
                    (". Index cities matching the stated city literals: " + ", ".join(matches) if lits else ". No city literal is stated.") +
                    (" (no index city matches: " + ", ".join(sorted(lits - set(matches))) + ")" if lits - set(matches) and lits else "")) if lits or True else ""
    elif role == "schedule":
        evidence = SCHEDULE_EVIDENCE
    else:
        evidence = ACCESS_EVIDENCE
    user = "Intake candidates (your fields only):\n" + "\n".join(lines) + "\n\nDomain evidence:\n" + evidence + \
           "\n\nReport your decision and obligations for every candidate (1.." + str(len(cands)) + ")."
    return [{"role": "system", "content": wf.roles[role].task}, {"role": "user", "content": user}]


def validate_receiver_packet(wf, role: str, packet: Packet, intake_packet: Packet) -> None:
    """Hard constraints (G.1 'external hard constraints'): every receiver decision equals the deterministic admissible
    decision of the intake candidate it addresses (no invention, no drop); missing candidates are filled deterministically.
    Obligations stated by the receiver for its own fields are kept (that is its domain judgement); unknown fields are always
    obligations. Equal decisions are coalesced (support summed)."""
    scope, spec = RECEIVER_SCOPE[role], wf.roles[role]
    cands = [c for c in intake_packet.candidates if not c.is_unresolved]
    merged: dict[str, Candidate] = {}
    for rc in packet.candidates:
        if rc.is_unresolved:
            merged[rc.key] = rc
            continue
        corrected, decisions = 0, {}
        for i, ic in enumerate(cands, 1):
            given = rc.artifact["decisions"].get(str(i), {"needs": {}, "obligations": []})
            needs, obl = {}, set()
            for n in ic.artifact["needs"]:
                vals = {}
                for f in scope:
                    want = det_decision(n[f], f, wf.known_counties)
                    got = given["needs"].get(n["need_id"], {}).get(f, R.UNKNOWN)
                    if not _same(got, want, f):
                        corrected += 1
                    vals[f] = want
                    if want == R.UNKNOWN:
                        obl.add((n["need_id"], f))
                needs[n["need_id"]] = vals
            for nid, f in given.get("obligations", []):
                if nid in needs and f in scope:
                    obl.add((nid, f))
            decisions[str(i)] = {"needs": needs, "obligations": sorted(obl)}
        norm = {"decisions": decisions}
        rc.key, rc.artifact, rc.text = canonical(norm), norm, spec.render(norm)
        if corrected:
            packet.extra["corrected"] = packet.extra.get("corrected", 0) + corrected
        if rc.key in merged:
            m = merged[rc.key]
            m.count += rc.count
            m.weight += rc.weight
            m.greedy = m.greedy or rc.greedy
            if rc.rep_logprob > m.rep_logprob:
                m.rep_text, m.rep_logprob = rc.rep_text, rc.rep_logprob
        else:
            merged[rc.key] = rc
    packet.candidates = sorted(merged.values(), key=lambda c: (c.is_unresolved, not c.greedy, -c.count))


def signature_of(decision: dict) -> tuple:
    """Receiver decision signature sigma_j(omega) = (A_adm, O_j) for one intake candidate."""
    return (canonical(decision["needs"]), tuple(decision["obligations"]))


def joint_invocation(wf, inv, parts: dict) -> "Invocation":
    """Deterministic joint planner input (G.2): Omega = intake candidates x receiver candidates, support = product, at most
    OMEGA_MAX kept and renormalized; each joint interpretation is a complete needs structure whose fields are the receivers'
    admissible decisions; obligations are the union over receivers. The downstream menu / VOI / planner mapping read it
    exactly like an intake packet."""
    import itertools
    from .workflow import Invocation
    spec = wf.roles["intake"]
    version = wf.next_version("joint")
    icands = [c for c in inv.packet.candidates if not c.is_unresolved]
    rc = {r: [c for c in parts[r].packet.candidates if not c.is_unresolved] for r in RECEIVERS}
    groups: dict[str, dict] = {}
    for i, ci in enumerate(icands, 1):
        for combo in itertools.product(*(rc[r] for r in RECEIVERS)):
            needs, obl, sigs = [], set(), {}
            for n in ci.artifact["needs"]:
                m = copy.deepcopy(n)
                for r, c in zip(RECEIVERS, combo):
                    d = c.artifact["decisions"].get(str(i), {"needs": {}, "obligations": []})
                    for f in RECEIVER_SCOPE[r]:
                        m[f] = copy.deepcopy(d["needs"].get(n["need_id"], {}).get(f, R.UNKNOWN))
                    obl |= {(nid, f) for nid, f in d["obligations"] if nid == n["need_id"]}
                    sigs[r] = signature_of(d)
                needs.append(m)
            key, norm = spec.canon({"needs": needs, "sources": ci.artifact.get("sources", [])})
            w = ci.weight
            for c in combo:
                w *= c.weight
            g = groups.setdefault(key, {"artifact": norm, "weight": 0.0, "count": 0, "greedy": False, "sources": set(), "src": ci,
                                        "obligations": set(), "sigs": {}})
            g["weight"] += w
            g["count"] = max(g["count"], min([ci.count] + [c.count for c in combo]))
            g["greedy"] = g["greedy"] or (ci.greedy and all(c.greedy for c in combo))
            g["sources"] |= set(ci.sources)
            g["obligations"] |= obl
            g["sigs"] = sigs
    ordered = sorted(groups.items(), key=lambda kv: (not kv[1]["greedy"], -kv[1]["weight"], kv[0]))
    kept = ordered[:wf.omega_max]
    total = sum(g["weight"] for _, g in kept) or 1.0
    cands = [Candidate(key=k, artifact=g["artifact"], text=spec.render(g["artifact"]), count=g["count"], weight=g["weight"] / total, greedy=g["greedy"],
                       rep_text=g["src"].rep_text, rep_logprob=g["src"].rep_logprob, sources=sorted(g["sources"])) for k, g in kept]
    if cands and not any(c.greedy for c in cands):
        cands[0].greedy = True
    if not cands:
        cands = [c for c in inv.packet.candidates if c.is_unresolved] or [Candidate(key=UNRESOLVED_KEY, artifact={"unresolved": True},
                 text="UNRESOLVED: no complete interpretation.", is_unresolved=True)]
    pkt = Packet(origin=f"joint#{version}", version=version, sender_role="joint", candidates=cands, residual_weight=0.0,
                 residual_count=inv.packet.residual_count + sum(p.packet.residual_count for p in parts.values()),
                 invalid_count=inv.packet.invalid_count + sum(p.packet.invalid_count for p in parts.values()),
                 n_stochastic=inv.packet.n_stochastic, ancestors=[inv.inv_id] + [p.inv_id for p in parts.values()], sources=list(inv.packet.sources))
    pkt.extra["topology"] = "fanout"
    pkt.extra["fanout_rev"] = 2
    pkt.extra["omega_total"] = len(ordered)
    pkt.extra["omega_dropped_mass"] = round(sum(g["weight"] for _, g in ordered[wf.omega_max:]), 4)
    pkt.extra["part_sizes"] = {"intake": len(icands), **{r: len(rc[r]) for r in RECEIVERS}}
    pkt.extra["obligations"] = {k: sorted(f"{nid}:{f}" for nid, f in g["obligations"]) for k, g in kept}
    pkt.extra["receiver_signatures"] = {k: {r: list(s) for r, s in g["sigs"].items()} for k, g in kept}
    pkt.extra["consumed_versions"] = {r: parts[r].version for r in RECEIVERS} | {"intake": inv.version}
    jinv = Invocation(f"joint#{version}", "joint", version, [], None, pkt, 0, "joint_deterministic")
    wf.invocations.append(jinv)
    wf.trace.add("invocation", inv_id=jinv.inv_id, role="joint", version=version, purpose="joint_deterministic", packet=pkt.public(),
                 n_samples=0, invalid=0, invalid_examples=[], finish_reasons=[])
    wf.trace.add("joint_checked", inv_id=jinv.inv_id, packet=pkt.public())
    return jinv


def sender_invocation(wf, env):
    """Sender side of one cycle under the fan-out topology (G.1 / G.7 revision rules): the intake agent re-reads the
    full dialogue; a receiver re-runs only when the intake candidates projected to its fields changed since its last
    invocation (otherwise its previous versioned packet stays valid: 'independent evidence and unrelated artifacts remain
    available'); the joint planner always recomputes."""
    inv = wf.invoke("intake", wf.intake_messages(env), tools=wf.roles["intake"].tools,
                    sources=list(range(1, len(env.user_replies) + 2)), purpose="intake")
    wf.check_intake_packet(env, inv.packet)
    wf.trace.add("intake_checked", inv_id=inv.inv_id, packet=inv.packet.public())
    parts = {}
    for role in RECEIVERS:
        key = projection_key(inv.packet, RECEIVER_SCOPE[role])
        cached = wf.receiver_cache.get(role)
        if cached is not None and cached[0] == key:
            parts[role] = cached[1]
            wf.trace.add("receiver_reused", role=role, inv_id=cached[1].inv_id, consumed=inv.inv_id)
            continue
        rinv = wf.invoke(role, receiver_messages(wf, role, inv.packet), tools=wf.roles[role].tools, ancestors=[inv.inv_id],
                         sources=list(inv.packet.sources), purpose=role)
        validate_receiver_packet(wf, role, rinv.packet, inv.packet)
        rinv.packet.extra["consumed"] = {"intake": inv.version}
        wf.trace.add("receiver_checked", inv_id=rinv.inv_id, role=role, packet=rinv.packet.public(), corrected=rinv.packet.extra.get("corrected", 0))
        wf.receiver_cache[role] = (key, rinv)
        parts[role] = rinv
    return joint_invocation(wf, inv, parts)


def question_footprint(q) -> tuple:
    """R_t(q): receivers reachable from the targeted premise, plus the downstream planner and retriever (G.3)."""
    if q.scope.get("pairs") is not None:
        owners = sorted({FIELD_OWNER[f] for _, f in q.scope["pairs"] if f in FIELD_OWNER})
    else:
        f = q.scope.get("field")
        owners = [FIELD_OWNER[f]] if f in FIELD_OWNER else []
    return tuple(owners) + ("planner", "retriever")


def diagnose_relaxation(wf, env, programs: list[dict]) -> dict:
    """G.4 counterfactual relaxation diagnosis: per unmatched need, apply the adjustments in RELAXATION_ORDER and keep the
    first (minimal) one that recovers resources; at most MAX_ADJUSTMENTS evaluations per step. Output format is the one
    fallback_round consumes (field words location/schedule/intake/documents/eligibility)."""
    ind, index = env.n["indiana211"], env.n["index"]
    word = {"schedule_time": "schedule", "schedule": "schedule", "location": "location", "intake": "intake", "documents": "documents", "eligibility": "eligibility"}

    def apply(prog: dict, rung: str) -> dict:
        a = copy.deepcopy(prog)
        if rung == "schedule_time":
            a["schedule"] = {k: v for k, v in (a.get("schedule") or {}).items() if k == "day"}
        elif rung == "schedule":
            a["schedule"] = {}
        elif rung == "location":
            a["counties"], a["cities"], a["zipcodes"] = [], [], []
        elif rung == "intake":
            a["intake_methods"] = []
        elif rung == "documents":
            a["available_documents"] = []
        elif rung == "eligibility":
            a["eligibility"] = []
        return a

    def applicable(prog: dict, rung: str) -> bool:
        sch = prog.get("schedule") or {}
        if rung == "schedule_time":
            return bool(sch) and any(k in sch for k in ("time", "start_time", "end_time"))
        if rung == "schedule":
            return bool(sch)
        if rung == "location":
            return bool(prog.get("counties") or prog.get("cities") or prog.get("zipcodes"))
        if rung == "intake":
            return bool(prog.get("intake_methods"))
        if rung == "documents":
            return bool(prog.get("available_documents"))
        return bool(prog.get("eligibility"))

    report, per_need, evaluated = {}, [[] for _ in programs], 0
    pending = [i for i, p in enumerate(programs) if not ind.execute_search_resources(index, p, limit=env.tool_limit).get("resources")]
    for rung in RELAXATION_ORDER:
        if not pending or evaluated >= MAX_ADJUSTMENTS:
            break
        for i in list(pending):
            if evaluated >= MAX_ADJUSTMENTS:
                break
            if not applicable(programs[i], rung):
                continue
            n = len(ind.execute_search_resources(index, apply(programs[i], rung), limit=env.tool_limit).get("resources") or [])
            evaluated += 1
            report[f"need-{i + 1}:{rung}"] = n
            if n > 0:
                per_need[i].append(word[rung])
                pending.remove(i)
    # pairs (two constraints at once) only when single adjustments recovered nothing and evaluations remain
    import itertools
    for i in list(pending):
        if evaluated >= MAX_ADJUSTMENTS:
            break
        rungs = [r for r in ("schedule", "location", "intake") if applicable(programs[i], r)]
        for a_, b_ in itertools.combinations(rungs, 2):
            if evaluated >= MAX_ADJUSTMENTS:
                break
            n = len(ind.execute_search_resources(index, apply(apply(programs[i], a_), b_), limit=env.tool_limit).get("resources") or [])
            evaluated += 1
            report[f"need-{i + 1}:{a_}+{b_}"] = n
            if n > 0:
                per_need[i].append(f"{word[a_]}+{word[b_]}")
                break
    union = []
    for found in per_need:
        for combo in found[:1]:
            for f in combo.split("+"):
                if f not in union:
                    union.append(f)
    minimal = ["+".join(union[:MAX_FALLBACK_QUESTIONS])] if union else []
    return {"report": report, "per_need": [f[:1] for f in per_need], "minimal": minimal, "evaluated": evaluated}
