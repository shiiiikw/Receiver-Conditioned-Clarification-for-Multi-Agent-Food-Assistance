"""Role specifications: task prompts, tool schemas, parsers, canonicalizers.

Every role emits its artifact as a native Qwen tool call (the format the
backbone is trained on), parsed by a lenient XML/JSON parser. HoosierHelp roles:
intake (user-facing interpreter), planner (search-program reasoner), retriever
(verifier/recommender); the fan-out receivers (access, schedule, logistics)
are defined in fanout2.py.

Task-specific modules:
  * HoosierHelp slot canonicalizer: artifacts are need x field structures with
    the native normalization plus explicit "unknown" / no-constraint states;
    atomic questions address one (need, field) distinction.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Callable

from .common import canonical, ensure_vendor_on_path

FIELDS = ("location", "schedule", "intake_methods", "available_documents", "eligibility")
FIELD_LABEL = {"location": "location", "schedule": "schedule", "intake_methods": "intake method",
               "available_documents": "available documents", "eligibility": "eligibility"}
INTAKE_GROUPS = (("location",), ("schedule", "intake_methods"), ("available_documents", "eligibility"))
UNKNOWN, NONE = "unknown", "none"
EMPTY = "__empty__"  # empty container from the model: none if the field was asked, unknown otherwise
NONE_TOKENS = {"none", "no constraint", "no constraints", "any", "no preference", "no", "n/a", "null", "not required", "no requirement", "flexible"}
UNKNOWN_TOKENS = {"unknown", "", "?", "not asked", "unasked", "not stated", "unspecified"}

TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


# --------------------------------------------------------------------------
# lenient JSON / tool-call parsing
# --------------------------------------------------------------------------
def repair_json(text: str) -> str:
    """Close unbalanced brackets (truncation) and fix a missing ']' before a top-level key."""
    out, stack, in_str, esc = [], [], False, False
    for ch in text:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack and ((ch == "]" and stack[-1] == "[") or (ch == "}" and stack[-1] == "{")):
                stack.pop()
            elif stack and stack[-1] == "[" and ch == "}":
                # a '}' while a list is open: the model forgot ']' -> close the list first
                out.insert(len(out) - 1, "]")
                stack.pop()
                if stack and stack[-1] == "{":
                    stack.pop()
    if in_str:
        out.append('"')
    while stack:
        out.append("]" if stack.pop() == "[" else "}")
    return "".join(out)


def _fix_common_slips(text: str) -> str:
    # {"unknown"} / {"none"} set literals (invalid JSON the 8B model writes for empty objects) -> plain tokens
    text = re.sub(r'\{\s*"(unknown|none|empty)"\s*\}', r'"\1"', text)
    # trailing commas before a closing bracket; a top-level key emitted inside the needs list
    text = re.sub(r",\s*([\]}])", r"\1", text)
    text = re.sub(r"\}\s*,\s*\"sources\"\s*:\s*(\[[^\]]*\])\s*\}?\s*\]", lambda m: "}], \"sources\": " + m.group(1), text)
    return text


def parse_json_object(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    fixed = _fix_common_slips(body)
    for candidate in (body, fixed, repair_json(body), repair_json(fixed)):
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return _salvage_truncated(fixed)


def _salvage_truncated(body: str, max_cuts: int = 80):
    """Last resort for an output cut off by the token limit: step back to the last structural boundary (',' / ':' outside
    strings) before the cut, drop the unfinished element, close the brackets and parse again. Keeps everything the model
    completed before the cut (the fan-out receivers' decisions are re-derived deterministically anyway)."""
    cuts, in_str, esc = [], False, False
    for i, ch in enumerate(body):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in ",:": cuts.append(i)
    for pos in reversed(cuts[-max_cuts:]):
        head = body[:pos]
        if body[pos] == ":":            # unfinished key -> drop the key string too
            q = head.rstrip().rfind('"', 0, len(head.rstrip()) - 1)
            head = head[:q] if q > 0 else head
            head = head.rstrip().rstrip(",")
        try:
            obj, _ = json.JSONDecoder().raw_decode(repair_json(re.sub(r",\s*$", "", head)))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """All <tool_call> blocks as (name, arguments); falls back to a bare JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    calls = []
    for m in TOOL_BLOCK.finditer(text):
        obj = parse_json_object(m.group(1))
        if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict) and isinstance(obj.get("name"), str):
            calls.append((obj["name"], obj["arguments"]))
    if calls:
        return calls
    obj = parse_json_object(text)
    if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict) and isinstance(obj.get("name"), str):
        return [(obj["name"], obj["arguments"])]
    if isinstance(obj, dict):
        return [("__bare__", obj)]
    return []


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


@dataclass
class RoleSpec:
    name: str
    task: str
    parse: Callable[[str], dict | None]
    canon: Callable[[dict], tuple[str, dict]]
    render: Callable[[dict], str]
    tools: list[dict]
    max_tokens: int | None = None   # per-role proposal length; None -> workflow.PROPOSAL_MAX_TOKENS


# --------------------------------------------------------------------------
# HoosierHelp
# --------------------------------------------------------------------------
def _scoring():
    ensure_vendor_on_path()
    from eval import scoring
    return scoring


def _norm_field(field: str, value):
    """Normalize one intake field value to 'unknown' | 'none' | EMPTY | concrete native value."""
    sc = _scoring()
    if isinstance(value, bool) or value is None:
        return UNKNOWN
    if isinstance(value, str):
        v = value.strip().lower()
        if v in UNKNOWN_TOKENS:
            return UNKNOWN
        if v in NONE_TOKENS:
            return NONE
        value = [value] if field in ("intake_methods", "available_documents", "eligibility") else {"text": value}
    if field in ("intake_methods", "available_documents", "eligibility"):
        if not isinstance(value, list):
            return UNKNOWN
        items = [str(x).strip() for x in value if isinstance(x, (str, int, float))]
        if not items:
            return EMPTY
        lowered = [x.lower() for x in items]
        if all(x in NONE_TOKENS or x in UNKNOWN_TOKENS for x in lowered):
            return UNKNOWN if any(x in UNKNOWN_TOKENS for x in lowered) else NONE
        items = [x for x in items if x.lower() not in NONE_TOKENS and x.lower() not in UNKNOWN_TOKENS]
        norm = {"intake_methods": sc._string_list, "available_documents": sc._document_list, "eligibility": sc._eligibility_list}[field](items)
        if field == "intake_methods" and len(set(norm)) >= 6:
            return NONE  # listing (nearly) every method means no constraint
        return sorted(norm) if norm else NONE
    if not isinstance(value, dict):
        return UNKNOWN
    if field == "location":
        if "text" in value:
            return UNKNOWN
        clean = {}
        for k in ("counties", "cities", "zipcodes"):
            raw = value.get(k)
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                continue
            raw = [x for x in raw if isinstance(x, str) and x.strip().lower() not in NONE_TOKENS | UNKNOWN_TOKENS]
            vals = sc._string_list(raw) if k == "zipcodes" else sc._location_list(raw)
            if vals:
                clean[k] = vals
        if clean:
            return clean
        explicit_none = any(isinstance(value.get(k), str) and value[k].strip().lower() in NONE_TOKENS for k in ("counties", "cities", "zipcodes"))
        return NONE if explicit_none else EMPTY
    if field == "schedule":
        if "text" in value:
            return UNKNOWN
        kept, explicit_none = {}, False
        for k, v in value.items():
            if v is None or v is False or (isinstance(v, str) and v.strip().lower() in UNKNOWN_TOKENS):
                continue
            if isinstance(v, str) and v.strip().lower() in NONE_TOKENS:
                explicit_none = True
                continue
            kept[k] = v
        if not kept:
            return NONE if explicit_none else (EMPTY if not value else UNKNOWN)
        norm = sc.normalize_schedule(kept)
        if not norm or (norm.get("day") and norm["day"] not in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}):
            return UNKNOWN
        return norm
    return UNKNOWN


INTAKE_TOOLS = [tool("report_evidence_state", "Report the current structured evidence state for the person's need(s).", {
    "needs": {"type": "array", "description": "One entry per distinct service need (one or two).", "items": {"type": "object", "properties": {
        "need_id": {"type": "string", "enum": ["need-1", "need-2"]},
        "service_categories": {"type": "array", "items": {"type": "string"}, "description": "Exactly one allowed category."},
        "location": {"description": "\"unknown\" | \"none\" | {\"counties\": [...], \"cities\": [...], \"zipcodes\": [...]} (only what the person said)"},
        "schedule": {"description": "\"unknown\" | \"none\" | {\"day\": \"mon\"..\"sun\", \"time\": \"HH:MM\"} | {\"day\":..., \"start_time\":..., \"end_time\":...} | {\"requires_24_hours\": true}"},
        "intake_methods": {"description": "\"unknown\" | \"none\" | array from [appointment, call, email, mail, online, text, walk_in]"},
        "available_documents": {"description": "\"unknown\" | \"none\" | array from [birth_certificate, insurance_card, lease, photo_id, proof_of_address, proof_of_income, social_security, utility_bill]"},
        "eligibility": {"description": "\"unknown\" | \"none\" | array from [disability, family, homeless, low_income, medicaid, pregnant, resident, senior, uninsured, veteran, youth]"}},
        "required": ["need_id", "service_categories", "location", "schedule", "intake_methods", "available_documents", "eligibility"]}},
    "sources": {"type": "array", "items": {"type": "integer"}, "description": "1-based indices of the person's messages supporting the entries."}},
    ["needs", "sources"])]


def parse_intake(text: str):
    for name, args in parse_tool_calls(text):
        if name in ("report_evidence_state", "__bare__") and isinstance(args.get("needs"), list) and args["needs"]:
            return args
    return None


def canon_intake(obj: dict) -> tuple[str, dict]:
    sc = _scoring()
    needs = []
    for i, need in enumerate(obj["needs"][:2]):
        if not isinstance(need, dict):
            continue
        cats = sc._string_list(need.get("service_categories"))
        n = {"need_id": f"need-{i + 1}", "service_categories": cats or UNKNOWN}
        for f in FIELDS:
            n[f] = _norm_field(f, need.get(f, UNKNOWN))
        needs.append(n)
    # a duplicated need (same single category) is one need
    deduped, seen = [], set()
    for n in needs:
        key = tuple(n["service_categories"]) if isinstance(n["service_categories"], list) else n["service_categories"]
        if key in seen:
            continue
        seen.add(key)
        n["need_id"] = f"need-{len(deduped) + 1}"
        deduped.append(n)
    needs = deduped
    if not needs:
        needs = [{"need_id": "need-1", "service_categories": UNKNOWN, **{f: UNKNOWN for f in FIELDS}}]
    sources = sorted({int(s) for s in (obj.get("sources") or []) if isinstance(s, (int, float)) and not isinstance(s, bool)})
    normalized = {"needs": needs, "sources": sources}
    return canonical(needs), normalized


def time_ranges(text: str) -> list[tuple[str, str]]:
    """Explicit time windows in the person's words ('between 8:30 AM and 9:30 AM', 'from 2 to 4 pm') as (start, end) HH:MM."""
    out = []
    pat = re.compile(r"(?:between|from)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*(?:and|to|-|–)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
    for m in pat.finditer(text or ""):
        h1, m1, ap1, h2, m2, ap2 = m.groups()
        ap1 = (ap1 or ap2 or "").replace(".", "").lower()
        ap2 = (ap2 or ap1 or "").replace(".", "").lower()
        if not ap1 and not ap2:
            # 'between 9:30 and 11:30 in the morning' / 'from 2 to 4 in the afternoon'
            tail = (text or "")[m.end():m.end() + 40].lower()
            if re.match(r"\s*(in the )?(morning)", tail):
                ap1 = ap2 = "am"
            elif re.match(r"\s*(in the )?(afternoon|evening|at night)", tail):
                ap1 = ap2 = "pm"
        def conv(h, mm, ap):
            h = int(h)
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            return f"{h:02d}:{mm or '00'}"
        a, b = conv(h1, m1, ap1), conv(h2, m2, ap2)
        if a < b:
            out.append((a, b))
    return out


def time_supported(hhmm: str, text: str) -> bool:
    """A schedule time literal must be traceable to the person's words (12h or 24h form)."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm or "")
    if not m:
        return False
    h, mm = int(m.group(1)), m.group(2)
    forms = {f"{h}:{mm}", f"{h % 12 or 12}:{mm}", f"{h:02d}:{mm}"}
    if mm == "00":
        forms |= {f"{h % 12 or 12} am", f"{h % 12 or 12} pm", f"{h % 12 or 12}am", f"{h % 12 or 12}pm", f"{h % 12 or 12} o'clock", f"{h} o'clock"}
    t = text.lower()
    return any(f in t for f in forms)


FIELD_VALUE_LEXICON = {
    "photo_id": ["photo id", "photo identification", "driver", "id card", "identification"], "birth_certificate": ["birth certificate"],
    "insurance_card": ["insurance"], "lease": ["lease"], "proof_of_address": ["proof of address", "address proof", "utility bill as proof"],
    "proof_of_income": ["proof of income", "income", "pay stub", "paystub"], "social_security": ["social security"], "utility_bill": ["utility bill"],
    "disability": ["disabilit", "disabled"], "family": ["family", "children", "kids", "child"], "homeless": ["homeless"], "low_income": ["low income", "low-income", "income"],
    "medicaid": ["medicaid"], "pregnant": ["pregnan", "expecting"], "resident": ["resident"], "senior": ["senior", "elderly", "older adult"],
    "uninsured": ["uninsured", "no insurance", "without insurance"], "veteran": ["veteran"], "youth": ["youth", "teen", "young"],
}


def render_intake(a: dict) -> str:
    lines = []
    for n in a["needs"]:
        parts = [f"{n['need_id']}: service={n['service_categories']}"]
        for f in FIELDS:
            parts.append(f"{f}={json.dumps(n[f]) if not isinstance(n[f], str) else n[f]}")
        lines.append("; ".join(parts))
    return "\n".join(lines)


def intake_missing(a: dict) -> list[tuple[str, str]]:
    out = []
    for n in a["needs"]:
        if n["service_categories"] == UNKNOWN:
            out.append((n["need_id"], "service_categories"))
    for group in INTAKE_GROUPS:
        for n in a["needs"]:
            for f in group:
                if n[f] in (UNKNOWN, EMPTY):
                    out.append((n["need_id"], f))
    return out


def intake_to_search_args(need: dict) -> dict | None:
    if need["service_categories"] == UNKNOWN or any(need[f] in (UNKNOWN, EMPTY) for f in FIELDS):
        return None
    loc = need["location"] if isinstance(need["location"], dict) else {}
    return {"service_categories": list(need["service_categories"]),
            "schedule": need["schedule"] if isinstance(need["schedule"], dict) else {},
            "counties": loc.get("counties", []), "cities": loc.get("cities", []), "zipcodes": loc.get("zipcodes", []),
            "intake_methods": need["intake_methods"] if isinstance(need["intake_methods"], list) else [],
            "available_documents": need["available_documents"] if isinstance(need["available_documents"], list) else [],
            "eligibility": need["eligibility"] if isinstance(need["eligibility"], list) else []}


def parse_planner(text: str):
    programs = [args for name, args in parse_tool_calls(text) if name == "search_resources" and isinstance(args, dict)]
    if not programs:
        bare = [args for name, args in parse_tool_calls(text) if name == "__bare__"]
        if bare and isinstance(bare[0].get("programs"), list):
            programs = [p.get("arguments") for p in bare[0]["programs"] if isinstance(p, dict) and isinstance(p.get("arguments"), dict)]
    if not programs:
        return None
    return {"programs": [{"need_id": f"need-{i + 1}", "arguments": a} for i, a in enumerate(programs[:2])]}


def canon_planner(obj: dict) -> tuple[str, dict]:
    sc = _scoring()
    programs = []
    for i, p in enumerate(obj["programs"][:2]):
        norm = sc.normalize_tool_args(p["arguments"])
        if not norm["service_categories"]:
            continue
        # none-like tokens are "no constraint" (empty), never literal values
        for f in ("intake_methods", "available_documents", "eligibility", "counties", "cities", "zipcodes"):
            norm[f] = [v for v in norm[f] if v.lower() not in NONE_TOKENS and v.lower() not in UNKNOWN_TOKENS]
        sch = norm.get("schedule") or {}
        if sch.get("day") and sch["day"] not in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}:
            sch = {}
        norm["schedule"] = {k: v for k, v in sch.items() if not (isinstance(v, str) and v.lower() in NONE_TOKENS | UNKNOWN_TOKENS)}
        programs.append({"need_id": f"need-{len(programs) + 1}", "arguments": norm})
    if not programs:
        raise ValueError("no valid program")
    normalized = {"programs": programs}
    return canonical([p["arguments"] for p in programs]), normalized


def parse_canon_planner(text: str):
    obj = parse_planner(text)
    if obj is None:
        return None
    try:
        canon_planner(obj)
    except ValueError:
        return None
    return obj


def render_planner(a: dict) -> str:
    return "\n".join(f"{p['need_id']}: search_resources({json.dumps(p['arguments'], ensure_ascii=False)})" for p in a["programs"])


RETRIEVER_EXTRA_TOOLS = [tool("request_fallback", "Report that a search returned no resources and name the constraint to ask the person to relax.", {
    "field": {"type": "string", "enum": ["location", "schedule", "intake"]}}, ["field"])]


def parse_retriever(text: str):
    for name, args in parse_tool_calls(text):
        if name == "final_recommendation" and isinstance(args.get("resource_ids"), list):
            return {"resource_ids": args["resource_ids"], "message": args.get("message", "")}
        if name == "request_fallback":
            return {"empty": True, "fallback_request": str(args.get("field") or "location, schedule or intake")}
        if name == "__bare__":
            if isinstance(args.get("resource_ids"), list):
                return {"resource_ids": args["resource_ids"], "message": args.get("message", "")}
            if args.get("empty") or args.get("fallback_request") or args.get("field"):
                return {"empty": True, "fallback_request": str(args.get("fallback_request") or args.get("field") or "location, schedule or intake")}
    return None


def canon_retriever(obj: dict) -> tuple[str, dict]:
    if isinstance(obj.get("resource_ids"), list) and obj["resource_ids"]:
        ids = sorted({str(x).strip() for x in obj["resource_ids"] if str(x).strip()})
        normalized = {"resource_ids": ids, "message": str(obj.get("message") or "")}
        return canonical(ids), normalized
    req = str(obj.get("fallback_request") or "").strip().lower() or "location, schedule or intake"
    normalized = {"empty": True, "fallback_request": req}
    return canonical({"fallback": req}), normalized


def render_retriever(a: dict) -> str:
    if a.get("empty"):
        return f"No matching resources; ask the user whether {a['fallback_request']} can change."
    return f"final_recommendation(resource_ids={a['resource_ids']})"


HOOSIER_INTAKE_TASK = """You are the INTAKE agent in a three-agent Indiana 211 resource-navigation team (intake -> planner -> retriever).
Your task: read the conversation with the person seeking help and report the current structured evidence state by calling the report_evidence_state tool exactly once.
Needs: if the person names two different service needs, create two entries (need-1 and need-2), each with exactly ONE service category. Otherwise one entry. Requests that the agent itself pay, buy, order, arrange or provide something are not separate needs: map them to the underlying service category (paying a bill -> Utility Assistance or Financial Assistance and Benefits as stated, buying groceries -> Food Assistance, a ride -> Transportation).
For each need record the service category and five search fields. Each field must be exactly one of:
  - "unknown": the person has not been asked about it or did not answer (silence is NOT "none");
  - "none": the person explicitly said they have no constraint / no preference / nothing for this field, or (after an empty search) that they are flexible on it;
  - a concrete value: location {"counties":["MARION"]} or {"cities":["Peru"]} or {"zipcodes":["47421"]} (only what the person said; never infer a county from a city or ZIP);
    schedule {"day":"thu"} or {"day":"thu","time":"14:00"} or {"day":"thu","start_time":"09:00","end_time":"12:00"} or {"requires_24_hours":true};
    intake_methods from [appointment, call, email, mail, online, text, walk_in] (phone -> call, in person -> walk_in; include only methods the person named as requirements; if any method is fine, "none");
    available_documents from [birth_certificate, insurance_card, lease, photo_id, proof_of_address, proof_of_income, social_security, utility_bill];
    eligibility from [disability, family, homeless, low_income, medicaid, pregnant, resident, senior, uninsured, veteran, youth].
If the person states a value and denies the very same value in one sentence ("X works for me, but I cannot do X"), X is their real requirement: record X.
Never write an empty list or empty object; write "unknown" or "none" instead. Never invent values. Bind each fact to the need the person attached it to; a fact stated for one need does not apply to the other unless the person said so.
After an empty search: if the person gives a new location/schedule/intake value, replace the old value with it; if they say they can adjust, are flexible, or do not need that constraint any more, set that field to "none"; if they say a constraint cannot change, keep it.
Record the 1-based indices of the person's messages that support the entries in "sources"."""

HOOSIER_PLANNER_TASK = """You are the PLANNER agent in a three-agent Indiana 211 resource-navigation team (intake -> planner -> retriever).
You do NOT see the user conversation. You receive the intake agent's handoff: alternative interpretations of the person's needs with empirical support. Your task: produce the concrete search program by calling search_resources exactly once per need (two needs -> two separate search_resources calls, each with its own single service category and its own schedule/intake values). Use only values allowed by the schema. Leave a field empty when the handoff says none/unknown; include every acceptable value when several were given. Never add a value (eligibility, documents, intake method, schedule, location) that the handoff does not state. Do not infer a county. If a required fact is genuinely unresolved, still produce your best program; the system may ask the intake agent for clarification separately."""

HOOSIER_RETRIEVER_TASK = """You are the RETRIEVER agent in a three-agent Indiana 211 resource-navigation team (intake -> planner -> retriever).
You receive the planner's search program handoff and the actual search results. Your task: call final_recommendation with the single best returned resource_id for each need (exactly as returned, one per need), or, if a search returned no resources, call request_fallback naming the constraint (location, schedule or intake) most likely to be relaxable. Call exactly one tool."""
def hoosier_roles(search_schema_nested: dict, final_schema_nested: dict) -> dict[str, RoleSpec]:
    return {
        "intake": RoleSpec("intake", HOOSIER_INTAKE_TASK, parse_intake, canon_intake, render_intake, INTAKE_TOOLS),
        "planner": RoleSpec("planner", HOOSIER_PLANNER_TASK, parse_canon_planner, canon_planner, render_planner, [search_schema_nested]),
        "retriever": RoleSpec("retriever", HOOSIER_RETRIEVER_TASK, parse_retriever, canon_retriever, render_retriever,
                              [final_schema_nested] + RETRIEVER_EXTRA_TOOLS),
    }
