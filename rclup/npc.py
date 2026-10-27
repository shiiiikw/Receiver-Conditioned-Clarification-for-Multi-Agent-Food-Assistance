"""Simulated user (environment side, OpenAI model set by RCLUP_NPC_MODEL) with cost accounting.

HoosierHelp reuses the benchmark's own LLMSimulatedUser prompt logic verbatim;
only the transport is wrapped so token usage flows into the cost ledger and
replies are cached on disk (RUNS/cache/npc) keyed by the transcript prefix.
"""
from __future__ import annotations

import os
import re

import json
import random
import threading
from typing import Any

from .common import ensure_vendor_on_path, load_openai_key, RUNS, digest
from .cost import CostLedger

NPC_MODEL = os.environ.get("RCLUP_NPC_MODEL", "gpt-4.1-mini")   # simulated-user model (OpenAI Responses API)
# optional: replies cached under these other model names are reused for identical transcript prefixes
NPC_CACHE_FALLBACK_MODELS = tuple(x for x in os.environ.get("RCLUP_NPC_CACHE_FALLBACK", "").split(",") if x)
_client_lock = threading.Lock()
_client = None


def openai_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=load_openai_key(), timeout=90.0, max_retries=4)
        return _client


NPC_CACHE_DIR = RUNS / "cache" / "npc"
NPC_CACHE_STATS = {"hits": 0, "misses": 0}


def _cache_path(key: str):
    return NPC_CACHE_DIR / key[:2] / f"{key}.json"


def responses_call(ledger: CostLedger, *, instructions: str, input_messages: list[dict], max_output_tokens: int,
                   purpose: str, temperature: float = 0.0, model: str = NPC_MODEL) -> str:
    """Metered Responses API call with a persistent on-disk reply cache.

    The simulated user runs at temperature 0, so an identical (model, instructions, transcript) prefix yields the same
    reply; every controller shares the transcript up to its first divergent decision, and re-runs of the same
    configuration are free. Cache files are written atomically (tmp + rename), so concurrent workers need no lock.
    Set RCLUP_NPC_CACHE=0 to bypass."""
    use_cache = os.environ.get("RCLUP_NPC_CACHE", "1") != "0" and temperature == 0.0
    key = digest({"model": model, "instructions": instructions, "input": input_messages, "max_output_tokens": max_output_tokens,
                  "temperature": temperature}) if use_cache else None
    if use_cache:
        path = _cache_path(key)
        keys = [key] + [digest({"model": m, "instructions": instructions, "input": input_messages, "max_output_tokens": max_output_tokens,
                                "temperature": temperature}) for m in NPC_CACHE_FALLBACK_MODELS if m != model]
        for k in keys:
            try:
                pth = _cache_path(k)
                if pth.exists():
                    cached = json.loads(pth.read_text())
                    NPC_CACHE_STATS["hits"] += 1
                    ledger.record_cached(purpose)
                    return cached["output"]
            except (OSError, ValueError, KeyError):
                pass
        NPC_CACHE_STATS["misses"] += 1
    ledger.check()
    response = openai_client().responses.create(model=model, instructions=instructions, input=input_messages,
                                                temperature=temperature, max_output_tokens=max_output_tokens, store=False)
    ledger.record(getattr(response, "model", model) or model, response.usage, purpose)
    out = (getattr(response, "output_text", "") or "").strip()
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"output": out, "purpose": purpose, "model": getattr(response, "model", model) or model}))
            os.replace(tmp, path)
        except OSError:
            pass
    return out


CONTRADICTION_RE = re.compile(r"\b(but|however)\b[^.?!]{0,80}\b(cannot|can't|can not|do not|don't|am not able|not able|won't)\b", re.I)


class HoosierUser:
    """Native HoosierHelp simulated user with metered, cached OpenAI transport."""

    def __init__(self, spec: dict, behavior: str, ledger: CostLedger, seed: int = 7, max_output_tokens: int = 512):
        ensure_vendor_on_path()
        from eval import llm_user as native
        self.native_module = native
        self.ledger = ledger
        outer = self

        class Metered(native.LLMSimulatedUser):
            """Native user + (i) metered/cached transport, (ii) the documented self-contradiction protocol made explicit:
            the official instruction says the contradiction is used once and later clarifications get the real fact, but
            an LLM user left to itself tends to repeat the contradiction on every follow-up. Once the user's
            own transcript already contains the contradiction, the turn instruction switches to the official
            'resolved' instruction, which is what the benchmark text describes."""
            _last_messages: list = []

            def respond(self, messages, agent_message):
                self._last_messages = messages
                return super().respond(messages, agent_message)

            def _turn_instruction(self, opening):
                if (not opening and self.user_behavior == "self_contradictory" and self.contradiction_area
                        and any(m.get("role") == "user" and CONTRADICTION_RE.search(str(m.get("content", "")))
                                for m in self._last_messages if not str(m.get("content", "")).startswith("Tool result"))):
                    return native.self_contradiction_resolved_instruction(self.contradiction_area)
                return super()._turn_instruction(opening)

            def _generate_responses_api(self, messages, context):
                return responses_call(outer.ledger, instructions=native.USER_SYSTEM_PROMPT,
                                      input_messages=[*native.llm_role_mapped_messages(messages), {"role": "user", "content": context}],
                                      max_output_tokens=self.max_output_tokens, purpose="hoosier_npc", temperature=self.temperature)

        self.user = Metered(spec=spec, user_behavior=behavior, provider="openai", model=NPC_MODEL, seed=seed,
                            temperature=0.0, max_output_tokens=max_output_tokens, enable_thinking=False,
                            thinking_budget_tokens=None, client=object())

    def opening(self) -> str:
        return self.user.opening()

    def respond(self, messages: list[dict], agent_message: str) -> str:
        return self.user.respond(messages, agent_message)

    @property
    def state(self) -> dict:
        return {"contradiction_area": self.user.contradiction_area, "fallback_revealed": self.user.fallback_revealed}
