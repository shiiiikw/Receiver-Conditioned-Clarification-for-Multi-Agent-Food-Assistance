"""HoosierHelp environment: native user simulator, tool, turn cap and scorer.

The transcript, tool-result formatting, turn accounting and scoring replicate
`eval.runner.evaluate_conversation` exactly, but the agent side is driven by
our multi-agent workflow instead of a single backend. One *agent turn* is one
assistant action: a user-facing message, one or more search calls, or the
final recommendation (native cap: 8).
"""
from __future__ import annotations

import copy
from typing import Any

from ..common import VENDOR_HOOSIER, ensure_vendor_on_path
from ..cost import CostLedger
from ..npc import HoosierUser

MAX_AGENT_TURNS = 8
TOOL_RESULT_LIMIT = 10


class _Native:
    _loaded = None

    @classmethod
    def load(cls):
        if cls._loaded is None:
            ensure_vendor_on_path()
            from eval import runner, scoring
            from tools import indiana211, tool_protocol
            index = indiana211.load_resource_index(VENDOR_HOOSIER / "data/benchmark/filtered_resources_tagged.csv")
            cls._loaded = {"runner": runner, "scoring": scoring, "indiana211": indiana211, "protocol": tool_protocol,
                           "index": index, "schemas": tool_protocol.agent_tool_schemas(index)}
        return cls._loaded


def native_tool_schemas() -> list[dict]:
    return copy.deepcopy(_Native.load()["schemas"])


def native_agent_prompt() -> str:
    ensure_vendor_on_path()
    from eval.agent_prompt import AGENT_SYSTEM_PROMPT
    return AGENT_SYSTEM_PROMPT


def service_categories() -> list[str]:
    return list(_Native.load()["index"].service_categories)


class HoosierEnv:
    def __init__(self, spec: dict, behavior: str, ledger: CostLedger, user_seed: int = 7,
                 max_agent_turns: int = MAX_AGENT_TURNS, tool_limit: int = TOOL_RESULT_LIMIT):
        self.n = _Native.load()
        self.spec = spec
        self.behavior = behavior
        self.max_agent_turns = max_agent_turns
        self.tool_limit = tool_limit
        self.user = HoosierUser(spec, behavior, ledger, seed=user_seed)
        self.messages: list[dict] = []
        self.agent_turns = 0
        self.predicted_tool_calls: list[dict] = []
        self.executed_tool_results: list[dict] = []
        self.final_action: dict | None = None
        self.user_replies: list[int] = []  # transcript indices of NPC replies (provenance)
        self.first_turn_tool_call = False
        self.termination_reason = "max_agent_turns"
        self.done = False

    # ----- lifecycle -----
    def reset(self) -> str:
        opening = self.user.opening()
        self.messages = [{"role": "user", "content": opening}]
        return opening

    @property
    def turns_left(self) -> int:
        return self.max_agent_turns - self.agent_turns

    def _use_turn(self):
        if self.done:
            raise RuntimeError("Episode finished")
        if self.agent_turns >= self.max_agent_turns:
            raise RuntimeError("Native agent-turn budget exhausted")
        self.agent_turns += 1
        if self.agent_turns >= self.max_agent_turns:
            self.done = True

    # ----- actions -----
    def ask_user(self, text: str) -> str:
        self._use_turn()
        text = self.n["runner"].clean_tool_call_text(text) if hasattr(self.n["runner"], "clean_tool_call_text") else text
        self.messages.append({"role": "assistant", "content": text})
        reply = self.user.respond(self.messages, text)
        self.messages.append({"role": "user", "content": reply or ""})
        self.user_replies.append(len(self.messages) - 1)
        return reply

    def search(self, calls: list[dict]) -> list[dict]:
        """Execute one or more search_resources calls in a single agent turn."""
        self._use_turn()
        if self.agent_turns == 1:
            self.first_turn_tool_call = True
        runner, ind = self.n["runner"], self.n["indiana211"]
        parsed = [type("TC", (), {"name": "search_resources", "arguments": c})() for c in calls]
        self.messages.append({"role": "assistant", "content": runner.assistant_tool_call_message(parsed)})
        results = []
        for args in calls:
            self.predicted_tool_calls.append(args)
            result = ind.execute_search_resources(self.n["index"], args, limit=self.tool_limit)
            idx = len(self.predicted_tool_calls)
            self.executed_tool_results.append({"call_index": idx, "arguments": self.n["scoring"].normalize_tool_args(args), "result": result})
            self.messages.append({"role": "user", "content": runner.tool_result_message(idx, result)})
            results.append(result)
        return results

    def finish(self, resource_ids: list[str], message: str = "") -> dict:
        self._use_turn()
        proto = self.n["protocol"]
        action = proto.normalize_final_recommendation_args({"resource_ids": resource_ids, "message": message})
        parsed = [type("TC", (), {"name": "final_recommendation", "arguments": action})()]
        self.messages.append({"role": "assistant", "content": self.n["runner"].assistant_tool_call_message(parsed)})
        self.final_action = action
        self.termination_reason = "explicit_final_tool"
        self.done = True
        return self.score()

    def fork(self) -> "HoosierEnv":
        """Deep-copy the environment state for branching (the NPC is stateless given the transcript)."""
        clone = HoosierEnv.__new__(HoosierEnv)
        clone.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("user", "n")})
        clone.n = self.n
        for k in ("messages", "predicted_tool_calls", "executed_tool_results", "final_action", "user_replies"):
            setattr(clone, k, copy.deepcopy(getattr(self, k)))
        clone.user = HoosierUser(self.spec, self.behavior, self.user.ledger, seed=self.user.user.seed)
        clone.user.user.turn = self.user.user.turn
        clone.user.user.fallback_revealed = self.user.user.fallback_revealed
        return clone

    def state(self) -> dict:
        return {"messages": copy.deepcopy(self.messages), "agent_turns": self.agent_turns,
                "predicted_tool_calls": copy.deepcopy(self.predicted_tool_calls), "executed_tool_results": copy.deepcopy(self.executed_tool_results),
                "final_action": copy.deepcopy(self.final_action), "user_replies": list(self.user_replies),
                "first_turn_tool_call": self.first_turn_tool_call, "termination_reason": self.termination_reason, "done": self.done,
                "user_turn": self.user.user.turn, "fallback_revealed": self.user.user.fallback_revealed}

    @classmethod
    def from_state(cls, spec: dict, behavior: str, ledger: CostLedger, state: dict, user_seed: int = 7) -> "HoosierEnv":
        env = cls(spec, behavior, ledger, user_seed=user_seed)
        for k in ("messages", "agent_turns", "predicted_tool_calls", "executed_tool_results", "final_action", "user_replies",
                  "first_turn_tool_call", "termination_reason", "done"):
            setattr(env, k, copy.deepcopy(state[k]))
        env.user.user.turn = state["user_turn"]
        env.user.user.fallback_revealed = state["fallback_revealed"]
        return env

    def returned_resource_ids(self) -> list[str]:
        ids = []
        for item in self.executed_tool_results:
            for r in item["result"].get("resources") or []:
                if r["resource_id"] not in ids:
                    ids.append(r["resource_id"])
        return ids

    # ----- scoring (native) -----
    def score(self) -> dict:
        runner, scoring = self.n["runner"], self.n["scoring"]
        spec = self.spec
        expected = runner.expected_tool_calls(spec)
        expected_ids = runner.expected_resource_ids_from_spec(spec)
        valid_by_need = runner.valid_resource_ids_for_expected_calls(self.n["index"], expected, self.tool_limit)
        selected = runner.normalize_selected_resource_ids(
            list(self.final_action["resource_ids"]) if self.final_action else [], self.executed_tool_results)
        profile = spec.get("constraint_profile") or "direct_match"
        scored_calls = runner.tool_calls_for_scoring(self.predicted_tool_calls, self.executed_tool_results, profile)
        tool_score = scoring.score_tool_calls(scored_calls, expected)
        resource_score = {**scoring.score_resource_selection(selected, expected_ids),
                          "valid_resource_ids_by_need": valid_by_need,
                          "invalid_resource_ids": [r for r in selected if r not in set(expected_ids)]}
        score = {**tool_score, **resource_score,
                 "explicit_final_action": self.final_action is not None,
                 "first_turn_tool_call": self.first_turn_tool_call,
                 "premature_tool_call": self.first_turn_tool_call and runner.expected_has_nonservice_constraints(expected),
                 "end_to_end_match": bool(self.final_action is not None and tool_score["all_match"] and resource_score["resource_exact_match"]),
                 "agent_turns": self.agent_turns, "user_questions": len(self.user_replies),
                 "termination_reason": self.termination_reason}
        return score

    def record(self) -> dict:
        return {"root_id": self.spec.get("root_id"), "user_spec_id": self.spec["user_spec_id"], "behavior": self.behavior,
                "case_type": self.spec.get("case_type"), "constraint_profile": self.spec.get("constraint_profile"),
                "messages": copy.deepcopy(self.messages), "user_reply_indices": list(self.user_replies),
                "predicted_tool_calls": [self.n["scoring"].normalize_tool_args(c) for c in self.predicted_tool_calls],
                "executed_tool_results": copy.deepcopy(self.executed_tool_results),
                "final_action": copy.deepcopy(self.final_action), "score": self.score(),
                "user_state": self.user.state}
