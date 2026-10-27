"""Candidate-indexed uncertainty packets and their readable rendering.

A Packet is the message one invocation sends to its receiver: the retained
candidate artifacts with empirical support weights, the residual category and
provenance (source utterance indices / cited evidence). Every role and every
controller reads the same readable candidates, support weights and provenance.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

ALPHA = 0.1
K_MAX = 4
RESIDUAL_KEY = "__residual__"
UNRESOLVED_KEY = "__unresolved__"


@dataclass
class Candidate:
    key: str                    # canonical identity (role-specific canonicalizer)
    artifact: Any               # normalized structured artifact
    text: str                   # readable rendering
    count: int = 0              # stochastic sample count n_ik
    weight: float = 0.0         # support weight w_ik
    greedy: bool = False
    rep_text: str = ""          # representative generation (max length-normalized logprob)
    rep_logprob: float = 0.0
    sources: list[int] = field(default_factory=list)  # cited evidence indices (provenance)
    is_unresolved: bool = False

    def public(self) -> dict:
        return {"key": self.key, "artifact": self.artifact, "text": self.text, "count": self.count,
                "weight": round(self.weight, 4), "greedy": self.greedy, "sources": self.sources,
                "is_unresolved": self.is_unresolved, "rep_logprob": round(self.rep_logprob, 4)}


@dataclass
class Packet:
    origin: str                 # invocation id, e.g. "intake#3"
    version: int
    sender_role: str
    candidates: list[Candidate]
    residual_weight: float
    residual_count: int
    invalid_count: int
    n_stochastic: int
    ancestors: list[str] = field(default_factory=list)   # origins this packet consumed
    sources: list[int] = field(default_factory=list)     # evidence indices the sender had access to
    receiver_task: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def greedy(self) -> Candidate:
        for c in self.candidates:
            if c.greedy:
                return c
        return self.candidates[0]

    def public(self) -> dict:
        return {"origin": self.origin, "version": self.version, "sender_role": self.sender_role,
                "candidates": [c.public() for c in self.candidates], "residual_weight": round(self.residual_weight, 4),
                "residual_count": self.residual_count, "invalid_count": self.invalid_count, "n_stochastic": self.n_stochastic,
                "ancestors": list(self.ancestors), "sources": list(self.sources),
                "extra": {k: v for k, v in self.extra.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}}

    def readable(self, max_candidates: int = K_MAX, with_sources: bool = True) -> str:
        """Rich-text rendering shared by every method (candidates, support, provenance)."""
        lines = [f"Handoff from the {self.sender_role} agent (message version {self.version}). "
                 f"Alternative interpretations with empirical support from {self.n_stochastic + 1} draws:"]
        for i, c in enumerate(self.candidates[:max_candidates], 1):
            tag = " [greedy]" if c.greedy else ""
            src = f" (grounded in user turns {c.sources})" if with_sources and c.sources else ""
            lines.append(f"Candidate {i}{tag}, support {c.weight:.2f}{src}:\n{c.text}")
        lines.append(f"Unlisted alternatives (residual support {self.residual_weight:.2f}; {self.invalid_count} invalid draws).")
        return "\n".join(lines)


def build_packet(origin: str, version: int, sender_role: str, samples, parse_fn, canon_fn, render_fn,
                 *, k_max: int = K_MAX, alpha: float = ALPHA, ancestors=(), sources=(), unresolved_text: str | None = None,
                 greedy_counts: bool = False) -> Packet:
    """Group S stochastic draws + 1 greedy draw into <= K candidates.

    `samples[0]` must be the greedy draw. parse_fn(text) -> artifact or None;
    canon_fn(artifact) -> (key, normalized_artifact); render_fn(normalized) -> str.
    """
    greedy, stochastic = samples[0], samples[1:]
    S = len(stochastic)
    groups: dict[str, dict] = {}
    invalid = 0

    def admit(sample, is_greedy):
        nonlocal invalid
        artifact = parse_fn(sample.text)
        if artifact is None:
            if not is_greedy:
                invalid += 1
            return None
        key, normalized = canon_fn(artifact)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"artifact": normalized, "count": 0, "greedy": False, "best": None, "sources": set()}
        if not is_greedy:
            g["count"] += 1
        else:
            g["greedy"] = True
        if g["best"] is None or sample.mean_logprob > g["best"].mean_logprob:
            g["best"] = sample
        for s in (normalized.get("sources") or []) if isinstance(normalized, dict) else []:
            g["sources"].add(int(s))
        return key

    admit(greedy, True)
    for s in stochastic:
        admit(s, False)
    # Retain the greedy artifact and the most frequent distinct alternatives (<= K total).
    ordered = sorted(groups.items(), key=lambda kv: (not kv[1]["greedy"], -kv[1]["count"], kv[0]))
    retained = ordered[:k_max]
    discarded = ordered[k_max:]
    k_i = len(retained)
    if greedy_counts:
        # paper App. G.2: support = cluster size / total samples, the greedy draw counted as one sample, no pseudocount
        denom = S + 1
        mass = lambda g: g["count"] + (1 if g["greedy"] else 0)
    else:
        denom = S + alpha * (k_i + 1)
        mass = lambda g: g["count"] + alpha
    candidates = []
    for key, g in retained:
        candidates.append(Candidate(key=key, artifact=g["artifact"], text=render_fn(g["artifact"]), count=g["count"],
                                    weight=mass(g) / denom, greedy=g["greedy"], rep_text=g["best"].text,
                                    rep_logprob=g["best"].mean_logprob, sources=sorted(g["sources"])))
    residual_count = sum(g["count"] for _, g in discarded) + invalid
    residual_weight = ((S + 1 - sum(mass(g) for _, g in retained)) / denom if greedy_counts
                       else (S - sum(c.count for c in candidates) + alpha) / denom)
    if unresolved_text is not None:
        candidates.append(Candidate(key=UNRESOLVED_KEY, artifact={"unresolved": True}, text=unresolved_text,
                                    count=0, weight=0.0, is_unresolved=True))
    return Packet(origin=origin, version=version, sender_role=sender_role, candidates=candidates,
                  residual_weight=residual_weight, residual_count=residual_count, invalid_count=invalid,
                  n_stochastic=S, ancestors=list(ancestors), sources=list(sources))
