"""OpenAI cost ledger with a hard budget cap (USD).

Prices (USD per 1M tokens) for gpt-4.1: input 2.00, cached input 0.50, output 8.00.
The ledger persists to a JSON file so that every driver run in a phase shares
one cap. Exceeding the cap raises BudgetExceeded *before* the next call.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

PRICES = {
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
}


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    def __init__(self, path: Path, cap_usd: float, label: str = ""):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self.label = label
        self._lock = threading.Lock()
        self.state = {"label": label, "cap_usd": self.cap_usd, "usd": 0.0, "calls": 0,
                      "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "by_purpose": {}}
        if self.path.exists():
            prior = json.loads(self.path.read_text())
            prior["cap_usd"] = self.cap_usd
            self.state = prior

    @property
    def usd(self) -> float:
        return float(self.state["usd"])

    def check(self) -> None:
        if self.usd >= self.cap_usd:
            raise BudgetExceeded(f"OpenAI budget cap reached: ${self.usd:.2f} >= ${self.cap_usd:.2f} ({self.label})")

    def record_cached(self, purpose: str = "npc") -> None:
        """Count a cache-served simulated-user reply (no spend); kept in the ledger for reporting cache savings."""
        self._cached = getattr(self, "_cached", 0) + 1

    def record(self, model: str, usage, purpose: str = "npc") -> float:
        base = model.split("-20")[0] if model.startswith("gpt-4.1") else model
        price = PRICES.get(base) or PRICES["gpt-4.1"]
        inp = int(getattr(usage, "input_tokens", 0) or (usage.get("input_tokens", 0) if isinstance(usage, dict) else 0))
        out = int(getattr(usage, "output_tokens", 0) or (usage.get("output_tokens", 0) if isinstance(usage, dict) else 0))
        details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        if isinstance(usage, dict):
            cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
        usd = ((inp - cached) * price["input"] + cached * price["cached"] + out * price["output"]) / 1e6
        import fcntl
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_suffix(".lock")
            with open(lock_path, "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    # Multi-process safe: merge our delta into the on-disk state.
                    if self.path.exists():
                        try:
                            self.state = json.loads(self.path.read_text())
                        except json.JSONDecodeError:
                            pass
                    s = self.state
                    s["cap_usd"] = self.cap_usd
                    s["usd"] += usd
                    s["calls"] += 1
                    s["input_tokens"] += inp
                    s["cached_tokens"] += cached
                    s["output_tokens"] += out
                    p = s.setdefault("by_purpose", {}).setdefault(purpose, {"usd": 0.0, "calls": 0})
                    p["usd"] += usd
                    p["calls"] += 1
                    s["updated"] = time.time()
                    tmp = self.path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(s, indent=1))
                    os.replace(tmp, self.path)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        return usd

    def summary(self) -> dict:
        return dict(self.state)
