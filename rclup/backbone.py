"""Frozen Qwen3-8B backbone: sampling client for a vLLM OpenAI-compatible server.

The server is located either through RCLUP_VLLM_URL (e.g. http://host:8000/v1)
or through a status file RUNS/worker_<id>.json written by scripts/start_vllm.sh
or scripts/worker.sbatch.  Prompts are rendered client-side with the Qwen chat
template (thinking disabled) and sent to /v1/completions with logprobs.
"""
from __future__ import annotations

import glob
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import MODEL_DIR, RUNS

_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"


@dataclass
class Sample:
    text: str
    tokens: list[str]
    logprobs: list[float]
    finish_reason: str
    greedy: bool = False

    @property
    def mean_logprob(self) -> float:
        return sum(self.logprobs) / max(1, len(self.logprobs))


@dataclass
class GenUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    by_purpose: dict = field(default_factory=dict)


def discover_worker(job_id: str | None = None) -> dict:
    job_id = job_id or os.environ.get("RCLUP_WORKER_JOB")
    paths = [RUNS / f"worker_{job_id}.json"] if job_id else sorted(glob.glob(str(RUNS / "worker_*.json")), key=os.path.getmtime)
    for path in reversed(list(paths)):
        try:
            status = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("status") == "ready":
            return status
        if status.get("status") == "starting" and status.get("host"):
            # The start script's readiness loop may have timed out while weights were still loading; probe directly.
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://{status['host']}:{status['port']}/v1/models", timeout=5) as r:
                    if b"qwen3-8b" in r.read():   # served-model-name used by the start scripts
                        status["status"] = "ready"
                        Path(path).write_text(json.dumps(status))
                        return status
            except Exception:
                continue
    raise RuntimeError("No ready vLLM worker found: set RCLUP_VLLM_URL or start scripts/start_vllm.sh / scripts/worker.sbatch")


class VLLMClient:
    """Thread-safe sampling client with usage accounting."""

    def __init__(self, base_url: str | None = None, model: str = "qwen3-8b", timeout: float = 600.0):
        from openai import OpenAI
        base_url = base_url or os.environ.get("RCLUP_VLLM_URL")
        if not base_url:
            status = discover_worker()
            base_url = f"http://{status['host']}:{status['port']}/v1"
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout, max_retries=3)
        self.usage = GenUsage()
        self._lock = threading.Lock()
        from transformers import AutoTokenizer  # eager: lazy imports race across threads
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=_OFFLINE)

    def render(self, messages: list[dict], tools=None, thinking: bool = False) -> str:
        """Client-side chat-template rendering (identical to the HF encoder's rendering)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=_OFFLINE)
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
        if tools:
            kwargs["tools"] = tools
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def chat(self, messages: list[dict], *, n: int = 1, temperature: float = 0.0, top_p: float = 1.0,
             max_tokens: int = 512, seed: int | None = None, tools: list[dict] | None = None,
             purpose: str = "gen", stop: list[str] | None = None, thinking: bool = False) -> list[Sample]:
        prompt = self.render(messages, tools, thinking=thinking)
        kwargs = dict(model=self.model, prompt=prompt, n=n, temperature=temperature, top_p=top_p,
                      max_tokens=max_tokens, logprobs=1, seed=seed, extra_body={"skip_special_tokens": True})
        if stop:
            kwargs["stop"] = stop
        start = time.monotonic()
        response = self.client.completions.create(**kwargs)
        elapsed = time.monotonic() - start
        samples = []
        for choice in response.choices:
            lp = choice.logprobs
            toks = list(lp.tokens) if lp and lp.tokens else []
            lps = [float(x) if x is not None else 0.0 for x in (lp.token_logprobs if lp and lp.token_logprobs else [])]
            samples.append(Sample(text=(choice.text or "").strip(), tokens=toks, logprobs=lps,
                                  finish_reason=choice.finish_reason or "", greedy=temperature == 0.0))
        with self._lock:
            u = self.usage
            u.calls += 1
            u.prompt_tokens += int(getattr(response.usage, "prompt_tokens", 0) or 0)
            u.completion_tokens += int(getattr(response.usage, "completion_tokens", 0) or 0)
            u.wall_seconds += elapsed
            p = u.by_purpose.setdefault(purpose, {"calls": 0, "completion_tokens": 0})
            p["calls"] += 1
            p["completion_tokens"] += int(getattr(response.usage, "completion_tokens", 0) or 0)
        return samples

    def propose(self, messages: list[dict], *, s: int = 8, temperature: float = 0.7, top_p: float = 0.9,
                max_tokens: int = 512, seed: int = 0, tools=None, purpose: str = "propose") -> list[Sample]:
        """S stochastic proposals plus one greedy proposal (greedy first)."""
        stochastic = self.chat(messages, n=s, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                               seed=seed, tools=tools, purpose=purpose)
        greedy = self.chat(messages, n=1, temperature=0.0, max_tokens=max_tokens, seed=seed, tools=tools, purpose=purpose + ":greedy")
        return greedy + stochastic

    def usage_dict(self) -> dict:
        return {"calls": self.usage.calls, "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens, "wall_seconds": self.usage.wall_seconds,
                "by_purpose": self.usage.by_purpose}
