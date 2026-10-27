"""Generate HoosierHelp user specs for training and held-out evaluation.

Evaluation specs: the native default configuration (main.py: sample_seed=1,
sample_count=48), i.e. the benchmark's own 48 base users. Training specs are
generated with a different seed and any spec sharing a ground-truth resource
with an evaluation spec (or an earlier training spec) is dropped, so target
resources are disjoint across splits. Training roots are hashed into fixed
partitions. Specs are simulator/evaluator inputs only; agents never see them.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from ..common import DATA, VENDOR_HOOSIER, ensure_vendor_on_path, partition, write_json, write_jsonl

HOOSIER_COMMIT = "7f6b5ec955bcb4b3ba861fcd86f80269b61f692a"


def resources_of(spec):
    return {need["ground_truth_resource_id"] for need in spec["needs"]}


def build(out_dir: Path, eval_seed=1, eval_count=48, train_seed=7777, train_count=640):
    ensure_vendor_on_path()
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(VENDOR_HOOSIER), "rev-parse", "HEAD"], text=True).strip()
    if commit != HOOSIER_COMMIT:
        raise RuntimeError(f"Unexpected HoosierHelp revision {commit}")
    from eval.spec_generation import build_user_specs
    from tools.indiana211 import load_resource_index
    index = load_resource_index(VENDOR_HOOSIER / "data/benchmark/filtered_resources_tagged.csv")
    evaluation = build_user_specs(index.resources, count=eval_count, seed=eval_seed, progress_every=0)
    for spec in evaluation:
        spec["root_id"] = f"hoosier:eval:{spec['user_spec_id']}"
        spec["split"] = "test"
    reserved = set().union(*(resources_of(s) for s in evaluation))
    pool = build_user_specs(index.resources, count=train_count, seed=train_seed, progress_every=0)
    training, dropped = [], collections.Counter()
    seen_needs = set()
    for spec in pool:
        ids = resources_of(spec)
        key = tuple(sorted((n["service_categories"][0], n["ground_truth_resource_id"]) for n in spec["needs"]))
        if ids & reserved:
            dropped["shares_resource_with_eval_or_train"] += 1
            continue
        if key in seen_needs:
            dropped["duplicate"] += 1
            continue
        seen_needs.add(key)
        reserved |= ids
        spec["root_id"] = f"hoosier:train:{spec['user_spec_id']}"
        spec["split"] = partition("hoosier", spec["root_id"])
        training.append(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eval_specs.jsonl", evaluation)
    write_jsonl(out_dir / "train_specs.jsonl", training)
    by = collections.Counter((s["split"], s["case_type"], s["constraint_profile"]) for s in training + evaluation)
    summary = {"evaluation": len(evaluation), "training": len(training), "dropped": dict(dropped),
               "by_split_case_profile": {"|".join(k): v for k, v in sorted(by.items())},
               "eval_seed": eval_seed, "train_seed": train_seed, "hoosier_commit": commit}
    write_json(out_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA / "hoosier"))
    p.add_argument("--train-count", type=int, default=640)
    a = p.parse_args()
    print(build(Path(a.out), train_count=a.train_count))
