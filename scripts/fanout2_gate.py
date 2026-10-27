"""Smoke gate for the fan-out topology: every episode must carry intake + three receiver invocations + a fan-out joint packet, no errors,
and the inference budget must not have degraded most invocations. usage: python scripts/fanout2_gate.py <run_dir>"""
import json, sys, collections
d = sys.argv[1]
recs = [json.loads(l) for l in open(f"{d}/records.jsonl")]
errs = [r for r in recs if "error" in r]
bad, stats = [], collections.Counter()
for r in recs:
    kinds = collections.Counter(e["kind"] for e in r.get("trace", []))
    roles = collections.Counter(e.get("role") for e in r.get("trace", []) if e["kind"] == "invocation")
    joint_rev = [e["packet"]["extra"].get("fanout_rev") for e in r.get("trace", []) if e["kind"] == "invocation" and e.get("role") == "joint"]
    stats["receiver_checked"] += kinds["receiver_checked"]; stats["receiver_reused"] += kinds["receiver_reused"]
    stats["budget_degraded"] += kinds["budget_degraded"]; stats["invocations"] += kinds["invocation"]
    stats["corrected"] += sum(e.get("corrected", 0) for e in r.get("trace", []) if e["kind"] == "receiver_checked")
    if "error" in r or not joint_rev or any(v != 2 for v in joint_rev) or roles["intake"] == 0 or any(roles[x] == 0 for x in ("access", "schedule", "logistics")):
        bad.append((r.get("root_id"), r.get("error", "")[:120], dict(roles), joint_rev))
print(json.dumps({"n": len(recs), "errors": len(errs), "bad": len(bad), **stats}))
for b in bad[:5]:
    print("BAD", b)
sys.exit(1 if (bad or not recs or stats["budget_degraded"] > stats["invocations"] // 2) else 0)
