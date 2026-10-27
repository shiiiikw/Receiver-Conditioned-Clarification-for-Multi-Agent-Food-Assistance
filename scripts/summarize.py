"""Print an end-to-end match table (controller x seed) for a protocol directory.  usage: python scripts/summarize.py runs/eval/atomic"""
import json, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
table, seeds = defaultdict(dict), set()
for f in sorted(root.glob("seed*/hoosier_*_b5/summary.json")):
    d = json.loads(f.read_text()); seed = f.parts[-3][4:]; ctrl = f.parts[-2][8:-3]
    table[ctrl][seed] = d; seeds.add(seed)
seeds = sorted(seeds)
print(f"{'controller':18s} " + " ".join(f"{'e2e s' + s:>9s} {'asks':>5s} {'err':>4s}" for s in seeds))
for ctrl in sorted(table, key=lambda c: -max(d.get("end_to_end_match", 0) for d in table[c].values())):
    cells = []
    for s in seeds:
        d = table[ctrl].get(s)
        cells.append(f"{d['end_to_end_match']:9.3f} {d['user_questions_avg']:5.2f} {d['errors']:4d}" if d and "end_to_end_match" in d else f"{'-':>9s} {'-':>5s} {'-':>4s}")
    print(f"{ctrl:18s} " + " ".join(cells))
