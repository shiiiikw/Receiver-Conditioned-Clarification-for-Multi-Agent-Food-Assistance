"""Drop records whose error matches a regex (default: receiver parse failures; use --pattern APIConnectionError for server
outages), so the next run_episodes resumes exactly those episodes.  usage: python scripts/strip_errors.py [--pattern REGEX] <out_dir> [...]"""
import json, os, re, shutil, sys
args = sys.argv[1:]
pat = r"No valid proposals for (logistics|access|schedule)#"
if args and args[0] == "--pattern":
    pat, args = args[1], args[2:]
rx = re.compile(pat)
for d in args:
    p = os.path.join(d, "records.jsonl")
    if not os.path.exists(p):
        continue
    rows = [l for l in open(p)]
    good = [l for l in rows if not rx.search(json.loads(l).get("error", "") or "")]
    if len(good) != len(rows):
        shutil.copy(p, p + ".bak_nvp")
        open(p, "w").writelines(good)
        s = os.path.join(d, "summary.json")
        if os.path.exists(s):
            os.remove(s)
    print(f"{d}: kept {len(good)}/{len(rows)}")
