#!/bin/bash
# Full evaluation of the fan-out system and the clarification baselines on HoosierHelp for one seed.
#   usage: scripts/run_fanout.sh <seed> [atomic|official]
#   atomic   (default) budgeted atomic questions: --variant atomic --intake-policy atomic --max-turns 8, ten controllers
#   official HoosierHelp protocol: --variant native --intake-policy native (fixed multi-item intake questions), six controllers
# Prerequisites: backbone server reachable (RCLUP_VLLM_URL or a worker status file), OPENAI_API_KEY, user specs built.
# Outputs: $RCLUP_RUNS/eval/<protocol>/seed<seed>/hoosier_<controller>_b5/{records.jsonl,summary.json}; logs under $RCLUP_RUNS/logs.
# A 6-episode smoke run is gated by scripts/fanout2_gate.py before the full runs start.  Re-running resumes unfinished runs.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
S=${1:?seed}; PROTO=${2:-atomic}; RUNS=${RCLUP_RUNS:-$HERE/runs}; LOGS=$RUNS/logs; mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1 RCLUP_RUNS=$RUNS
CAP=${RCLUP_CAP_USD:-450}          # hard cap on the shared OpenAI ledger total (USD)
WORKERS=${RCLUP_WORKERS:-10}       # concurrent episodes
B5=normal,rambling,impatience,self_contradictory,unsupported_request
if [ "$PROTO" = official ]; then
  COMMON="--variant native --intake-policy native --behaviors $B5 --workers $WORKERS --seed $S --cap-usd $CAP"
  CONTROLLERS=${RCLUP_CONTROLLERS:-"voi_det procot atd sage voi_llm text_mas"}
else
  COMMON="--variant atomic --intake-policy atomic --max-turns 8 --behaviors $B5 --workers $WORKERS --seed $S --cap-usd $CAP"
  CONTROLLERS=${RCLUP_CONTROLLERS:-"voi_det disagreement_ask max_uncertainty fixed_order voi_myopic text_mas procot atd sage voi_llm"}
fi
step() { local name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; "$@" > "$LOGS/$name.log" 2>&1; local rc=$?
         echo "[$(date +%H:%M:%S)] END $name rc=$rc $(grep -E '"end_to_end_match"' "$LOGS/$name.log" | tail -1 | cut -c1-80)"; return $rc; }
SMOKE=smoke/$PROTO/seed$S/voi_det
if [ ! -f "$RUNS/$SMOKE/GATE_OK" ]; then
  rm -rf "$RUNS/$SMOKE"
  step smoke_${PROTO}_s$S python -m rclup.run_episodes $COMMON --controller voi_det --limit 6 --out "$SMOKE" || { echo "SMOKE FAILED"; tail -40 "$LOGS/smoke_${PROTO}_s$S.log"; exit 1; }
  python scripts/fanout2_gate.py "$RUNS/$SMOKE" || { echo "SMOKE GATE FAILED (see above)"; exit 1; }
  touch "$RUNS/$SMOKE/GATE_OK"
fi
E=eval/$PROTO/seed$S
for C in $CONTROLLERS; do
  [ -f "$RUNS/$E/hoosier_${C}_b5/summary.json" ] && { echo "[$(date +%H:%M:%S)] SKIP $C (done)"; continue; }
  step ${PROTO}_${C}_s$S python -m rclup.run_episodes $COMMON --controller $C --limit 240 --out "$E/hoosier_${C}_b5"
done
echo "[$(date +%H:%M:%S)] SEED $S ($PROTO) DONE"
