#!/bin/bash
# SLURM variant of start_vllm.sh: one GPU job that serves the backbone and then runs a driver script on the same node.
# EDIT the #SBATCH lines for your cluster (account, partition, GPU type, memory, walltime).
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=rcc_worker
#SBATCH --output=rcc_worker_%j.log
# usage: sbatch scripts/worker.sbatch scripts/run_fanout.sh <seed> [atomic|official]
# Environment (export before sbatch, or set here): RCLUP_RUNS, RCLUP_MODEL_DIR, OPENAI_API_KEY, optional VLLM_* (see start_vllm.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
export PYTHONUNBUFFERED=1
MODEL=${RCLUP_MODEL_DIR:-Qwen/Qwen3-8B}; RUNS=${RCLUP_RUNS:-$HERE/runs}; PORT=$((20000 + SLURM_JOB_ID % 40000))
mkdir -p "$RUNS/logs"; STATUS=$RUNS/worker_${SLURM_JOB_ID}.json
# Optional: stage a local snapshot to node-local scratch when the shared file system is slow (set RCLUP_STAGE_DIR, e.g. $LOCAL).
if [ -n "${RCLUP_STAGE_DIR:-}" ] && [ -d "$MODEL" ]; then
  mkdir -p "$RCLUP_STAGE_DIR/qwen3-8b" && cp -rL "$MODEL"/. "$RCLUP_STAGE_DIR/qwen3-8b"/ && MODEL=$RCLUP_STAGE_DIR/qwen3-8b
fi
echo "{\"job_id\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"starting\"}" > "$STATUS"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-8b --port "$PORT" --host 0.0.0.0 \
  --dtype "${VLLM_DTYPE:-bfloat16}" --gpu-memory-utilization "${VLLM_UTIL:-0.85}" --max-model-len "${VLLM_MAXLEN:-16384}" \
  --enable-prefix-caching --max-num-seqs 96 --seed 0 --disable-log-requests > "$RUNS/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q qwen3-8b; then
    echo "{\"job_id\": \"$SLURM_JOB_ID\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"ready\"}" > "$STATUS"; echo "vLLM ready after ${i}x5s"; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then echo "vLLM died"; tail -30 "$RUNS/logs/vllm_${SLURM_JOB_ID}.log"; echo "{\"job_id\": \"$SLURM_JOB_ID\", \"status\": \"failed\"}" > "$STATUS"; exit 1; fi
  sleep 5
done
trap "kill $VPID 2>/dev/null; echo '{\"job_id\": \"$SLURM_JOB_ID\", \"status\": \"stopped\"}' > $STATUS" EXIT
export RCLUP_VLLM_URL=http://127.0.0.1:$PORT/v1
DRIVER=$1; shift; echo "running $DRIVER $*"; bash "$DRIVER" "$@"; echo "driver exited rc=$?"
