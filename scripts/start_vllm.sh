#!/bin/bash
# Start the frozen Qwen3-8B backbone as a vLLM OpenAI-compatible server on this machine and publish its address for the
# drivers (RCLUP_RUNS/worker_local.json).  Needs one GPU with >= 24 GB (bf16 weights ~16 GB + KV cache).
# Configure through environment variables (defaults in brackets):
#   RCLUP_MODEL_DIR  [Qwen/Qwen3-8B]   HF id or local snapshot directory
#   RCLUP_RUNS       [<repo>/runs]     output root (the status file is written here)
#   VLLM_PORT        [8000]            server port
#   VLLM_UTIL        [0.85]            --gpu-memory-utilization
#   VLLM_MAXLEN      [16384]           --max-model-len
#   VLLM_DTYPE       [bfloat16]        use float16 on V100-class GPUs
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${RCLUP_MODEL_DIR:-Qwen/Qwen3-8B}; RUNS=${RCLUP_RUNS:-$HERE/runs}; PORT=${VLLM_PORT:-8000}
mkdir -p "$RUNS/logs"; STATUS=$RUNS/worker_local.json
echo "{\"job_id\": \"local\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"starting\"}" > "$STATUS"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-8b --port "$PORT" --host 0.0.0.0 \
  --dtype "${VLLM_DTYPE:-bfloat16}" --gpu-memory-utilization "${VLLM_UTIL:-0.85}" --max-model-len "${VLLM_MAXLEN:-16384}" \
  --enable-prefix-caching --max-num-seqs 96 --seed 0 --disable-log-requests > "$RUNS/logs/vllm_local.log" 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q qwen3-8b; then
    echo "{\"job_id\": \"local\", \"host\": \"$(hostname)\", \"port\": $PORT, \"status\": \"ready\"}" > "$STATUS"
    echo "vLLM ready on port $PORT (pid $VPID); drivers find it through $STATUS or RCLUP_VLLM_URL=http://$(hostname):$PORT/v1"; break
  fi
  if ! kill -0 $VPID 2>/dev/null; then echo "vLLM died:"; tail -30 "$RUNS/logs/vllm_local.log"; echo "{\"job_id\": \"local\", \"status\": \"failed\"}" > "$STATUS"; exit 1; fi
  sleep 5
done
wait $VPID
