#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/cpfs/sprite/workspace/plan_sft/LlamaFactory"
VLLM="/mnt/data/cpfs/sprite/workspace/plan_sft/vllm-0.11/bin/vllm"
RUN_DIR="$ROOT/llamafactory_runs/gemma3-270m-full-sft_260919_v1"
MODEL="$ROOT/saves/gemma3-270m/full/sft"
LOG_FILE="$RUN_DIR/vllm-sft.log"
PID_FILE="$RUN_DIR/vllm-sft.pid"

start() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "vLLM is already running, PID=$(cat "$PID_FILE")"
        exit 0
    fi

    cd "$ROOT"

    nohup setsid env \
        -u VLLM_USE_MODELSCOPE \
        -u USE_MODELSCOPE_HUB \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        "$VLLM" serve "$MODEL" \
        --host 127.0.0.1 \
        --port 8001 \
        --served-model-name gemma3-270m-sft \
        --dtype bfloat16 \
        --max-model-len 4096 \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 \
        --gpu-memory-utilization 0.2 \
        --max-num-seqs 256 \
        --enable-prefix-caching \
        --generation-config vllm \
        > "$LOG_FILE" 2>&1 < /dev/null &

    echo $! > "$PID_FILE"
    echo "Started vLLM, PID=$(cat "$PID_FILE")"
    echo "Log: $LOG_FILE"
    echo "API: http://127.0.0.1:8001/v1"
}

stop() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "PID file not found"
        exit 0
    fi

    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -- "-$pid"
        echo "Stopped vLLM process group $pid"
    else
        echo "vLLM is not running"
    fi
    rm -f "$PID_FILE"
}

status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "running, PID=$(cat "$PID_FILE")"
        curl -sf http://127.0.0.1:8001/health >/dev/null \
            && echo "API ready" \
            || echo "API is still starting"
    else
        echo "stopped"
    fi
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    logs) tail -f "$LOG_FILE" ;;
    *) echo "Usage: $0 {start|stop|restart|status|logs}"; exit 1 ;;
esac
