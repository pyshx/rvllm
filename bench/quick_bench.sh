#!/bin/bash
# quick_bench.sh -- Simple throughput benchmark against a running OpenAI-compat server.
# Usage: bash quick_bench.sh <base_url> <model_name>
# Example: bash quick_bench.sh http://localhost:8000 /root/models/Qwen2.5-7B
set -euo pipefail

URL="${1:?usage: quick_bench.sh <url> <model>}"
MODEL="${2:?usage: quick_bench.sh <url> <model>}"
OUTPUT_LEN="${3:-128}"

echo "## Benchmark: ${MODEL}"
echo "- URL: ${URL}"
echo "- Output tokens: ${OUTPUT_LEN}"
echo ""

# Warmup
echo "Warmup (8 requests)..."
for i in $(seq 1 8); do
    curl -s -X POST "${URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"'"${MODEL}"'","prompt":"Hello world","max_tokens":'"${OUTPUT_LEN}"',"temperature":0.7}' \
        --max-time 60 >/dev/null &
done
wait
sleep 2
echo "Warmup done."
echo ""

echo "| N | tok/s | wall_ms | tokens | reqs |"
echo "|---|-------|---------|--------|------|"

for CONC in 1 4 16 32 64 128; do
    REQS=$((CONC * 4))
    [ $REQS -lt 16 ] && REQS=16

    TMPD=$(mktemp -d)
    START=$(date +%s%N)

    for i in $(seq 1 $REQS); do
        (
            RESP=$(curl -s -X POST "${URL}/v1/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"'"${MODEL}"'","prompt":"The meaning of life is","max_tokens":'"${OUTPUT_LEN}"',"temperature":0.7}' \
                --max-time 120)
            TOKS=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
            echo "$TOKS" > "${TMPD}/${i}"
        ) &

        # Throttle to concurrency
        while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do
            wait -n 2>/dev/null || true
        done
    done
    wait

    END=$(date +%s%N)
    WALL=$(( (END - START) / 1000000 ))

    TOK=0
    for f in "${TMPD}"/*; do
        read -r t < "$f"
        TOK=$((TOK + t))
    done
    rm -rf "$TMPD"

    TPS=0
    [ "$WALL" -gt 0 ] && TPS=$((TOK * 1000 / WALL))
    echo "| ${CONC} | ${TPS} | ${WALL} | ${TOK} | ${REQS} |"
done

echo ""
echo "Done."
