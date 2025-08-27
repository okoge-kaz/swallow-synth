#!/bin/bash
#PBS -q rt_HF
#PBS -N math_textbook_style
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/vllm_server/math/textbook_style

set -e
cd $PBS_O_WORKDIR
echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

module load cuda/12.6/12.6.1
source .venv/bin/activate

export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"

if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  exit 1
fi
INDEX=$(printf "%05d" $INDEX)

INPUT_JSONL="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage2/train-${INDEX}-Qwen3-32B-FP8.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/textbook-style"
mkdir -p $OUTPUT_DIR
OUTPUT_JSONL="${OUTPUT_DIR}/train-${INDEX}.jsonl"

vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 \
   --tensor-parallel-size 8 \
   --enable-expert-parallel \
   --enable-chunked-prefill \
   --gpu-memory-utilization 0.95 \
   --dtype auto \
   --port 8000 &

SERVER_PID=$!

echo "vLLM server started with PID: $SERVER_PID"

# Wait for server to be ready with proper health check
echo "Waiting for vLLM server to be ready..."
for i in {1..70}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "Attempt $i: Server not ready yet, waiting 10 seconds..."
    sleep 10
done

# Final health check
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server failed to start properly"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "Server successfully started and ready to accept requests"

# Execute math rewrite request
echo "Starting math rewrite processing..."
python src/request.py math_rewrite \
    --input-jsonl "${INPUT_JSONL}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --base-url "http://localhost:8000" \
    --prompt-type "${PROMPT_TYPE:-text-book-style}" \
    --batch-size "${BATCH_SIZE:-4096}" \
    --model-max-length "${MODEL_MAX_LENGTH:-262144}"

REQUEST_EXIT_CODE=$?

echo "Math rewrite processing completed with exit code: $REQUEST_EXIT_CODE"

# Gracefully shut down the server
echo "Shutting down vLLM server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "vLLM server shut down successfully"

exit $REQUEST_EXIT_CODE



