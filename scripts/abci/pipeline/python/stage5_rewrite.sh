#!/bin/bash
#PBS -q rt_HF
#PBS -N stage5
#PBS -l select=1
#PBS -l walltime=60:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage5_python

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

# Check if INDEX is provided
if [ -z "${INDEX:-}" ]; then
  echo "Error: INDEX variable is not set"
  exit 1
fi

# Keep numeric before padding
INDEX_INT=$((10#$INDEX))  # tolerate leading zeros
if [ "$INDEX_INT" -lt 0 ] || [ "$INDEX_INT" -gt 162 ]; then
  echo "Error: INDEX must be in [0,162], got $INDEX_INT"
  exit 1
fi
OTHER_INT=$((163 - INDEX_INT))

module load cuda/12.6/12.6.1

# Format INDEX to 4 digits with leading zeros
INDEX_PAD=$(printf "%04d" "$INDEX_INT")
OTHER_PAD=$(printf "%04d" "$OTHER_INT")

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

QUALITY=medium
MODEL_NAME=Qwen3-235B-A22B-Instruct-2507-FP8

INPUT_A="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage4/python/low_medium_high/train_${INDEX_PAD}_Qwen3-14B_${QUALITY}_Qwen3-14B.json"
INPUT_B="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage4/python/low_medium_high/train_${OTHER_PAD}_Qwen3-14B_${QUALITY}_Qwen3-14B.json"

OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage5/python/${MODEL_NAME}/${QUALITY}"
mkdir -p $OUTPUT_DIR


export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"

# --- shard A: INDEX (GPU 0-3) ---
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python src/pipeline.py rewrite \
  --input-jsonl $INPUT_A \
  --output-jsonl $OUTPUT_DIR/train_${INDEX_PAD}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --lang python \
  --batch-size 4096 \
  --tensor-parallel-size 4 &

PID_A=$!

# --- shard B: OTHER (GPU 4-7) ---
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python src/pipeline.py rewrite \
  --input-jsonl $INPUT_B \
  --output-jsonl $OUTPUT_DIR/train_${OTHER_PAD}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --lang python \
  --batch-size 4096 \
  --tensor-parallel-size 4 &

PID_B=$!

# Wait for both processes to finish
set +e
wait "$PID_A"; STATUS_A=$?
wait "$PID_B"; STATUS_B=$?
set -e

if [ $STATUS_A -ne 0 ] || [ $STATUS_B -ne 0 ]; then
  echo "Error: One or both processes failed"
  exit 1
fi

echo "Both processes completed successfully"
echo "Output files:"
echo "$OUTPUT_DIR/train_${INDEX}.jsonl"
echo "$OUTPUT_DIR/train_${OTHER}.jsonl"
echo "Job completed successfully"
exit 0
