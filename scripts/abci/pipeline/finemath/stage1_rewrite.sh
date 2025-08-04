#!/bin/bash
#PBS -q rt_HG
#PBS -N math_stage1
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage1_math

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

# Check if INDEX is provided
if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  exit 1
fi

module load cuda/12.6/12.6.1

# Format INDEX to 4 digits with leading zeros
INDEX=$(printf "%05d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage0/train-${INDEX}-of-00128.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage1"
mkdir -p $OUTPUT_DIR

MODEL_NAME=Qwen3-32B-FP8

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
uv run python src/pipeline.py finemath_rewrite \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train-${INDEX}-${MODEL_NAME}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --batch-size 4096 \
  --tensor-parallel-size 1
