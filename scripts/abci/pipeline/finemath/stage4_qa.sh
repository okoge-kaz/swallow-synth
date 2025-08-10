#!/bin/bash
#PBS -q rt_HG
#PBS -N math_stage4
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage4_math

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

MODEL_NAME=Qwen3-30B-A3B-Instruct-2507-FP8

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage2/train-${INDEX}-Qwen3-32B-FP8.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage4/${MODEL_NAME}"
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
python src/pipeline.py finemath_rewrite \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train-${INDEX}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --batch-size 4096 \
  --tensor-parallel-size 1 \
  --prompt-type "question-answer"
