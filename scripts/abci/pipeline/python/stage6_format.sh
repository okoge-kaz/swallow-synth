#!/bin/bash
#PBS -q rt_HC
#PBS -N stage6
#PBS -l select=1:ncpus=32
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage6_python

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

# Check if INDEX is provided
if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  echo "Usage: qsub -v INDEX=0002 stage6_format.sh"
  exit 1
fi
INDEX=$(printf "%04d" $INDEX)
QUALITY=medium

INPUT_FILE=/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage5/python/Qwen3-235B-A22B-Instruct-2507-FP8/${QUALITY}/train_${INDEX}.jsonl
OUTPUT_FILE_DIR=/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage6/python/${QUALITY}_quality/Qwen3-235B-A22B-Instruct-2507-FP8

OUTPUT_FILE=${OUTPUT_FILE_DIR}/train_${INDEX}.jsonl
mkdir -p $OUTPUT_FILE_DIR

export PYTHONPATH="$PWD:$PYTHONPATH"
uv run python src/pipeline.py format_check \
  --input-jsonl $INPUT_FILE \
  --output-jsonl $OUTPUT_FILE \
  --workers 32 \
  --lang python \
  --batch-size 1024 \
  --target-key "improved_code"
