#!/bin/bash
#PBS -q rt_HG
#PBS -N stage4
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/python/stage4

set -e
cd $PBS_O_WORKDIR

if [ -z "${INDEX:-}" ]; then
  echo "Error: INDEX variable is not set"
  exit 1
fi

module load cuda/12.8/12.8.1
module load hpcx/2.20

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

QUALITY=medium
MODEL_NAME=Qwen3-235B-A22B-Instruct-2507-FP8

DATASET_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/swallow/swallow-code-v2
INPUT_FILE_PATH="$DATASET_DIR/stage3-llm-score/python/train_${INDEX}.jsonl"
OUTPUT_FILE_PATH="$DATASET_DIR/stage4-llm-rewrite/python/$QUALITY/train_${INDEX}.jsonl"
mkdir -p $(dirname $OUTPUT_FILE_PATH)

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES="0"
mpirun --oversubscribe -np 1 python src/pipeline.py gpu \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_FILE_PATH \
  --lang python \
  --input-target-key text \
  --output-target-key text \
  --process-stage 4 \
  --model $MODEL_NAME \
  --tensor-parallel-size 1 \
  --model-max-length 40960 \
  --prompt-type stage4 \
  --gpu-backend vllm
