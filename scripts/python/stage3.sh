#!/bin/bash
#PBS -q rt_HG
#PBS -N stage3
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/python/stage3

set -e
cd $PBS_O_WORKDIR

# Load required modules
module load cuda/12.8/12.8.1
module load gcc
module load hpcx/2.20

if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  exit 1
fi
INDEX=$(printf "%04d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .trt/bin/activate

DATASET_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/swallow/swallow-code-v2
INPUT_FILE_PATH="$DATASET_DIR/stage2-length-filter/python/train_${INDEX}.jsonl"
OUTPUT_FILE_PATH="$DATASET_DIR/stage3-llm-score/python/train_${INDEX}.jsonl"
mkdir -p $(dirname $OUTPUT_FILE_PATH)

MODEL_NAME=Qwen3-14B
MODEL_PATH="/groups/gag51395/hf_checkpoints/${MODEL_NAME}"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"

mpirun --oversubscribe -np 1 python src/pipeline.py \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_FILE_PATH \
  --lang python \
  --input-target-key text \
  --output-target-key score \
  --process-stage 3 \
  gpu \
  --model $MODEL_PATH \
  --tensor-parallel-size 1 \
  --model-max-length 32768 \
  --medium-score-threshold 3 \
  --high-score-threshold 7 \
  --gpu-backend tensorrt-llm
