#!/bin/bash
#PBS -q rt_HG
#PBS -N stage2
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage2_python

set -e
cd $PBS_O_WORKDIR

# Load required modules
module load cuda/12.8/12.8.1
module load gcc

# Check if INDEX is provided
if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  echo "Usage: qsub -v INDEX=0002 stage2_parameterized.sh"
  exit 1
fi

# Format INDEX to 4 digits with leading zeros
INDEX=$(printf "%04d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source ./.trt/bin/activate

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage1/python/train_${INDEX}.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage2/python"
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"

mpirun --oversubscribe -np 1 python src/pipeline.py llm_auto_fix \
  --input-jsonl $INPUT_FILE_PATH \
  --output-dir $OUTPUT_DIR \
  --model "/groups/gag51395/hf_checkpoints/Qwen3-30B-A3B" \
  --model-max-length 32768 \
  --lang python \
  --batch-size 1024 \
  --tensor-parallel-size 1 \
  --code_key text \
  --lint_key lint_report
