#!/bin/bash
#PBS -q rt_HF
#PBS -N open_code_reasoning
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/open_code_reasoning_python

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

# Check if INDEX is provided
if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  echo "Usage: qsub -v INDEX=0002 stage5_rewrite.sh"
  exit 1
fi

module load cuda/12.6/12.6.1

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

MODEL_NAME=Qwen3-235B-A22B

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/competitive_programming/python/open_code_reasoning/train_${INDEX}.json"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/competitive_programming/python/open_code_reasoning-${MODEL_NAME}"
mkdir -p $OUTPUT_DIR


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
uv run python src/pipeline.py competitive_programming_write \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train_${INDEX}_${MODEL_NAME}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --lang python \
  --batch-size 4096 \
  --tensor-parallel-size 8
