#!/bin/bash
#PBS -q rt_HF
#PBS -N stage8
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage8_python

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

# Format INDEX to 4 digits with leading zeros
INDEX=$(printf "%04d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

QUALITY=medium

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage7/python/medium_quality/train_${INDEX}_medium_Qwen3-235B-A22B.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage8/python/${QUALITY}"
mkdir -p $OUTPUT_DIR

MODEL_NAME=Qwen3-235B-A22B-Instruct-2507

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"
mpirun --oversubscribe -np 1 python src/pipeline.py second_rewrite \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train_${INDEX}_${MODEL_NAME}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --lang python \
  --tensor-parallel-size 8 \
  --code_key text_formatted
