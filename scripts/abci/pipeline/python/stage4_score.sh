#!/bin/bash
#PBS -q rt_HF
#PBS -N stage4
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage4_python

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"Add commentMore actions
cat $PBS_NODEFILE

# Load required modules
module load cuda/12.8/12.8.1
module load gcc

# Check if INDEX is provided
if [ -z "$INDEX" ]; then
  echo "Error: INDEX variable is not set"
  echo "Usage: qsub -v INDEX=0002 stage4_score.sh"
  exit 1
fi

# Format INDEX to 4 digits with leading zeros
INDEX=$(printf "%04d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage3/python/train_${INDEX}.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage4/python"
mkdir -p $OUTPUT_DIR

MODEL_NAME=Qwen3-14B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"

mpirun --oversubscribe -np 1 python src/pipeline.py llm_scoring \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train_${INDEX}_${MODEL_NAME}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --model-max-length 32768 \
  --lang python \
  --tensor-parallel-size 8 \
  --code-key text_formatted
