#!/bin/bash
#PBS -q rt_HG
#PBS -N code_question
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/code_question

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

module load cuda/12.6/12.6.1

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate
MODEL_NAME=Qwen3-30B-A3B-Instruct-2507-FP8

INDEX=60
INDEX=$(printf "%04d" $INDEX)

MODE=competitive

INPUT_FILE_PATH=/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/question/python/medium_quality/Qwen3-30B-A3B-Instruct-2507-FP8/sample.jsonl
OUTPUT_FILE_PATH=/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/$MODE-question/python/medium_quality/Qwen3-30B-A3B-Instruct-2507-FP8/sample.jsonl
mkdir -p $(dirname $OUTPUT_FILE_PATH)

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"

python src/change_question.py \
  --model-path /groups/gag51395/hf_checkpoints/$MODEL_NAME \
  --input-jsonl "$INPUT_FILE_PATH" \
  --output-jsonl "$OUTPUT_FILE_PATH" \
  --gen-max-tokens 16384 \
  --batch-size 10 \
  --tensor-parallel-size 1
