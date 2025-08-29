#!/bin/bash
#PBS -q rt_HG
#PBS -N wiki_qa
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/wikipedia_qa

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

INDEX=14

INPUT_FILE_PATH=/groups/gag51395/datasets/raw/pretrain/wikipedia/processed/ja/ja_wikipedia_$INDEX.jsonl
OUTPUT_FILE_PATH=/groups/gag51395/datasets/raw/pretrain/wikipedia/qa_choices/ja_wikipedia_qa_choices_$INDEX.jsonl

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0"

python src/wikipedia_qa.py \
  --model-path /groups/gag51395/hf_checkpoints/$MODEL_NAME \
  --input-jsonl "$INPUT_FILE_PATH" \
  --output-jsonl "$OUTPUT_FILE_PATH" \
  --qa-mode "free" \
  --lang "ja" \
  --chunk-max-tokens 1024 \
  --gen-max-tokens 512 \
  --batch-size 1000 \
  --tensor-parallel-size 1
