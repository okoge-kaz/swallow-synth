#!/bin/bash
#PBS -q rt_HF
#PBS -N cosmopedia
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/cosmopedia

set -e
cd $PBS_O_WORKDIR

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

INDEX=0

# Format INDEX to 5 digits with leading zeros
INDEX=$(printf "%05d" $INDEX)

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INPUT_FILE_PATH="/groups/gag51395/datasets/raw/pretrain/cosmopedia/data/stanford-jsonl/train-${INDEX}-of-00013.jsonl"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-cosmopedia/stanford"
mkdir -p $OUTPUT_DIR

MODEL_NAME=Qwen3-235B-A22B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
uv run python src/cosmopedia_pipeline.py \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_DIR/train_${INDEX}_${MODEL_NAME}.jsonl \
  --model "/groups/gag51395/hf_checkpoints/${MODEL_NAME}" \
  --batch-size 1024 \
  --tensor-parallel-size 8 \
  --disable-thinking
