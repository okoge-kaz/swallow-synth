#!/bin/bash
#PBS -q rt_HC
#PBS -N dataset
#PBS -l select=1:ncpus=32
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/dataset/merge_jsonl

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

INPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-cosmopedia/auto_math_text"
OUTPUT_FILE=${INPUT_DIR}/merged.jsonl

python src/tools/merge_jsonl.py \
  --input-dir $INPUT_DIR \
  --output-jsonl $OUTPUT_FILE \
  --key "generated_text" \
  --min-length 100
