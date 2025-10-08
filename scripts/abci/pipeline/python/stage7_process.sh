#!/bin/bash
#PBS -q rt_HF
#PBS -N stage7
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage7_python

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage6/python/medium_quality/Qwen3-235B-A22B-Instruct-2507"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage7/python/medium_quality/Qwen3-235B-A22B-Instruct-2507"
mkdir -p $OUTPUT_DIR

export PYTHONPATH="$PWD:$PYTHONPATH"
mpirun --oversubscribe -np 1 python src/pipeline.py filter_rewritten_code \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR
