#!/bin/bash
#PBS -q rt_HC
#PBS -N stage1
#PBS -l select=1:ncpus=32
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/wikipedia

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

python src/tools/wikipedia/english_wiki.py \
  --input-dir /groups/gag51395/datasets/raw/pretrain/wikipedia/raw/en_wiki \
  --output-file /groups/gag51395/datasets/raw/pretrain/wikipedia/processed/en_wikipedia.jsonl
