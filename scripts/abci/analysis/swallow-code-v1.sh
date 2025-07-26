#!/bin/bash
#PBS -q rt_HC
#PBS -N analysis
#PBS -l select=1:ncpus=32
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/analysis

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

# EXP3
echo "EXP 3: linter filtering"
uv run python src/tools/swallow-code-v1/text_length_analysis.py \
  --input-dir /groups/gag51395/datasets/raw/pretrain/swallow-code/ablation/exp3-linter-filtered/jsonl \
  --analysis-key "text" \

# EXP5: SGCR
echo "EXP 5: SGCR"
uv run python src/tools/swallow-code-v1/text_length_analysis.py \
  --input-dir /groups/gag51395/datasets/raw/pretrain/swallow-code/ablation/exp5-sgcr/jsonl \
  --analysis-key "text"

# EXP11 SCOR
echo "EXP 11: SCOR"
uv run python src/tools/swallow-code-v1/text_length_analysis.py \
  --input-dir /groups/gag51395/datasets/raw/pretrain/swallow-code/ablation/exp11-scor/jsonl \
  --analysis-key "text"

