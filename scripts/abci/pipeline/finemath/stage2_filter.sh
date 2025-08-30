#!/bin/bash
#PBS -q rt_HF
#PBS -N math_stage2
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage2_math

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

INPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/textbook-style"
OUTPUT_DIR="$INPUT_DIR/filtered"
mkdir -p $OUTPUT_DIR

uv run python src/tools/finemath/filter_finemath.py \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR
