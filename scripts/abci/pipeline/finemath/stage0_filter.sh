#!/bin/bash
#PBS -q rt_HF
#PBS -N math_stage0
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage0_math

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

INPUT_DIR="/groups/gag51395/datasets/raw/pretrain/finemath/finemath-3plus-jsonl/"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-math-v2/stage0"
mkdir -p $OUTPUT_DIR

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"
uv run python src/tools/math_length_filter.py \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --tokenizer "/groups/gag51395/hf_checkpoints/Qwen3-32B-FP8"
