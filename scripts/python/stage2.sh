#!/bin/bash
#PBS -q rt_HC
#PBS -N stage2
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/python/stage2

set -e
cd $PBS_O_WORKDIR

module load hpcx/2.20

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

DATASET_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/swallow/swallow-code-v2
INPUT_DIR="$DATASET_DIR/stage1-auto-format/python/"
OUTPUT_DIR="$DATASET_DIR/stage2-length-filter/python/"
mkdir -p $OUTPUT_DIR

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"

MIN_INDEX=1
MAX_INDEX=162

for ((i=MIN_INDEX; i<=MAX_INDEX; i++)); do
  INDEX=$(printf "%04d" $i)

  INPUT_FILE="$INPUT_DIR/train_${INDEX}.jsonl"
  OUTPUT_FILE="$OUTPUT_DIR/train_${INDEX}.jsonl"

  if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file not found: $INPUT_FILE"
    continue
  fi

  if mpirun --oversubscribe -np 1 python src/pipeline.py cpu \
      --input-jsonl $INPUT_FILE \
      --output-jsonl $OUTPUT_FILE \
      --input-target-key text \
      --output-target-key text \
      --process-stage 2 \
      --filter-threshold-length 40960 \
      --num-cpu-workers 16 \
      --lang python \
      --read-batch-size 4096; then
    echo "Successfully processed $INPUT_FILE"
  else
    echo "Failed to process $INPUT_FILE"
  fi
done
