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
#PBS -o outputs/c/stage1

set -e
cd $PBS_O_WORKDIR

module load hpcx/2.20

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INDEX=0

INPUT_FILE_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/public/the-stack-v2/data/C/train-0000$INDEX-of-00004.parquet/train-0000$INDEX-of-00004
OUTPUT_FILE_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/swallow/swallow-code-v2/stage1-auto-format/c

for FILE in $(ls $INPUT_FILE_DIR/*.jsonl); do
  echo "Processing $FILE"

  #（files_0007.jsonl -> files_0007）
  BASE_NAME=$(basename "$FILE" .jsonl)

  # （ex: 0007 -> 7）
  if [[ $BASE_NAME =~ files_([0-9]+) ]]; then
    FILE_NUMBER=${BASH_REMATCH[1]}

    FILE_NUMBER=$((10#$FILE_NUMBER))
  else
    echo "Warning: Could not extract file number from $BASE_NAME"
    FILE_NUMBER=0
  fi

  FILE_INDEX=$(($INDEX * 17 + $FILE_NUMBER))

  OUTPUT_FILE=${OUTPUT_FILE_DIR}/train_$(printf "%04d" $FILE_INDEX).jsonl
  mkdir -p $(dirname $OUTPUT_FILE)

  echo "Input: $FILE"
  echo "Output: $OUTPUT_FILE (INDEX=$INDEX, FILE_NUMBER=$FILE_NUMBER, FILE_INDEX=$FILE_INDEX)"

  export PYTHONPATH="$PWD:$PYTHONPATH"
  mpirun --oversubscribe -np 1 python src/pipeline.py \
    --input-jsonl $FILE \
    --output-jsonl $OUTPUT_FILE \
    --input-target-key text \
    --output-target-key text \
    --process-stage 1 \
    --lang c \
    cpu \
    --num-cpu-workers 16 \
    --read-batch-size 1024
done
