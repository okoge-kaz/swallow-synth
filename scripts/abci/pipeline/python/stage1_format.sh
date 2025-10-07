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
#PBS -o outputs/pipeline/stage1_python

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INDEX=8

INPUT_FILE_DIR=/groups/gag51395/datasets/raw/pretrain/the-stack-v2/the-stack-v2/data/Python/train-0000$INDEX-of-00009.parquet/train-0000$INDEX-of-00009
OUTPUT_FILE_DIR=/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage1/python

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

  # file index calc: INDEX * 18 + FILE_NUMBER
  FILE_INDEX=$(($INDEX * 18 + $FILE_NUMBER))

  OUTPUT_FILE=${OUTPUT_FILE_DIR}/train_$(printf "%04d" $FILE_INDEX).jsonl
  mkdir -p $(dirname $OUTPUT_FILE)

  echo "Input: $FILE"
  echo "Output: $OUTPUT_FILE (INDEX=$INDEX, FILE_NUMBER=$FILE_NUMBER, FILE_INDEX=$FILE_INDEX)"

  export PYTHONPATH="$PWD:$PYTHONPATH"
  mpirun --oversubscribe -np 1 python src/pipeline.py auto_format \
    --input-jsonl $FILE \
    --output-jsonl $OUTPUT_FILE \
    --workers 16 \
    --lang python \
    --batch-size 256
done
