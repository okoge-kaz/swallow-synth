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
#PBS -o outputs/python/stage1

set -e
cd $PBS_O_WORKDIR

module load hpcx/2.20

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

QUALITY=medium

DATASET_DIR=/groups/gch51639/fujii/datasets/raw/pretrain/swallow/swallow-code-v2
INPUT_FILE_DIR="$DATASET_DIR/stage4-llm-rewrite/python/$QUALITY"
OUTPUT_DIR="$DATASET_DIR/stage5-auto-format/python/${QUALITY}"
mkdir -p $OUTPUT_DIR

FILE_LIST=($(ls $INPUT_FILE_DIR/train_*.jsonl))
for FILE_PATH in ${FILE_LIST[@]}; do
  FILE_NAME=$(basename $FILE_PATH)
  OUTPUT_FILE_PATH="$OUTPUT_DIR/$FILE_NAME"

  echo "Processing $FILE_NAME ..."
  mpirun --oversubscribe -np 1 python src/pipeline.py \
    --input-jsonl $FILE_PATH \
    --output-jsonl $OUTPUT_FILE_PATH \
    --lang python \
    --input-target-key text \
    --output-target-key text \
    --process-stage 5 \
    cpu \
    --num-cpu-workers 32 \
    --read-batch-size 4096 \
    --tmp-dir /local/$PBS_JOBID
done
