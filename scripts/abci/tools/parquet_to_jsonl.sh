#!/bin/bash
#PBS -q rt_HC
#PBS -N llama-3.1-8b
#PBS -l select=1:ncpus=32
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/dataset/convert_parquet_to_jsonl

set -e
cd $PBS_O_WORKDIR

# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

PROGRAMMING_LANGUAGE="Python"

PARQUET_FILE_DIR="/groups/gag51395/datasets/raw/pretrain/the-stack-v2/the-stack-v2/data/${PROGRAMMING_LANGUAGE}"
JSONL_FILE_DIR="/groups/gag51395/datasets/raw/pretrain/the-stack-v2-jsonl/${PROGRAMMING_LANGUAGE}"
mkdir -p "${JSONL_FILE_DIR}"

# Convert Parquet files to JSONL
for PARQUET_FILE in "${PARQUET_FILE_DIR}"/*.parquet; do
    BASENAME=$(basename "${PARQUET_FILE}" .parquet)
    JSONL_FILE="${JSONL_FILE_DIR}/${BASENAME}.jsonl"

    echo "Converting ${PARQUET_FILE} to ${JSONL_FILE}"

    uv run python src/common/parquet_to_jsonl.py \
        --parquet-file "${PARQUET_FILE}" \
        --jsonl-file "${JSONL_FILE}"
    echo "Conversion completed: ${JSONL_FILE}"
    echo "----------------------------------------"
done
