#!/bin/bash
#PBS -q rt_HG
#PBS -N stage4
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/medical

set -e
cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles
module load cuda/12.8

# if [ -z "${INDEX:-}" ]; then
#   echo "Error: INDEX variable is not set"
#   exit 1
# fi

# INDEX=$(printf "%03d" $INDEX)

source .venv/bin/activate

# environment variables
export HF_HOME="/groups/gag51395/fujii/hf_cache"

MODEL_NAME=openai/gpt-oss-120b
REASONING_EFFORT="medium"
# vLLM Log: Maximum concurrency for 32,768 tokens per request: 51.68x
MAX_NUM_SEQS=50

DATASET_DIR=/groups/gag51395/datasets/raw/pretrain/pdf-dataset/medical_jsonl
INPUT_FILE_PATH="$DATASET_DIR/merged.jsonl"
OUTPUT_FILE_PATH="$DATASET_DIR/gpt-oss-120b/$REASONING_EFFORT/train_0002.jsonl"
mkdir -p $(dirname $OUTPUT_FILE_PATH)

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="$PWD:$PYTHONPATH"

# check
if [ ! -f $INPUT_FILE_PATH ]; then
  echo "Error: Input file $INPUT_FILE_PATH does not exist."
  exit 1
fi
if [ -f $OUTPUT_FILE_PATH ]; then
  echo "Error: Output file $OUTPUT_FILE_PATH already exists."
  exit 1
fi

export CUDA_VISIBLE_DEVICES="0"
python src/pipeline.py \
  --input-jsonl $INPUT_FILE_PATH \
  --output-jsonl $OUTPUT_FILE_PATH \
  --lang medical \
  --input-target-key text \
  --output-target-key conversation \
  --process-stage 4 \
  gpu \
  --model $MODEL_NAME \
  --tensor-parallel-size 1 \
  --model-max-length 32768 \
  --prompt-type stage4 \
  --gpu-backend vllm \
  --reasoning-effort $REASONING_EFFORT \
  --max-num-seqs $MAX_NUM_SEQS \
  --medical-data
