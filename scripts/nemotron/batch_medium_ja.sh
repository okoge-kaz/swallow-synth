#!/bin/bash

# code
for INDEX in {0..37}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/swallow/Qwen3-Swallow-SFT/nemotron-post-training-v1-ja/code/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/code/code-ja-reasoning-medium.sh
done

# # math
for INDEX in {0..40}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/swallow/Qwen3-Swallow-SFT/nemotron-post-training-v1-ja/math/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/math/math-ja-reasoning-medium.sh
done

# stem
for INDEX in {1..49}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/swallow/Qwen3-Swallow-SFT/nemotron-post-training-v1-ja/science/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gag51395 -v INDEX=$INDEX scripts/nemotron/science/science-ja-reasoning-medium.sh
done
