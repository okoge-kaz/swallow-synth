#!/bin/bash

# code
for INDEX in {0..632}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/public/Nemotron-Post-Training-Dataset-v1/code-jsonl/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/nemotron/code/code-en-reasoning-medium.sh
done

# # math
for INDEX in {0..681}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/public/Nemotron-Post-Training-Dataset-v1/math-jsonl/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/nemotron/math/math-en-reasoning-medium.sh
done

# stem
for INDEX in {1..999}
do
  OUTPUT_DIR=/groups/gch51639/fujii/datasets/raw/instruct/swallow/Qwen3-Swallow-SFT/swallow-sft-reasoning/stem/gpt-oss-120b/medium/
  FORMATTED_INDEX=$(printf "%03d" $INDEX)
  if [ -f ${OUTPUT_DIR}/train_${FORMATTED_INDEX}.jsonl ]; then
    echo "Skipping INDEX ${FORMATTED_INDEX} as output file already exists."
    continue
  fi
  qsub -P gch51639 -q R9920251300 -v RTYPE=rt_HG,INDEX=$INDEX scripts/nemotron/science/science-en-reasoning-medium.sh
done
