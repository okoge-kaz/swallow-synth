#!/bin/bash
#PBS -q rt_HC
#PBS -N stage3
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/pipeline/stage3_python

# set -e
# cd $PBS_O_WORKDIR


# environment variables
export TMP="/groups/gag51395/fujii/tmp"
export TMP_DIR="/groups/gag51395/fujii/tmp"
export HF_HOME="/groups/gag51395/fujii/hf_cache"

source .venv/bin/activate

INPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage2/python"
OUTPUT_DIR="/groups/gag51395/datasets/raw/pretrain/swallow-code-v2/stage3/python"
mkdir -p $OUTPUT_DIR

export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="/groups/gag51395/fujii/src/swallow-code-v2:$PYTHONPATH"

# 処理範囲の設定
MIN_INDEX=1
MAX_INDEX=162

echo "Processing INDEX range: $(printf "%04d" $MIN_INDEX) to $(printf "%04d" $MAX_INDEX)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

processed_count=0
failed_count=0
not_found_count=0

# INDEXを直接指定してループ
for ((i=MIN_INDEX; i<=MAX_INDEX; i++)); do
  # INDEXを4桁でフォーマット
  INDEX=$(printf "%04d" $i)

  # 入力ファイルパスと出力ファイルパスを構築
  INPUT_FILE_PATH="$INPUT_DIR/train_${INDEX}_without_errors.jsonl"
  OUTPUT_FILE_PATH="$OUTPUT_DIR/train_${INDEX}.jsonl"

  echo "=== Processing INDEX: $INDEX ($((i+1))/$(($MAX_INDEX-$MIN_INDEX+1))) ==="
  echo "Input:  $INPUT_FILE_PATH"
  echo "Output: $OUTPUT_FILE_PATH"

  # 入力ファイルの存在確認
  if [ ! -f "$INPUT_FILE_PATH" ]; then
    echo "⚠️  Input file not found: train_${INDEX}_without_errors.jsonl"
    ((not_found_count++))
    echo "----------------------------------------"
    continue
  fi

  # ファイルサイズ確認
  file_size=$(stat -c%s "$INPUT_FILE_PATH" 2>/dev/null || echo "0")
  if [ "$file_size" -eq 0 ]; then
    echo "⚠️  Input file is empty: train_${INDEX}_without_errors.jsonl"
    ((failed_count++))
    echo "----------------------------------------"
    continue
  fi

  echo "Input file size: $file_size bytes"

  # 出力ファイルが既に存在する場合の確認
  if [ -f "$OUTPUT_FILE_PATH" ]; then
    output_size=$(stat -c%s "$OUTPUT_FILE_PATH" 2>/dev/null || echo "0")
    if [ "$output_size" -gt 0 ]; then
      echo "✅ Output file already exists and is not empty: train_${INDEX}.jsonl ($output_size bytes)"
      echo "   Skipping processing..."
      ((processed_count++))
      echo "----------------------------------------"
      continue
    else
      echo "⚠️  Output file exists but is empty, will reprocess: train_${INDEX}.jsonl"
    fi
  fi

  echo "Starting processing at: $(date)"

  # 処理実行
  if uv run python src/pipeline.py long_context_sample \
    --input-jsonl "$INPUT_FILE_PATH" \
    --output-path "$OUTPUT_FILE_PATH" \
    --tokenizer "/groups/gag51395/hf_checkpoints/Qwen3-235B-A22B" \
    --threshold-length 20480; then

    echo "✅ Successfully processed: train_${INDEX}_without_errors.jsonl at $(date)"

    # 出力ファイルの確認
    if [ -f "$OUTPUT_FILE_PATH" ]; then
      output_size=$(stat -c%s "$OUTPUT_FILE_PATH" 2>/dev/null || echo "0")
      echo "   Output file size: $output_size bytes"
      if [ "$output_size" -gt 0 ]; then
        ((processed_count++))
      else
        echo "❌ Output file is empty!"
        ((failed_count++))
      fi
    else
      echo "❌ Output file was not created!"
      ((failed_count++))
    fi
  else
    echo "❌ Failed to process: train_${INDEX}_without_errors.jsonl at $(date)"
    ((failed_count++))
  fi

  echo "Progress: $((processed_count + failed_count + not_found_count))/$((MAX_INDEX - MIN_INDEX + 1))"
  echo "----------------------------------------"
done

echo ""
echo "=== Final Processing Summary ==="
echo "INDEX range: $(printf "%04d" $MIN_INDEX) to $(printf "%04d" $MAX_INDEX)"
echo "Total expected files: $((MAX_INDEX - MIN_INDEX + 1))"
echo "Successfully processed: $processed_count files"
echo "Failed to process: $failed_count files"
echo "Input files not found: $not_found_count files"

if [ $failed_count -eq 0 ] && [ $not_found_count -eq 0 ]; then
  echo "🎉 All files processed successfully!"
  exit 0
elif [ $failed_count -eq 0 ]; then
  echo "✅ All available files processed successfully!"
  echo "⚠️  $not_found_count input files were not found."
  exit 0
else
  echo "⚠️  $failed_count files failed to process."
  exit 1
fi
