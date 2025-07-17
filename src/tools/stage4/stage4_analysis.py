import json
import argparse
from collections import defaultdict
import os


def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="Extract text_formatted and Qwen3-14B_evaluation by score from JSONL file"
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    args = parser.parse_args()

    # スコアごとのデータを保持する辞書
    score_data = defaultdict(list)

    # JSONLファイルを読み込む
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                score = data.get("score")
                if score is not None and 0 <= score <= 10:
                    score_data[score].append(
                        {
                            "text_formatted": data.get("text_formatted"),
                            "Qwen3-14B_evaluation": data.get("Qwen3-14B_evaluation"),
                        }
                    )
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")

    # 出力ディレクトリを作成
    output_dir = "score_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 各スコアについて最大10個のデータをファイルに保存
    for score in range(11):
        output_file = os.path.join(output_dir, f"score_{score}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in score_data[score][:10]:  # 最大10個
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {min(len(score_data[score]), 10)} items for score {score} to {output_file}")


if __name__ == "__main__":
    main()
