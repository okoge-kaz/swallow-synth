#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def process_jsonl_file(input_file, output_file):
    """
    JSONLファイルを処理して、judgement=="right"のデータのみを抽出し、
    questionとr1_generationを結合した新しいテキストフィールドを作成
    """

    # 入力ファイルの存在確認
    if not Path(input_file).exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

    processed_count = 0
    filtered_count = 0
    error_count = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()

            # 空行をスキップ
            if not line:
                continue

            try:
                # JSONデータを解析
                data = json.loads(line)

                # judgementフィールドのチェック
                if "judgement" not in data:
                    print(f"警告: {line_num}行目 - 'judgement'フィールドが見つかりません")
                    error_count += 1
                    continue

                # judgement == "right" のもののみを処理
                if data["judgement"] != "right":
                    filtered_count += 1
                    continue

                # 必要なフィールドが存在するかチェック
                required_fields = ["question", "r1_generation"]
                missing_fields = [field for field in required_fields if field not in data]

                if missing_fields:
                    print(f"警告: {line_num}行目 - 必要なフィールドが不足: {missing_fields}")
                    error_count += 1
                    continue

                # 新しいテキストフィールドを作成
                text_content = (
                    "Question:\n\n" + str(data["question"]) + "\n\nSolution:\n\n" + str(data["r1_generation"])
                )

                # 元のデータを保持しつつ、新しいtextフィールドを追加
                output_data = data.copy()
                output_data["text"] = text_content

                # 出力ファイルに書き込み
                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"エラー: {line_num}行目 - JSON解析エラー: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"エラー: {line_num}行目 - 処理中にエラーが発生: {e}")
                error_count += 1
                continue

    # 処理結果の表示
    print(f"処理完了:")
    print(f"  - 処理済み (judgement=='right'): {processed_count}行")
    print(f"  - フィルタされた (judgement!='right'): {filtered_count}行")
    print(f"  - エラー: {error_count}行")
    print(f"  - 出力先: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="JSONLファイルからjudgement=='right'のデータを抽出し、questionとr1_generationを結合して出力します"
    )

    parser.add_argument("--input-jsonl", required=True, help="入力JSONLファイルのパス")

    parser.add_argument("--output-jsonl", required=True, help="出力JSONLファイルのパス")

    args = parser.parse_args()

    try:
        process_jsonl_file(args.input_jsonl, args.output_jsonl)
    except Exception as e:
        print(f"エラー: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
