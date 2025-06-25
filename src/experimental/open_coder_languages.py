#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def split_jsonl_by_lang(input_dir, output_dir):
    """
    指定されたディレクトリ内のすべてのJSONLファイルを読み込み、
    'lang'キーの値に基づいて言語別のJSONLファイルに分割する
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 出力ディレクトリが存在しない場合は作成
    output_path.mkdir(parents=True, exist_ok=True)

    # 言語ごとのデータを格納する辞書
    lang_data = defaultdict(list)

    # 入力ディレクトリ内のすべてのJSONLファイルを処理
    jsonl_files = list(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"警告: {input_dir} にJSONLファイルが見つかりません")
        return

    print(f"処理対象ファイル数: {len(jsonl_files)}")

    for jsonl_file in jsonl_files:
        print(f"処理中: {jsonl_file.name}")

        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # 'lang'キーが存在するかチェック
                        if "lang" not in data:
                            print(f"警告: {jsonl_file.name}:{line_num} に'lang'キーがありません")
                            continue

                        lang = data["lang"]
                        if not lang:
                            print(f"警告: {jsonl_file.name}:{line_num} の'lang'値が空です")
                            continue

                        lang_data[lang].append(data)

                    except json.JSONDecodeError as e:
                        print(f"エラー: {jsonl_file.name}:{line_num} のJSON解析に失敗: {e}")
                        continue

        except Exception as e:
            print(f"エラー: {jsonl_file.name} の読み込みに失敗: {e}")
            continue

    # 言語別にJSONLファイルを出力
    if not lang_data:
        print("警告: 有効なデータが見つかりませんでした")
        return

    print(f"\n検出された言語: {list(lang_data.keys())}")

    for lang, data_list in lang_data.items():
        output_file = output_path / f"{lang}.jsonl"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

            print(f"作成: {output_file} ({len(data_list)} 件)")

        except Exception as e:
            print(f"エラー: {output_file} の書き込みに失敗: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="JSONLファイルを言語別に分割します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input-dir", "-i", required=True, help="入力JSONLファイルが格納されているディレクトリ")

    parser.add_argument("--output-dir", "-o", required=True, help="分割されたJSONLファイルを保存するディレクトリ")

    args = parser.parse_args()

    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        print(f"エラー: 入力ディレクトリ '{args.input_dir}' が存在しません")
        return 1

    if not os.path.isdir(args.input_dir):
        print(f"エラー: '{args.input_dir}' はディレクトリではありません")
        return 1

    print(f"入力ディレクトリ: {args.input_dir}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print("-" * 50)

    split_jsonl_by_lang(args.input_dir, args.output_dir)

    print("-" * 50)
    print("処理完了")
    return 0


if __name__ == "__main__":
    exit(main())
