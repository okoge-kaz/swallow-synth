import argparse
import json
import re
from pathlib import Path


def clean_generated_solution(solution: str) -> str:
    """Remove content between <think> and </think> (inclusive)."""
    return re.sub(r"<think>.*?</think>", "", solution, flags=re.DOTALL)


def process_file(input_file: Path, output_file: Path) -> None:
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "a", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num} in {input_file}: {e}")
                continue

            # clean generated_solution
            solution = clean_generated_solution(item.get("generated_solution", ""))

            # build text field
            text = (
                f"Problem:\n\n{item.get('problem', '')}\n\n"
                f"Solution\n\n{solution}\n\n"
                f"Answer: {item.get('expected_answer', '')}"
            )

            # add new field
            item["text"] = text

            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Process cot-*.jsonl files")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing cot-*.jsonl files")
    parser.add_argument("--output-jsonl", type=Path, required=True,
                        help="Output JSONL file path")
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("cot-*.jsonl"))
    if not input_files:
        print(f"No files matching cot-*.jsonl found in {args.input_dir}")
        return

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        pass  # clear existing file

    for input_file in input_files:
        process_file(input_file, args.output_jsonl)


if __name__ == "__main__":
    main()
