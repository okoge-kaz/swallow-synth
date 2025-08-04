import json
import argparse
import re


def extract_python_code(text):
    """Extracts Python code blocks from text marked with ```python."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else ""


def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get("improved_code") == "":
                python_code = extract_python_code(data.get("improved_text", ""))
                if python_code:
                    data["improved_code"] = python_code
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL file to extract Python code from improved_text and insert into improved_code."
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL file")
    args = parser.parse_args()

    process_jsonl(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
