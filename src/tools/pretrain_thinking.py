import json
import re
import argparse


def process_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                improved_text = data.get("improved_text", "")

                think_pattern = r"<think>(.*?)</think>"
                matches = re.findall(think_pattern, improved_text, re.DOTALL)
                if not matches:
                    continue

                text = " ".join(matches)
                if "text_formatted" in data:
                    code = str(data["text_formatted"]).strip()

                thinking_text = f"<think>{text}</think>\n\n```python\n{code}\n```\n"
                output_data = {"thinking_text": thinking_text}

                json.dump(output_data, outfile, ensure_ascii=False)
                outfile.write("\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
            except Exception as e:
                print(f"Error processing line: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file to extract thinking text.")
    parser.add_argument("--input-file", type=str, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, help="Output JSONL file path")

    args = parser.parse_args()

    process_jsonl(args.input_file, args.output_file)
