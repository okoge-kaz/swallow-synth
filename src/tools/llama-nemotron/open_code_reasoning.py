import json
import argparse
import os
from transformers import AutoTokenizer


def main(input_file, tokenizer_path, output_dir, include_thinking=False):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for file splitting
    max_lines_per_file = 100000
    file_index = 0
    line_count = 0
    current_file = None

    # Read input JSONL file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            # Open new file if needed
            if line_count % max_lines_per_file == 0:
                if current_file:
                    current_file.close()
                output_file = os.path.join(output_dir, f"train_{file_index}.jsonl")
                current_file = open(output_file, "w", encoding="utf-8")
                file_index += 1

            data = json.loads(line.strip())

            # Check conditions: judgement is true and solution is 10+ characters
            if data.get("judgement", False) and len(data.get("solution", "")) >= 10:
                output_entry = {
                    "solution": data["solution"],
                    "question": data["question"],
                    "question_solution_pair": f"### Question:\n{data['question']}\n\n### Answer:\n{data['solution']}",
                    "question_solution_llama_chat_style": f"<|start_header_id|>user<|end_header_id>\n\n{data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n{data['solution']}<|eot_id>",
                }

                # Include thinking if enabled
                if include_thinking:
                    r1_gen = data.get("r1_generation", "")
                    if "<think>" in r1_gen and "</think>" in r1_gen:
                        tokens = tokenizer.encode(r1_gen, add_special_tokens=False)
                        token_length = len(tokens)
                        output_entry["question_thinking_solution_llama_chat_style"] = (
                            f"<|start_header_id|>user<|end_header_id>\n\n{data['question']}<op_think>{token_length}</op_think><|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n{r1_gen}{data['solution']}<|eot_id>"
                        )
                    else:
                        output_entry["question_thinking_solution_llama_chat_style"] = ""

                # Write to current file
                current_file.write(json.dumps(output_entry, ensure_ascii=False) + "\n")  # type: ignore
                line_count += 1

    # Close the last file
    if current_file:
        current_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL file to generate split output files.")
    parser.add_argument("--input-file", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--tokenizer-path",
        default="/groups/gag51395/hf_checkpoints/Meta-Llama-3.1-8B",
        help="Path or identifier for the tokenizer",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to save output JSONL files")
    parser.add_argument(
        "--include-thinking", action="store_true", help="Include question_thinking_solution_llama_chat_style"
    )
    args = parser.parse_args()

    main(args.input_file, args.tokenizer_path, args.output_dir, args.include_thinking)
