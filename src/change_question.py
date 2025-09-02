import argparse
import json
import os
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

# Define the competitive programming prompt as a message structure for chat templates
COMPETITIVE_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant tasked with rewriting problem statements into competitive programming style, similar to CodeForces, TopCoder, or AtCoder, at Medium to Hard difficulty level.",
    },
    {
        "role": "user",
        "content": """Given the following problem statement, rewrite it into a competitive programming problem format at Medium to Hard level. Make it self-contained, with sections like: Problem Description, Input Format, Output Format, Constraints, Sample Input/Output, and possibly Notes or Time/Space Limits.

For topics like DB operations, reframe them as simulations using data structures (e.g., arrays, maps) without actual databases. For API creation, convert to problems involving query optimizations, access counts, or data processing with limits.

The problem should be solvable in Python.

Original Question:
{question}

Provide the rewritten question right after "**Rewritten Question:**". Ensure the marker is exactly as shown, including the asterisks.""",
    },
]

# Define the standard library prompt as a message structure for chat templates
STANDARD_LIB_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant tasked with rewriting problem statements to be solvable using Python's standard library in a single function, in a style similar to LeetCode or coding interview problems used by companies.",
    },
    {
        "role": "user",
        "content": """Given the following problem statement, rewrite it into an educational coding problem that can assess a candidate's level, similar to those in technical interviews. It should be based on real-world business abstractions, like data processing, file handling, or algorithm applications in practical scenarios, rather than pure competitive programming puzzles.

The problem must be solvable using only Python's standard library (no external packages) and implemented as a single function. Include details on the function signature, input/output, constraints, examples, etc.

Avoid making it overly complex like hard competitive problems; focus on practical, realistic cases.

The original code does not need to be an exact solution; adapt the problem as needed.

Original Question:
{question}

Provide the rewritten question right after "**Rewritten Question:**". Ensure the marker is exactly as shown, including the asterisks.""",
    },
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with clear defaults and type hints."""
    parser = argparse.ArgumentParser(
        description="Rewrite problem statements from JSONL using vLLM into competitive programming or standard library styles."
    )
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to input JSONL file containing questions.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the vLLM model.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for processing and saving.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["competitive", "standard"],
        default="competitive",
        help="Mode: 'competitive' for CodeForces-style (Medium/Hard), 'standard' for LeetCode/interview-style with std lib.",
    )
    parser.add_argument("--gen-max-tokens", type=int, default=16384, help="Max tokens for generated output.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM.")
    return parser.parse_args()


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Load and return a tokenizer for the specified model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def get_prompt_template(mode: str) -> List[Dict[str, str]]:
    """Return the appropriate chat message template based on mode."""
    prompt_map = {
        "competitive": COMPETITIVE_CHAT_MESSAGES,
        "standard": STANDARD_LIB_CHAT_MESSAGES,
    }
    return prompt_map.get(mode, COMPETITIVE_CHAT_MESSAGES)


def read_jsonl(file_path: str) -> List[Dict[str, Union[str, List]]]:
    """Read and parse JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_to_jsonl(file_path: str, data_list: List[Dict[str, Union[str, List]]]) -> None:
    """Append data to a JSONL file."""
    mode = "a" if os.path.exists(file_path) else "w"
    with open(file_path, mode, encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_generated_text(text: str) -> Dict[str, str]:
    """Parse generated text into a structured question dictionary."""
    print(f"Generated Text:\n{text}\n{'-' * 40}", flush=True)
    qa: Dict[str, str] = {}

    try:
        # Find the index of "**Rewritten Question:**" in the text
        question_marker = "**Rewritten Question:**"
        start_idx = text.find(question_marker)
        if start_idx == -1:
            raise ValueError("No '**Rewritten Question:**' marker found in generated text")

        # Extract everything after "**Rewritten Question:**" as the question
        question_text = text[start_idx + len(question_marker) :].strip()
        if not question_text:
            raise ValueError("Question text is empty")

        qa["question"] = question_text
    except Exception as e:
        qa = {"error": f"Failed to parse question: {str(e)}"}

    return qa


def process_batch(
    batch_lines: List[Dict[str, Union[str, List]]],
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    prompt_template: List[Dict[str, str]],
    gen_max_tokens: int,
) -> List[Dict[str, Union[str, List]]]:
    """Process a batch of JSONL lines and rewrite questions using chat template."""
    batch_inputs = []
    valid_indices = []

    for local_idx, item in enumerate(batch_lines):
        question = item.get("question", "")
        if question:
            # Create a copy of the prompt template and format the user message with the question
            formatted_messages = [msg.copy() for msg in prompt_template]
            for msg in formatted_messages:
                if msg["role"] == "user":
                    msg["content"] = msg["content"].format(question=question.strip())  # type: ignore
            # Apply the chat template to format the prompt
            prompt = tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(prompt)
            valid_indices.append(local_idx)

    if not batch_inputs:
        return []

    # Generate rewritten questions using vLLM
    outputs = llm.generate(
        batch_inputs,
        sampling_params=SamplingParams(
            max_tokens=gen_max_tokens,
            temperature=0.0,
            top_p=1.0,
        ),
    )

    questions = [parse_generated_text(output.outputs[0].text) for output in outputs]
    processed_lines = []

    # Create new output entries with the rewritten question and original answer
    for qa_idx, line_idx in enumerate(valid_indices):
        q = questions[qa_idx]

        if "error" in q:
            print(f"Warning: {q['error']} in line {line_idx}, skipping item")
            continue

        # Create a new dictionary with the rewritten question and original answer
        processed_lines.append(
            {
                "question": q["question"],
                "answer": batch_lines[line_idx].get("answer", ""),
            }
        )

    return processed_lines


def main():
    """Main function to process JSONL and rewrite questions using vLLM with chat template."""
    args = parse_args()

    # Initialize tokenizer and vLLM
    tokenizer = load_tokenizer(args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    # Get prompt template based on mode
    prompt_template = get_prompt_template(args.mode)

    # Read input JSONL
    lines = read_jsonl(args.input_jsonl)

    # Process in batches
    for start_idx in range(0, len(lines), args.batch_size):
        batch_lines = lines[start_idx : start_idx + args.batch_size]
        processed_lines = process_batch(batch_lines, tokenizer, llm, prompt_template, args.gen_max_tokens)
        if processed_lines:
            save_to_jsonl(args.output_jsonl, processed_lines)


if __name__ == "__main__":
    main()
