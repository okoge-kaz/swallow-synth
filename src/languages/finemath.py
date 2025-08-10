import json
import time
from pathlib import Path
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.prompts.finemath.pretrain_math_text import PRE_TRAIN_MATH_TEXT
from src.prompts.finemath.textbook_math import TEXT_BOOK_MATH_TEXT
from src.prompts.finemath.question_answer import QUESTION_ANSWER_PROMPT
from src.languages.abc import RewritePipeline


class FinemathRewritePipeline(RewritePipeline):
    def __init__(self, model_name: str = "qwen-3", tensor_parallel_size: int = 1, max_model_len: int = 40960) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=max_model_len,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompts: list[str]) -> list[str]:
        # Greedy / deterministic inference
        params = SamplingParams(temperature=0)
        outputs = self.llm.generate(prompts, params)
        return [output.text for output in outputs]  # type: ignore

    def rewrite_codes(self, texts: list[str], prompt_type: str = "pre-train-text") -> list[str]:
        # Construct chat templates for batch processing
        prompts: list[str] = []

        # Select prompt based on prompt_type
        if prompt_type == "pre-train-text":
            PROMPT = PRE_TRAIN_MATH_TEXT
        elif prompt_type == "text-book-style":
            PROMPT = TEXT_BOOK_MATH_TEXT
        elif prompt_type == "question-answer":
            PROMPT = QUESTION_ANSWER_PROMPT
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}. Supported types: 'pre-train-text'")

        for text in texts:
            prompt = (
                "<|im_start|>system\n"
                + PROMPT
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "### Input\n"
                + "```\n"
                + f"{text}\n"
                + "```\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
            prompts.append(prompt)

        tokenized_prompts_len = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        max_len = max(tokenized_prompts_len)
        if max_len >= self.max_model_len:
            raise ValueError(
                f"Prompt length exceeds model limit: {max_len} >= {self.max_model_len}. "
                "Consider reducing the input size or using a smaller model."
            )
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=self.max_model_len - max_len))
        return [output.outputs[0].text for output in outputs]  # type: ignore

    def competitive_programming_write(self, questions: List[str]) -> List[str]:
        # Not applicable for math content
        raise NotImplementedError("Competitive programming is not applicable for math content")


def extract_math_text(text: str) -> str:
    """Extract math text from the response"""
    start_marker = "<|MATH_TEXT|>"
    start_index = text.find(start_marker)
    if start_index == -1:
        return text.strip()  # Return original text if marker not found

    return text[start_index + len(start_marker) :].strip()


def stream_jsonl_math(file_path: Path, batch_size: int = 1024):
    """Stream JSONL file in batches for math processing"""
    batch = []
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # return remaining data
            yield batch


def math_rewrite(
    input_path: Path,
    output_path: Path,
    model_name: str = "qwen-3",
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    model_max_length: int = 40960,
    prompt_type: str = "pre-train-text",
) -> None:
    """Math text rewriting using GPU processing"""
    pipeline = FinemathRewritePipeline(
        model_name=model_name, tensor_parallel_size=tensor_parallel_size, max_model_len=model_max_length
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting math rewriting with {tensor_parallel_size} GPUs using {prompt_type} prompt...")

    with output_path.open("w", encoding="utf-8") as fout:
        for batch in stream_jsonl_math(input_path, batch_size):
            total_items += len(batch)
            print(f"Processing batch of {len(batch)} items...")

            # Use "text" field for math rewriting
            if not all("text" in item for item in batch):
                raise ValueError("All items in the batch must contain 'text' key for math rewriting")
            texts = [item.get("text", "") for item in batch]

            # Call pipeline.rewrite_codes with specified prompt type
            try:
                rewritten_texts = pipeline.rewrite_codes(texts, prompt_type=prompt_type)

                # Write results to output file
                for index, item in enumerate(batch):
                    rewritten_text = rewritten_texts[index] if index < len(rewritten_texts) else ""
                    extracted_math = extract_math_text(rewritten_text)
                    item["llm_output"] = rewritten_text
                    item["llm_extracted_math_text"] = extracted_math
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error during math rewriting: {e}")

    actual_time = time.time() - start_time
    print(f"Math rewriting completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")
