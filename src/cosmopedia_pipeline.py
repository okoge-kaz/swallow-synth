import json
import time
from pathlib import Path
from typing import Any, Iterator

SYSTEM_PROMPT = """You are an expert educational content creator tasked with writing high-quality, engaging, and rigorous educational content for a textbook or similar resource. The content is intended for a specified audience (e.g., high school students, college students, professionals) and a specific subject area (e.g., mathematics, history, engineering, or others). Your task is to create a detailed sub-unit, lesson, or section that aligns with the provided topic, audience, and subject, building on any prior context or chapters mentioned. The content must be tailored to the audience's age, knowledge level, and interests, using appropriate language and examples while maintaining a narrative-driven, engaging tone and a practical, applied approach.
"""


class DataGenerationPipeline:
    """Generic data generation pipeline for any type of content"""

    def __init__(self, model_name: str, tensor_parallel_size: int = 1, model_max_length: int = 131072):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.model_max_length = model_max_length
        self.enable_thinking = True

    def set_thinking_mode(self, enable: bool):
        """Set thinking mode on/off"""
        self.enable_thinking = enable

    def generate_from_prompts(
        self, prompts: list[str], max_new_tokens: int = 2048, temperature: float = 0.7, top_p: float = 1.0
    ) -> list[str]:
        """
        Generate responses from prompts using the language model
        This method should be implemented based on your specific model interface
        """
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, max_model_len=self.model_max_length
        )

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)

        # Apply chat template with thinking mode
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            formatted_prompt = llm.get_tokenizer().apply_chat_template(
                messages,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            formatted_prompts.append(formatted_prompt)

        outputs = llm.generate(formatted_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


def get_generation_pipeline(
    model_name: str, tensor_parallel_size: int = 1, model_max_length: int = 131072, enable_thinking: bool = True
) -> DataGenerationPipeline:
    """Get data generation pipeline with thinking mode configuration"""

    pipeline = DataGenerationPipeline(model_name, tensor_parallel_size, model_max_length)
    pipeline.set_thinking_mode(enable_thinking)

    return pipeline


def stream_jsonl(file_path: Path, batch_size: int = 1024) -> Iterator[list[dict[str, Any]]]:
    """Stream JSONL file in batches"""
    batch = []
    with file_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # return remaining data
            yield batch


def llm_data_generation(
    input_path: Path,
    output_path: Path,
    model_name: str = "qwen-3",
    batch_size: int = 1024,
    tensor_parallel_size: int = 1,
    model_max_length: int = 131072,
    enable_thinking: bool = True,
    prompt_key: str = "prompt",
    output_key: str = "generated_text",
    max_new_tokens: int = 16384,
) -> None:
    """Generate data using LLM based on prompts from input JSONL"""

    pipeline = get_generation_pipeline(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        model_max_length=model_max_length,
        enable_thinking=enable_thinking,
    )

    total_items = 0
    start_time = time.time()

    print(f"Starting LLM data generation with {tensor_parallel_size} GPUs...")
    print(f"Model: {model_name}")
    print(f"Thinking mode: {'enabled' if enable_thinking else 'disabled'}")
    print(f"Input prompt key: '{prompt_key}'")
    print(f"Output key: '{output_key}'")
    print(f"Max new tokens: {max_new_tokens}")

    with output_path.open("w", encoding="utf-8") as fout:
        for batch in stream_jsonl(input_path, batch_size):
            total_items += len(batch)
            print(f"Processing batch of {len(batch)} items...")

            # Extract prompts from batch
            prompts = []
            valid_items = []

            for item in batch:
                if prompt_key in item:
                    prompts.append(item[prompt_key])
                    valid_items.append(item)
                else:
                    print(f"Warning: Item missing '{prompt_key}' key, skipping...")
                    continue

            if not prompts:
                print("No valid prompts found in batch, skipping...")
                continue

            # Generate responses using the pipeline
            try:
                generated_texts = pipeline.generate_from_prompts(
                    prompts,
                    max_new_tokens=max_new_tokens,
                )

                # Write results to output file
                for item, generated_text in zip(valid_items, generated_texts):
                    item[output_key] = generated_text
                    item["generation_metadata"] = {
                        "model_name": model_name,
                        "enable_thinking": enable_thinking,
                        "max_new_tokens": max_new_tokens,
                        "timestamp": time.time(),
                    }
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error during generation: {e}")
                # Write original items without generated text in case of error
                for item in valid_items:
                    item[output_key] = ""
                    item["generation_error"] = str(e)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    actual_time = time.time() - start_time
    print(f"LLM data generation completed: {actual_time:.1f}s total ({actual_time / total_items:.3f}s per item)")


# === CLI Entrypoint ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Data Generation Pipeline")

    # Data generation subcommand
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input JSONL file with prompts")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output JSONL file with generated data")
    parser.add_argument("--model", type=str, default="qwen-3", help="Model identifier for inference")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument("--model-max-length", type=int, default=40960, help="Maximum model length")
    parser.add_argument(
        "--enable-thinking", action="store_true", default=False, help="Enable thinking mode (<think> tags)"
    )
    parser.add_argument(
        "--disable-thinking", action="store_true", default=False, help="Explicitly disable thinking mode"
    )
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Key in JSON object containing the prompt")
    parser.add_argument("--output-key", type=str, default="generated_text", help="Key to store generated text")
    parser.add_argument("--max-new-tokens", type=int, default=20480, help="Maximum number of new tokens to generate")

    args = parser.parse_args()

    # Determine thinking mode
    enable_thinking = True  # Default
    if args.disable_thinking:
        enable_thinking = False
    elif args.enable_thinking:
        enable_thinking = True

    llm_data_generation(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        model_name=args.model,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        model_max_length=args.model_max_length,
        enable_thinking=enable_thinking,
        prompt_key=args.prompt_key,
        output_key=args.output_key,
        max_new_tokens=args.max_new_tokens,
    )
