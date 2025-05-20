import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Callable

from src.languages.python import process_item_cpu as python_process_item_cpu, RewritePipeline as PythonRewritePipeline
from src.languages.rust import process_item_cpu as rust_process_item_cpu, RewritePipeline as RustRewritePipeline
from src.languages.java import process_item_cpu as java_process_item_cpu, RewritePipeline as JavaRewritePipeline


def get_process_item_cpu(lang: str) -> Callable:
    processors = {
        "python": python_process_item_cpu,
        "rust": rust_process_item_cpu,
        "java": java_process_item_cpu,
        # other languages
    }
    if lang not in processors:
        raise ValueError(f"Unsupported language: {lang}")
    return processors[lang]


def get_rewrite_pipeline(lang: str, model_name: str):
    pipelines = {
        "python": PythonRewritePipeline,
        "rust": RustRewritePipeline,
        "java": JavaRewritePipeline,
        # other languages
    }
    if lang not in pipelines:
        raise ValueError(f"Unsupported language: {lang}")
    return pipelines[lang](model_name)


def preprocess(input_path: Path, intermediate_path: Path, lang: str, n_workers: int = 16) -> None:
    n_workers = n_workers or cpu_count()
    process_item_cpu = get_process_item_cpu(lang)

    # Load all items
    with input_path.open("r", encoding="utf-8") as fin:
        items = [json.loads(l) for l in fin]

    # Parallel CPU processing
    with Pool(n_workers) as pool, intermediate_path.open("w", encoding="utf-8") as fout:
        for result in pool.imap_unordered(process_item_cpu, items):
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


def rewrite(intermediate_path: Path, output_path: Path, lang: str, model_name: str = "qwen-3") -> None:
    pipeline = get_rewrite_pipeline(lang, model_name)

    with intermediate_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            result = pipeline.process_item_gpu(item)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


# === CLI Entrypoint ===

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("preprocess", help="CPU pre-processing stage")
    p1.add_argument("--input-jsonl", type=Path, required=True)
    p1.add_argument("--intermediate-jsonl", type=Path, required=True)
    p1.add_argument("--workers", type=int, default=32, help="Number of CPU workers")
    p1.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")

    p2 = sub.add_parser("rewrite", help="GPU rewriting stage")
    p2.add_argument("--intermediate-jsonl", type=Path, required=True)
    p2.add_argument("--output-jsonl", type=Path, required=True)
    p2.add_argument("--model", type=str, default="qwen-3", help="Local Qwen model identifier for vLLM")
    p2.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, rust, java)")

    args = parser.parse_args()
    if args.cmd == "preprocess":
        preprocess(
            input_path=args.input_jsonl,
            intermediate_path=args.intermediate_jsonl,
            n_workers=args.workers,
            lang=args.lang,
        )
    elif args.cmd == "rewrite":
        rewrite(
            intermediate_path=args.intermediate_jsonl,
            output_path=args.output_jsonl,
            model_name=args.model,
            lang=args.lang,
        )
