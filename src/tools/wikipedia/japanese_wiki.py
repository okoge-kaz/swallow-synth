import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List
import time
import multiprocessing
from tqdm import tqdm

import datasets as ds
from bs4 import BeautifulSoup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Japanese Wikipedia dataset and save as JSONL.")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing jawiki_namespace_0_*.ndjson files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="jawiki_processed.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--include-disambiguation-page",
        action="store_true",
        default=False,
        help="Include disambiguation pages in the dataset (default: False)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of processes for multiprocessing (default: number of CPU cores)",
    )
    return parser.parse_args()


def load_paths(input_dir: str) -> List[Path]:
    paths = []
    for path in Path(input_dir).glob("jawiki_namespace_0_*.ndjson"):
        [idx] = re.findall(r"jawiki_namespace_0_(\d+).ndjson", path.name)
        paths.append((int(idx), path))
    return [path for _, path in sorted(paths, key=lambda x: x[0])]


def load_jsonl(example: Dict) -> Dict:
    abstract = example.get("abstract")
    templates: List[str] = [t["name"] for t in example.get("templates", [])]

    is_disambiguation_page = any("Template:Dmbox" in template or "Template:Aimai" in template for template in templates)
    is_sexual_page = any("Template:性的" in template for template in templates)
    is_violent_page = any("Template:暴力的" in template for template in templates)

    return {
        "identifier": example["identifier"],
        "title": example["name"],
        "abstract": abstract,
        "html": example["article_body"].get("html"),
        "wikitext": example["article_body"].get("wikitext"),
        "url": example["url"],
        "date_created": example.get("date_created"),
        "date_modified": example.get("date_modified"),
        "is_disambiguation_page": is_disambiguation_page,
        "is_sexual_page": is_sexual_page,
        "is_violent_page": is_violent_page,
        "templates": templates,
    }


def gen(paths: List[Path]):
    for path in tqdm(paths, desc="Processing input files"):
        with open(path) as f:
            for line in f:
                yield load_jsonl(json.loads(line))


SECTIONS_TO_IGNORE = ["脚注", "出典", "参考文献", "関連項目", "外部リンク"]
TAGS_TO_REMOVE = ["table"]
INNER_TAGS_TO_REMOVE = ["sup"]
TAGS_TO_EXTRACT = ["p"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    text = "".join(char for char in text if char.isprintable())
    return text.strip()


def mapping_jsonl(example: Dict) -> Dict:
    soup = BeautifulSoup(example["html"], features="lxml")
    section_title = ""
    section = soup.find(["section"])

    paragraph_id = 0
    paragraphs = []

    while section:
        if section.h2 is not None:  # type: ignore
            section_title = section.h2.text  # type: ignore

        for tag in section.find_all(TAGS_TO_REMOVE):  # type: ignore
            tag.clear()  # type: ignore

        for tag in section.find_all(TAGS_TO_EXTRACT):  # type: ignore
            for inner_tag in tag.find_all(INNER_TAGS_TO_REMOVE):  # type: ignore
                inner_tag.clear()  # type: ignore

            paragraph_text = normalize_text(tag.text)

            if section_title in SECTIONS_TO_IGNORE:
                continue

            current_title = "概要" if paragraph_id == 0 and not section_title else section_title

            paragraphs.append(
                {
                    "title": current_title,
                    "text": paragraph_text,
                    "tag": tag.name,  # type: ignore
                    "paragraph_id": paragraph_id,
                }
            )
            paragraph_id += 1

        section = section.find_next_sibling(["section"])

    return {"paragraphs": paragraphs}


def process(example: Dict) -> Dict:
    # Group paragraphs by title
    section_dict = {}
    for paragraph in example["paragraphs"]:
        title = paragraph["title"]
        if title not in section_dict:
            section_dict[title] = []
        section_dict[title].append(paragraph["text"])

    # Create sections list with section_id, title, and combined text
    sections = []
    for section_id, (title, texts) in enumerate(section_dict.items()):
        combined_text = "\n\n".join(text for text in texts if text)
        sections.append({"section_id": section_id, "title": title, "text": combined_text})

    # Combine all section texts for the 'text' field
    all_text = "\n\n".join(
        f"## {section['title']}\n\n{section['text']}" if section["title"] else section["text"]
        for section in sections
        if section["text"]
    )

    # Append a single newline if all_text is not empty
    if all_text:
        all_text += "\n"

    return {"text": all_text, "sections": sections}


def save_to_jsonl(dataset: ds.Dataset, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Saving to JSONL"):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    # Start timing
    start_time = time.time()
    print("Starting dataset processing...")

    # Load paths
    path_start = time.time()
    paths = load_paths(args.input_dir)
    print(f"Loading paths took {time.time() - path_start:.2f} seconds")

    # Create dataset
    dataset_start = time.time()
    dataset = ds.Dataset.from_generator(
        lambda: gen(paths),
        cache_dir=args.input_dir,
    )
    print(f"Creating dataset took {time.time() - dataset_start:.2f} seconds")

    # Filter out examples without HTML
    filter_html_start = time.time()
    dataset = dataset.filter(  # type: ignore
        lambda x: x["html"] is not None,
        num_proc=args.num_proc,  # type: ignore
        desc="Filtering HTML",  # type: ignore
    )  # type: ignore
    print(f"Filtering HTML took {time.time() - filter_html_start:.2f} seconds")

    # Filter out disambiguation pages unless --include-disambiguation-page is specified
    filter_disambig_start = time.time()
    if not args.include_disambiguation_page:
        dataset = dataset.filter(
            lambda x: not x["is_disambiguation_page"], num_proc=args.num_proc, desc="Filtering disambiguation pages"
        )  # type: ignore
    print(f"Filtering disambiguation pages took {time.time() - filter_disambig_start:.2f} seconds")

    # Map HTML to paragraphs
    map_start = time.time()
    dataset = dataset.map(
        mapping_jsonl, remove_columns=["html"], num_proc=args.num_proc, desc="Mapping HTML to paragraphs"
    )
    print(f"Mapping HTML to paragraphs took {time.time() - map_start:.2f} seconds")

    # Process paragraphs to text and sections
    process_start = time.time()
    dataset = dataset.map(process, num_proc=args.num_proc, desc="Processing paragraphs")
    print(f"Processing paragraphs took {time.time() - process_start:.2f} seconds")

    # Sort and select columns
    sort_start = time.time()
    dataset = dataset.sort("identifier")  # type: ignore
    dataset = dataset.select_columns(
        [
            "identifier",
            "title",
            "text",
            "paragraphs",
            "sections",
            "abstract",
            "wikitext",
            "date_created",
            "date_modified",
            "is_disambiguation_page",
            "is_sexual_page",
            "is_violent_page",
            "templates",
            "url",
        ]
    )
    dataset = dataset.rename_column("identifier", "id")
    print(f"Sorting and selecting columns took {time.time() - sort_start:.2f} seconds")

    # Save to JSONL
    save_start = time.time()
    save_to_jsonl(dataset, args.output_file)  # type: ignore
    print(f"Saving to JSONL took {time.time() - save_start:.2f} seconds")

    # Total time
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Dataset saved to {args.output_file}")


if __name__ == "__main__":
    main()
