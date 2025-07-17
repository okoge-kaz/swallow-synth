import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze score distribution from multiple JSONL files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing train_*_Qwen3-14B.jsonl files")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def read_jsonl_files(input_dir, debug=False):
    pattern = os.path.join(input_dir, "train_*_Qwen3-14B.jsonl")
    files = glob.glob(pattern)

    if debug:
        print(f"Debug: Pattern used: {pattern}")
        print(f"Debug: Files found: {files}")

    if not files:
        print(f"Error: No files matching 'train_*_Qwen3-14B.jsonl' found in {input_dir}")
        return [], []

    scores = []
    file_counts = []

    for file_path in sorted(files):
        file_scores = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "score" in data:
                        file_scores.append(float(data["score"]))
                except (json.JSONDecodeError, ValueError):
                    continue

        scores.extend(file_scores)
        file_counts.append((file_path, len(file_scores)))

        print(f"Processed {file_path}: {len(file_scores)} scores")

    return scores, file_counts


def compute_statistics(scores):
    scores = np.array(scores)

    stats = {
        "count": len(scores),
        "mean": np.mean(scores),
        "median": np.median(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "q1": np.percentile(scores, 25),
        "q3": np.percentile(scores, 75),
        "tertile1": np.percentile(scores, 33.33),
        "tertile2": np.percentile(scores, 66.67),
    }
    return stats


def plot_distribution(scores, output_path, debug=False):
    if not scores:
        print("Error: No scores to plot")
        return

    scores_array = np.array(scores)

    below_zero_count = np.sum(scores_array < 0)
    above_ten_count = np.sum(scores_array > 10)
    valid_scores = scores_array[(scores_array >= 0) & (scores_array <= 10)]

    plt.figure(figsize=(12, 8))

    plt.hist(valid_scores, bins=30, range=(0, 10), edgecolor="black", alpha=0.7, color="skyblue")

    plt.title("Score Distribution (0 to 10)", fontsize=14)
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)

    mean_score = np.mean(valid_scores)
    median_score = np.median(valid_scores)
    plt.axvline(mean_score, color="red", linestyle="--", label=f"Mean: {mean_score:.4f}")
    plt.axvline(median_score, color="green", linestyle="--", label=f"Median: {median_score:.4f}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved ({len(valid_scores)} scores within 0-10).")
    print(f"Scores < 0: {below_zero_count}, Scores > 10: {above_ten_count}")


def main():
    args = parse_args()

    scores, file_counts = read_jsonl_files(args.input_dir, args.debug)
    if not scores:
        print("Error: No valid scores found.")
        return

    scores_array = np.array(scores)
    below_zero_count = np.sum(scores_array < 0)
    above_ten_count = np.sum(scores_array > 10)
    valid_scores = scores_array[(scores_array >= 0) & (scores_array <= 10)]

    print(f"Total scores: {len(scores)}")
    print(f"Scores below 0: {below_zero_count}")
    print(f"Scores above 10: {above_ten_count}")
    print(f"Valid scores (0-10): {len(valid_scores)}")

    stats = compute_statistics(valid_scores)
    if stats:
        print("\nScore Statistics (0-10):")
        for k, v in stats.items():
            print(f"{k.capitalize()}: {v:.4f}")

    output_plot = os.path.join(args.input_dir, "score_distribution_0_10.png")
    plot_distribution(scores, output_plot, args.debug)


if __name__ == "__main__":
    main()
