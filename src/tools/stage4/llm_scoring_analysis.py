import argparse
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from pathlib import Path


def load_jsonl(file_path, max_lines=100000):
    """Load up to max_lines from JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            data.append(json.loads(line.strip()))
    return data


def extract_and_validate_scores(data, score_key="score", id_key="blob_id"):
    """Extract scores, validate range [0-10], and return a dictionary with invalid count."""
    score_dict = {}
    invalid_scores = []
    for item in data:
        if id_key in item and score_key in item:
            score = item[score_key]
            # Validate score is numeric and within [0-10]
            if isinstance(score, (int, float)) and not np.isnan(score):
                if 0 <= score <= 10:
                    score_dict[item[id_key]] = float(score)
                else:
                    invalid_scores.append((item[id_key], score))
    return score_dict, len(invalid_scores)


def compare_scores(reference_scores, model_scores):
    """Compare model scores to reference scores and calculate correlations."""
    common_ids = set(reference_scores.keys()) & set(model_scores.keys())
    if not common_ids:
        return None, None, 0

    ref_vals = np.array([reference_scores[id] for id in common_ids])
    model_vals = np.array([model_scores[id] for id in common_ids])

    # Check for NaN or infinite values
    if np.any(np.isnan(ref_vals)) or np.any(np.isnan(model_vals)):
        print("Warning: NaN values detected")
        return None, None, len(common_ids)
    if np.any(np.isinf(ref_vals)) or np.any(np.isinf(model_vals)):
        print("Warning: Infinite values detected")
        return None, None, len(common_ids)

    # Calculate correlations
    pearson_corr = None
    spearman_corr = None
    if np.var(ref_vals) > 0 and np.var(model_vals) > 0:
        pearson_corr, _ = pearsonr(ref_vals, model_vals)
        spearman_corr, _ = spearmanr(ref_vals, model_vals)

    return pearson_corr, spearman_corr, len(common_ids)


def plot_score_distribution_pair(reference_scores, model_scores, model_name, output_dir=None):
    """Plot score distribution for reference and one model in range [0-10]."""
    plt.figure(figsize=(8, 5))

    # Fixed bins for 0-10 range
    bins = np.linspace(0, 10, 21)  # 20 bins from 0 to 10

    # Plot reference distribution
    ref_vals = np.array(list(reference_scores.values()))
    plt.hist(ref_vals, bins=bins, alpha=0.5, label="Reference", color="blue", density=True, edgecolor="black")

    # Plot model distribution
    model_vals = np.array(list(model_scores.values()))
    plt.hist(
        model_vals, bins=bins, alpha=0.5, label=Path(model_name).stem, color="red", density=True, edgecolor="black"
    )

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(f"Score Distribution: Reference vs {Path(model_name).stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)

    # Save plot
    output_path = Path(output_dir) if output_dir else Path(".")
    output_path.mkdir(exist_ok=True)
    safe_model_name = "".join(c for c in Path(model_name).stem if c.isalnum() or c in ("-", "_")).rstrip()
    plt.savefig(output_path / f"score_dist_{safe_model_name}.png", dpi=300, bbox_inches="tight")
    print(f"Distribution plot saved to: {output_path / f'score_dist_{safe_model_name}.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM scores with range validation and paired distribution analysis."
    )
    parser.add_argument("--reference-jsonl", required=True, help="Path to reference JSONL file")
    parser.add_argument("--model-jsonl", nargs="+", required=True, help="Paths to model JSONL files")
    parser.add_argument("--output-dir", "-o", help="Directory to save plots")
    args = parser.parse_args()

    # Set matplotlib backend
    plt.switch_backend("Agg")

    # Load and validate reference scores (up to 100,000 lines)
    reference_data = load_jsonl(args.reference_jsonl, max_lines=100000)
    reference_scores, ref_invalid_count = extract_and_validate_scores(reference_data)
    print(
        f"Reference: Loaded {len(reference_scores)} valid scores (from up to 100,000 lines), {ref_invalid_count} scores outside [0-10]"
    )

    # Process each model
    all_model_scores = {}
    correlations = {}
    for model_path in args.model_jsonl:
        model_data = load_jsonl(model_path, max_lines=100000)
        model_scores, invalid_count = extract_and_validate_scores(model_data)
        all_model_scores[model_path] = model_scores
        print(
            f"Model {Path(model_path).stem}: Loaded {len(model_scores)} valid scores (from up to 100,000 lines), {invalid_count} scores outside [0-10]"
        )

        # Compare scores
        pearson_corr, spearman_corr, num_common = compare_scores(reference_scores, model_scores)
        correlations[model_path] = {"pearson": pearson_corr, "spearman": spearman_corr, "num_common": num_common}

        print(f"\nResults for model: {model_path}")
        if pearson_corr is None:
            print("No valid comparison possible (no common IDs or data issues).")
        else:
            print(f"Pearson Correlation: {pearson_corr:.4f}")
            print(f"Spearman Correlation: {spearman_corr:.4f}")
            print(f"Number of compared scores: {num_common}")

        # Plot paired distribution
        if model_scores:
            plot_score_distribution_pair(reference_scores, model_scores, model_path, args.output_dir)

    # Identify closest model
    valid_corrs = {model: res["pearson"] for model, res in correlations.items() if res["pearson"] is not None}
    if valid_corrs:
        print("\nModel Ranking by Pearson Correlation (Closest to Reference):")
        sorted_models = sorted(valid_corrs.items(), key=lambda x: x[1], reverse=True)
        for i, (model, corr) in enumerate(sorted_models, 1):
            print(f"{i}. {Path(model).stem}: Pearson = {corr:.4f}")
        closest_model = sorted_models[0][0]
        print(f"\nClosest model to reference: {Path(closest_model).stem} (Pearson = {sorted_models[0][1]:.4f})")


if __name__ == "__main__":
    main()
