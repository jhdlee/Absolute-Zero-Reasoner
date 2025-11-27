"""
Uncertainty Quantification Analysis Script

This script analyzes how well each UQ method can predict when the LLM label
differs from the ground truth execution label.

The ideal UQ method should:
- Assign HIGH uncertainty when labels DIFFER (LLM verification unreliable)
- Assign LOW uncertainty when labels MATCH (LLM verification reliable)

Metrics computed:
- AUROC: Area Under the ROC Curve for predicting label disagreement
- AUPRC: Area Under the Precision-Recall Curve
- Correlation: Point-biserial correlation between UQ metric and label disagreement
- Separation: How well the metric separates the two groups

Usage:
    python -m analysis.analyze_uncertainty --input_path results.json --output_dir analysis_output/
"""

import os
import sys
import json
import argparse
import math
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

try:
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
    from sklearn.metrics import average_precision_score
    from scipy.stats import pointbiserialr, mannwhitneyu
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn/scipy not available, using basic metrics only")


def load_results(input_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def extract_uq_data(tasks: List[Dict]) -> Tuple[Dict[str, List[float]], List[int]]:
    """
    Extract UQ metrics and labels from tasks.

    Returns:
        uq_metrics: Dict mapping metric name to list of values
        labels_differ: List of binary labels (1 if labels differ, 0 if match)
    """
    uq_metrics = {}
    labels_differ = []

    for task in tasks:
        # Skip tasks without UQ metrics or label info
        if not task.get("uncertainty_metrics"):
            continue
        if task.get("labels_match") is None:
            continue

        # Binary label: 1 if labels differ (we want to detect this)
        labels_differ.append(0 if task["labels_match"] else 1)

        # Extract each metric
        for metric_name, value in task["uncertainty_metrics"].items():
            if metric_name not in uq_metrics:
                uq_metrics[metric_name] = []

            # Handle None and NaN
            if value is None or (isinstance(value, float) and math.isnan(value)):
                uq_metrics[metric_name].append(None)
            else:
                uq_metrics[metric_name].append(value)

    return uq_metrics, labels_differ


def compute_auroc_manual(y_true: List[int], y_scores: List[float]) -> float:
    """
    Compute AUROC manually without sklearn.
    Uses the Wilcoxon-Mann-Whitney statistic interpretation.
    """
    pos_scores = [s for s, y in zip(y_scores, y_true) if y == 1]
    neg_scores = [s for s, y in zip(y_scores, y_true) if y == 0]

    if not pos_scores or not neg_scores:
        return float('nan')

    # Count pairs where positive score > negative score
    n_correct = 0
    n_ties = 0
    n_total = len(pos_scores) * len(neg_scores)

    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                n_correct += 1
            elif ps == ns:
                n_ties += 1

    # AUROC = (correct + 0.5 * ties) / total
    auroc = (n_correct + 0.5 * n_ties) / n_total
    return auroc


def analyze_single_metric(
    metric_name: str,
    values: List[Optional[float]],
    labels_differ: List[int],
    higher_means_uncertain: bool = True,
) -> Dict:
    """
    Analyze a single UQ metric for its ability to predict label disagreement.

    Args:
        metric_name: Name of the metric
        values: List of metric values (may contain None)
        labels_differ: Binary labels (1 if labels differ)
        higher_means_uncertain: If True, higher values = more uncertain
                               If False, flip the metric for AUROC computation

    Returns:
        Dict with analysis results
    """
    # Filter out None values
    valid_data = [(v, l) for v, l in zip(values, labels_differ) if v is not None]

    if len(valid_data) < 5:
        return {
            "metric": metric_name,
            "n_samples": len(valid_data),
            "error": "insufficient_data",
        }

    valid_values = [v for v, _ in valid_data]
    valid_labels = [l for _, l in valid_data]

    # Basic statistics
    match_values = [v for v, l in valid_data if l == 0]
    differ_values = [v for v, l in valid_data if l == 1]

    results = {
        "metric": metric_name,
        "n_samples": len(valid_data),
        "n_match": len(match_values),
        "n_differ": len(differ_values),
        "higher_means_uncertain": higher_means_uncertain,
    }

    if not match_values or not differ_values:
        results["error"] = "only_one_class"
        return results

    # Compute means and stds
    results["mean_when_match"] = np.mean(match_values)
    results["mean_when_differ"] = np.mean(differ_values)
    results["std_when_match"] = np.std(match_values)
    results["std_when_differ"] = np.std(differ_values)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(match_values) + np.var(differ_values)) / 2)
    if pooled_std > 0:
        cohens_d = (results["mean_when_differ"] - results["mean_when_match"]) / pooled_std
        results["cohens_d"] = cohens_d

    # For AUROC, we want: high score when labels differ
    # If higher values mean MORE uncertainty (good for detecting disagreement), use as-is
    # If higher values mean LESS uncertainty (like agreement_rate), flip the sign
    if higher_means_uncertain:
        scores_for_auroc = valid_values
    else:
        scores_for_auroc = [-v for v in valid_values]

    # Compute AUROC
    if HAS_SKLEARN:
        try:
            auroc = roc_auc_score(valid_labels, scores_for_auroc)
            results["auroc"] = auroc

            # Compute AUPRC
            auprc = average_precision_score(valid_labels, scores_for_auroc)
            results["auprc"] = auprc

            # ROC curve data
            fpr, tpr, thresholds = roc_curve(valid_labels, scores_for_auroc)
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            # Precision-Recall curve data
            precision, recall, _ = precision_recall_curve(valid_labels, scores_for_auroc)
            results["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

        except Exception as e:
            results["auroc_error"] = str(e)
    else:
        # Manual AUROC computation
        auroc = compute_auroc_manual(valid_labels, scores_for_auroc)
        results["auroc"] = auroc

    # Statistical significance (Mann-Whitney U test)
    if HAS_SKLEARN:
        try:
            # Test if differ_values are significantly higher/different from match_values
            statistic, pvalue = mannwhitneyu(differ_values, match_values, alternative='two-sided')
            results["mannwhitney_statistic"] = statistic
            results["mannwhitney_pvalue"] = pvalue
        except Exception as e:
            results["mannwhitney_error"] = str(e)

    # Point-biserial correlation
    if HAS_SKLEARN:
        try:
            corr, pvalue = pointbiserialr(valid_labels, valid_values)
            results["pointbiserial_corr"] = corr
            results["pointbiserial_pvalue"] = pvalue
        except Exception as e:
            results["correlation_error"] = str(e)

    return results


def get_metric_direction(metric_name: str) -> bool:
    """
    Determine if higher values mean more uncertainty.

    Returns True if higher = more uncertain (good for detecting disagreement)
    Returns False if higher = more certain (need to flip for AUROC)
    """
    # Metrics where HIGHER = MORE uncertain
    higher_is_uncertain = {
        "sc_entropy",
        "sc_entropy_normalized",
        "n_unique_answers",
        "lexical_ttr",
        "avg_pairwise_edit_dist",
        "perplexity",
        "logprob_variance",
    }

    # Metrics where HIGHER = MORE certain (lower uncertainty)
    higher_is_certain = {
        "agreement_rate",
        "seq_logprob",
        "seq_logprob_per_token",
        "verbalized_confidence",
    }

    if metric_name in higher_is_uncertain:
        return True
    elif metric_name in higher_is_certain:
        return False
    else:
        # Default assumption: higher = more uncertain
        return True


def plot_roc_curves(results: List[Dict], output_path: str):
    """Plot ROC curves for all metrics with distinct visual styles."""
    if not HAS_MATPLOTLIB:
        return

    # Filter results with ROC data
    valid_results = [r for r in results if "roc_curve" in r]

    if not valid_results:
        print("  No ROC curves to plot")
        return

    # Sort by AUROC
    valid_results.sort(key=lambda x: x.get("auroc", 0), reverse=True)

    # Distinct colors for better differentiation
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
        '#17becf',  # Cyan
        '#bcbd22',  # Yellow-green
    ]

    # Line styles for additional differentiation
    line_styles = ['-', '--', '-.', ':']

    # Markers (used sparingly for clarity)
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, result in enumerate(valid_results):
        fpr = result["roc_curve"]["fpr"]
        tpr = result["roc_curve"]["tpr"]
        auroc = result.get("auroc", 0)

        color = distinct_colors[i % len(distinct_colors)]
        linestyle = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # Use markers only at intervals for clarity
        markevery = max(1, len(fpr) // 8)

        label = f"{result['metric']} (AUROC={auroc:.3f})"
        ax.plot(
            fpr, tpr,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker=marker,
            markersize=6,
            markevery=markevery,
            label=label,
            alpha=0.85,
        )

    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUROC=0.5)')

    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(
        'ROC Curves: Predicting Label Disagreement\n'
        '(Higher AUROC = Better at detecting when LLM verification is unreliable)',
        fontsize=13
    )

    # Improved legend placement
    ax.legend(
        loc='lower right',
        fontsize=10,
        framealpha=0.9,
        edgecolor='gray',
    )

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ROC curves saved to: {output_path}")


def plot_metric_distributions(results: List[Dict], tasks: List[Dict], output_path: str):
    """Plot distributions of metrics split by label agreement."""
    if not HAS_MATPLOTLIB:
        return

    # Get metrics with enough data
    valid_results = [r for r in results if r.get("n_samples", 0) >= 5 and "error" not in r]

    if not valid_results:
        print("  No distributions to plot")
        return

    # Sort by AUROC
    valid_results.sort(key=lambda x: x.get("auroc", 0), reverse=True)

    n_metrics = len(valid_results)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    # Extract data for each metric
    uq_metrics, labels_differ = extract_uq_data(tasks)

    for i, result in enumerate(valid_results):
        ax = axes[i]
        metric_name = result["metric"]

        if metric_name not in uq_metrics:
            continue

        values = uq_metrics[metric_name]

        # Split by label
        match_values = [v for v, l in zip(values, labels_differ) if l == 0 and v is not None]
        differ_values = [v for v, l in zip(values, labels_differ) if l == 1 and v is not None]

        if not match_values or not differ_values:
            continue

        # Plot histograms
        bins = 20
        alpha = 0.6

        ax.hist(match_values, bins=bins, alpha=alpha, label=f'Match (n={len(match_values)})', color='green')
        ax.hist(differ_values, bins=bins, alpha=alpha, label=f'Differ (n={len(differ_values)})', color='red')

        auroc = result.get("auroc", 0)
        ax.set_title(f'{metric_name}\nAUROC={auroc:.3f}', fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(len(valid_results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('UQ Metric Distributions by Label Agreement\n(Green = Labels Match, Red = Labels Differ)', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Distributions saved to: {output_path}")


def plot_auroc_comparison(results: List[Dict], output_path: str):
    """Plot AUROC comparison bar chart."""
    if not HAS_MATPLOTLIB:
        return

    # Filter and sort
    valid_results = [r for r in results if "auroc" in r and not math.isnan(r["auroc"])]
    valid_results.sort(key=lambda x: x["auroc"], reverse=True)

    if not valid_results:
        print("  No AUROC values to plot")
        return

    metrics = [r["metric"] for r in valid_results]
    aurocs = [r["auroc"] for r in valid_results]

    # Color by performance
    colors = ['green' if a > 0.6 else 'orange' if a > 0.5 else 'red' for a in aurocs]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(len(metrics)), aurocs, color=colors, edgecolor='black')

    # Add value labels
    for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
        plt.text(auroc + 0.01, i, f'{auroc:.3f}', va='center', fontsize=9)

    plt.yticks(range(len(metrics)), metrics)
    plt.xlabel('AUROC', fontsize=12)
    plt.title('UQ Methods Ranked by AUROC\n(Ability to Predict Label Disagreement)', fontsize=12)
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Random (0.5)')
    plt.xlim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  AUROC comparison saved to: {output_path}")


def print_summary_table(results: List[Dict]):
    """Print a summary table of all UQ methods."""
    # Sort by AUROC
    sorted_results = sorted(
        [r for r in results if "auroc" in r],
        key=lambda x: x.get("auroc", 0),
        reverse=True
    )

    print("\n" + "=" * 100)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS SUMMARY")
    print("=" * 100)
    print("\nGoal: Find UQ methods that assign HIGH uncertainty when labels DIFFER")
    print("      (i.e., when LLM verification is unreliable)")
    print()

    # Header
    print(f"{'Metric':<30} {'AUROC':>8} {'AUPRC':>8} {'Cohen d':>8} {'Mean(Match)':>12} {'Mean(Differ)':>12} {'p-value':>10}")
    print("-" * 100)

    for r in sorted_results:
        metric = r.get("metric", "?")[:30]
        auroc = r.get("auroc", float('nan'))
        auprc = r.get("auprc", float('nan'))
        cohens_d = r.get("cohens_d", float('nan'))
        mean_match = r.get("mean_when_match", float('nan'))
        mean_differ = r.get("mean_when_differ", float('nan'))
        pvalue = r.get("mannwhitney_pvalue", float('nan'))

        auroc_str = f"{auroc:.3f}" if not math.isnan(auroc) else "N/A"
        auprc_str = f"{auprc:.3f}" if not math.isnan(auprc) else "N/A"
        cohens_d_str = f"{cohens_d:.3f}" if not math.isnan(cohens_d) else "N/A"
        mean_match_str = f"{mean_match:.4f}" if not math.isnan(mean_match) else "N/A"
        mean_differ_str = f"{mean_differ:.4f}" if not math.isnan(mean_differ) else "N/A"
        pvalue_str = f"{pvalue:.4f}" if not math.isnan(pvalue) else "N/A"

        print(f"{metric:<30} {auroc_str:>8} {auprc_str:>8} {cohens_d_str:>8} {mean_match_str:>12} {mean_differ_str:>12} {pvalue_str:>10}")

    print("-" * 100)
    print()
    print("Interpretation:")
    print("  - AUROC > 0.5: Metric can predict label disagreement better than random")
    print("  - AUROC > 0.7: Good predictive ability")
    print("  - AUROC > 0.8: Excellent predictive ability")
    print("  - Cohen's d > 0: Higher metric values when labels differ (as desired)")
    print("  - p-value < 0.05: Statistically significant difference between groups")
    print()

    # Identify best metrics
    best_metrics = [r for r in sorted_results if r.get("auroc", 0) > 0.6]
    if best_metrics:
        print("Best UQ methods for detecting unreliable LLM verification:")
        for r in best_metrics[:5]:
            print(f"  - {r['metric']}: AUROC={r.get('auroc', 0):.3f}")
    print()


def save_analysis_results(results: List[Dict], output_path: str):
    """Save detailed analysis results to JSON."""
    # Convert numpy types to Python types
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if k in ["roc_curve", "pr_curve"]:
                continue  # Skip curve data for JSON (too large)
            if isinstance(v, np.floating):
                val = float(v)
                clean_r[k] = None if math.isnan(val) or math.isinf(val) else val
            elif isinstance(v, np.integer):
                clean_r[k] = int(v)
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_r[k] = None
            else:
                clean_r[k] = v
        clean_results.append(clean_r)

    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"  Analysis results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze UQ methods for predicting label disagreement")
    parser.add_argument("--input_path", type=str, required=True, help="Path to results JSON from generate_tasks.py")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for plots and results")
    args = parser.parse_args()

    print("=" * 80)
    print("UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"\nLoading results from: {args.input_path}")
    tasks = load_results(args.input_path)
    print(f"  Loaded {len(tasks)} tasks")

    # Extract UQ data
    uq_metrics, labels_differ = extract_uq_data(tasks)

    n_with_uq = len(labels_differ)
    print(f"  Tasks with UQ metrics: {n_with_uq}")

    if n_with_uq == 0:
        print("\nError: No tasks with UQ metrics found.")
        print("  Make sure generate_tasks.py was run with --n_samples > 1 for self-consistency")
        print("  Check that tasks have 'uncertainty_metrics' key")
        # Debug: show first task's keys
        if tasks:
            print(f"\n  Debug - First task keys: {list(tasks[0].keys())}")
        return

    n_differ = sum(labels_differ)
    n_match = n_with_uq - n_differ

    print(f"  Labels match: {n_match} ({100*n_match/n_with_uq:.1f}%)")
    print(f"  Labels differ: {n_differ} ({100*n_differ/n_with_uq:.1f}%)")

    if n_differ == 0 or n_match == 0:
        print("\nWarning: Only one class present. AUROC analysis not possible.")
        print("Need both matching and differing labels to evaluate UQ methods.")

    # Analyze each metric
    print("\nAnalyzing UQ metrics...")
    results = []

    for metric_name, values in uq_metrics.items():
        higher_means_uncertain = get_metric_direction(metric_name)
        result = analyze_single_metric(
            metric_name=metric_name,
            values=values,
            labels_differ=labels_differ,
            higher_means_uncertain=higher_means_uncertain,
        )
        results.append(result)
        print(f"  - {metric_name}: AUROC={result.get('auroc', 'N/A')}")

    # Print summary table
    print_summary_table(results)

    # Generate plots
    if HAS_MATPLOTLIB:
        print("Generating plots...")

        # ROC curves
        plot_roc_curves(
            results,
            os.path.join(args.output_dir, "roc_curves.png")
        )

        # Metric distributions
        plot_metric_distributions(
            results,
            tasks,
            os.path.join(args.output_dir, "metric_distributions.png")
        )

        # AUROC comparison
        plot_auroc_comparison(
            results,
            os.path.join(args.output_dir, "auroc_comparison.png")
        )

    # Save detailed results
    save_analysis_results(
        results,
        os.path.join(args.output_dir, "uq_analysis_results.json")
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {args.output_dir}/")
    print("  - roc_curves.png: ROC curves for all UQ methods")
    print("  - metric_distributions.png: Distribution histograms")
    print("  - auroc_comparison.png: AUROC bar chart comparison")
    print("  - uq_analysis_results.json: Detailed numerical results")


if __name__ == "__main__":
    main()
