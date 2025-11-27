"""
Comparison Analysis for Different Verification Modes

This script analyzes and compares the results from training with:
1. Full Execution (ground truth)
2. Full LLM (LLM-as-a-judge)
3. Adaptive (uncertainty-guided)

Usage:
    python -m analysis.compare_verification_modes --input_dir training_results --output_dir comparison_output
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


# ==============================================================================
# Data Loading
# ==============================================================================

def load_experiment_results(input_dir: str) -> Dict[str, Dict]:
    """Load results from all three experiments."""
    results = {}

    modes = ["full_execution", "full_llm", "adaptive"]

    for mode in modes:
        path = os.path.join(input_dir, f"experiment_{mode}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[mode] = json.load(f)
            print(f"  Loaded {mode} results")
        else:
            print(f"  Warning: {mode} results not found at {path}")

    # Also try loading comparison.json if it exists
    comparison_path = os.path.join(input_dir, "comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
            for mode, data in comparison_data.items():
                if mode not in results:
                    results[mode] = data

    return results


def load_detailed_results(input_dir: str) -> Dict[str, List[Dict]]:
    """Load detailed per-task results."""
    detailed = {}

    modes = ["full_execution", "full_llm", "adaptive"]

    for mode in modes:
        path = os.path.join(input_dir, f"detailed_{mode}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                detailed[mode] = json.load(f)

    return detailed


# ==============================================================================
# Analysis Functions
# ==============================================================================

def compute_comparison_metrics(results: Dict[str, Dict]) -> Dict:
    """Compute comparison metrics across all modes."""
    metrics = {}

    for mode, data in results.items():
        stats = data.get("aggregate_stats", {})
        metrics[mode] = {
            "total_tasks": stats.get("total_tasks", 0),
            "total_time_s": stats.get("total_time_s", 0),
            "mean_reward": stats.get("mean_reward", 0),
            "std_reward": stats.get("std_reward", 0),
            "ground_truth_accuracy": stats.get("ground_truth_accuracy"),
            "false_positive_rate": stats.get("false_positive_rate"),
            "false_negative_rate": stats.get("false_negative_rate"),
        }

        # Verification breakdown
        vb = stats.get("verification_breakdown", {})
        metrics[mode]["execution_count"] = vb.get("execution_verifications", 0)
        metrics[mode]["llm_count"] = vb.get("llm_verifications", 0)
        metrics[mode]["execution_time_ms"] = vb.get("execution_time_ms", 0)
        metrics[mode]["llm_time_ms"] = vb.get("llm_time_ms", 0)

        # Compute time per task
        total = metrics[mode]["execution_count"] + metrics[mode]["llm_count"]
        if total > 0:
            metrics[mode]["time_per_task_ms"] = (
                (metrics[mode]["execution_time_ms"] + metrics[mode]["llm_time_ms"]) / total
            )
        else:
            metrics[mode]["time_per_task_ms"] = 0

    return metrics


def compute_reward_quality_tradeoff(results: Dict[str, Dict]) -> Dict:
    """
    Compute the tradeoff between reward quality and computational cost.

    Reward quality is measured by alignment with ground truth (execution).
    Cost is measured by total verification time.
    """
    tradeoff = {}

    # Use full_execution as baseline (100% accuracy by definition)
    baseline_time = 0
    if "full_execution" in results:
        stats = results["full_execution"].get("aggregate_stats", {})
        baseline_time = stats.get("total_time_s", 0)

    for mode, data in results.items():
        stats = data.get("aggregate_stats", {})

        tradeoff[mode] = {
            "total_time_s": stats.get("total_time_s", 0),
            "time_ratio": (
                stats.get("total_time_s", 0) / baseline_time
                if baseline_time > 0 else 1.0
            ),
        }

        # Quality metric
        if mode == "full_execution":
            tradeoff[mode]["quality"] = 1.0  # Ground truth by definition
        else:
            gt_acc = stats.get("ground_truth_accuracy")
            if gt_acc is not None:
                tradeoff[mode]["quality"] = gt_acc
            else:
                tradeoff[mode]["quality"] = None

        # Efficiency score: quality / time_ratio (higher is better)
        if tradeoff[mode]["quality"] is not None and tradeoff[mode]["time_ratio"] > 0:
            tradeoff[mode]["efficiency"] = (
                tradeoff[mode]["quality"] / tradeoff[mode]["time_ratio"]
            )
        else:
            tradeoff[mode]["efficiency"] = None

    return tradeoff


def analyze_reward_distribution(detailed: Dict[str, List[Dict]]) -> Dict:
    """Analyze reward distributions for each mode."""
    distributions = {}

    for mode, tasks in detailed.items():
        rewards = [t.get("reward", 0) for t in tasks if "reward" in t]

        if rewards:
            distributions[mode] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards),
                "median": np.median(rewards),
                "positive_fraction": np.mean([r > 0 for r in rewards]),
                "n_samples": len(rewards),
            }

    return distributions


def analyze_label_agreement(detailed: Dict[str, List[Dict]]) -> Dict:
    """Analyze agreement between different verification methods and ground truth."""
    agreement = {}

    for mode, tasks in detailed.items():
        if mode == "full_execution":
            # Execution is ground truth
            agreement[mode] = {
                "agreement_with_gt": 1.0,
                "false_positives": 0,
                "false_negatives": 0,
            }
            continue

        agreements = 0
        false_pos = 0
        false_neg = 0
        total = 0

        for t in tasks:
            gt = t.get("ground_truth_correct")
            verified = t.get("verification_correct")

            if gt is None or verified is None:
                continue

            total += 1
            if gt == verified:
                agreements += 1
            elif verified and not gt:
                false_pos += 1
            elif not verified and gt:
                false_neg += 1

        if total > 0:
            agreement[mode] = {
                "agreement_with_gt": agreements / total,
                "false_positive_rate": false_pos / total,
                "false_negative_rate": false_neg / total,
                "n_samples": total,
            }

    return agreement


def analyze_uncertainty_correlation(detailed: Dict[str, List[Dict]]) -> Optional[Dict]:
    """Analyze how uncertainty correlates with verification errors."""
    # Only applicable for adaptive mode
    if "adaptive" not in detailed:
        return None

    tasks = detailed["adaptive"]

    uncertainties = []
    errors = []

    for t in tasks:
        uncertainty = t.get("uncertainty")
        gt = t.get("ground_truth_correct")
        verified = t.get("verification_correct")

        if uncertainty is None or gt is None or verified is None:
            continue

        uncertainties.append(uncertainty)
        errors.append(1 if gt != verified else 0)

    if len(uncertainties) < 10:
        return None

    # Compute correlation
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)

    correlation = np.corrcoef(uncertainties, errors)[0, 1]

    # Bin analysis
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    bin_error_rates = []

    for i in range(len(bins) - 1):
        mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
        if mask.sum() > 0:
            bin_error_rates.append({
                "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                "error_rate": errors[mask].mean(),
                "n_samples": mask.sum(),
            })

    return {
        "correlation": correlation,
        "bin_analysis": bin_error_rates,
    }


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_reward_comparison(metrics: Dict, output_dir: str):
    """Plot reward comparison across modes."""
    if not HAS_MATPLOTLIB:
        return

    modes = list(metrics.keys())
    means = [metrics[m]["mean_reward"] for m in modes]
    stds = [metrics[m]["std_reward"] for m in modes]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"full_execution": "#2ecc71", "full_llm": "#e74c3c", "adaptive": "#3498db"}
    bar_colors = [colors.get(m, "#95a5a6") for m in modes]

    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)

    ax.set_xlabel("Verification Mode", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Reward Comparison Across Verification Modes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.3f}",
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=150)
    plt.close()


def plot_time_comparison(metrics: Dict, output_dir: str):
    """Plot time comparison across modes."""
    if not HAS_MATPLOTLIB:
        return

    modes = list(metrics.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total time
    ax1 = axes[0]
    times = [metrics[m]["total_time_s"] for m in modes]
    colors = {"full_execution": "#2ecc71", "full_llm": "#e74c3c", "adaptive": "#3498db"}
    bar_colors = [colors.get(m, "#95a5a6") for m in modes]

    x = np.arange(len(modes))
    bars = ax1.bar(x, times, color=bar_colors, alpha=0.8)

    ax1.set_xlabel("Verification Mode", fontsize=12)
    ax1.set_ylabel("Total Time (seconds)", fontsize=12)
    ax1.set_title("Total Verification Time", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", "\n") for m in modes])

    for bar, t in zip(bars, times):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.1f}s",
            ha='center', va='bottom', fontsize=10
        )

    # Time per task
    ax2 = axes[1]
    times_per_task = [metrics[m]["time_per_task_ms"] for m in modes]

    bars = ax2.bar(x, times_per_task, color=bar_colors, alpha=0.8)

    ax2.set_xlabel("Verification Mode", fontsize=12)
    ax2.set_ylabel("Time per Task (ms)", fontsize=12)
    ax2.set_title("Average Time per Task", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", "\n") for m in modes])

    for bar, t in zip(bars, times_per_task):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{t:.1f}ms",
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=150)
    plt.close()


def plot_accuracy_comparison(metrics: Dict, output_dir: str):
    """Plot ground truth accuracy comparison."""
    if not HAS_MATPLOTLIB:
        return

    # Only for modes that have ground truth accuracy
    modes = []
    accuracies = []
    fp_rates = []
    fn_rates = []

    for mode, m in metrics.items():
        if m.get("ground_truth_accuracy") is not None:
            modes.append(mode)
            accuracies.append(m["ground_truth_accuracy"])
            fp_rates.append(m.get("false_positive_rate", 0))
            fn_rates.append(m.get("false_negative_rate", 0))

    if not modes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(modes))
    width = 0.25

    bars1 = ax.bar(x - width, accuracies, width, label='Agreement with GT', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, fp_rates, width, label='False Positive Rate', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, fn_rates, width, label='False Negative Rate', color='#f39c12', alpha=0.8)

    ax.set_xlabel("Verification Mode", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Ground Truth Alignment Analysis", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes])
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.1%}",
                ha='center', va='bottom', fontsize=9
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()


def plot_verification_breakdown(metrics: Dict, output_dir: str):
    """Plot breakdown of verification methods used."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Execution", "LLM"]

    for ax, (mode, m) in zip(axes, metrics.items()):
        exec_count = m.get("execution_count", 0)
        llm_count = m.get("llm_count", 0)

        sizes = [exec_count, llm_count]
        if sum(sizes) == 0:
            continue

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
        )

        ax.set_title(f"{mode.replace('_', ' ').title()}\n(n={sum(sizes)})", fontsize=12)

    plt.suptitle("Verification Method Breakdown", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "verification_breakdown.png"), dpi=150)
    plt.close()


def plot_tradeoff(tradeoff: Dict, output_dir: str):
    """Plot quality vs cost tradeoff."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"full_execution": "#2ecc71", "full_llm": "#e74c3c", "adaptive": "#3498db"}
    markers = {"full_execution": "s", "full_llm": "o", "adaptive": "^"}

    for mode, data in tradeoff.items():
        if data["quality"] is None:
            continue

        ax.scatter(
            data["time_ratio"],
            data["quality"],
            s=200,
            c=colors.get(mode, "#95a5a6"),
            marker=markers.get(mode, "o"),
            label=mode.replace("_", " ").title(),
            alpha=0.8,
            edgecolors='black',
        )

        # Add label
        ax.annotate(
            mode.replace("_", " ").title(),
            (data["time_ratio"], data["quality"]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
        )

    ax.set_xlabel("Time Ratio (relative to full execution)", fontsize=12)
    ax.set_ylabel("Quality (Ground Truth Accuracy)", fontsize=12)
    ax.set_title("Quality vs Cost Tradeoff", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect accuracy')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline time')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_cost_tradeoff.png"), dpi=150)
    plt.close()


def plot_uncertainty_analysis(uncertainty_data: Dict, output_dir: str):
    """Plot uncertainty vs error analysis."""
    if not HAS_MATPLOTLIB or uncertainty_data is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = uncertainty_data.get("bin_analysis", [])
    if not bins:
        return

    x_labels = [b["bin"] for b in bins]
    error_rates = [b["error_rate"] for b in bins]
    sample_counts = [b["n_samples"] for b in bins]

    x = np.arange(len(x_labels))
    bars = ax.bar(x, error_rates, color='#e74c3c', alpha=0.8)

    ax.set_xlabel("Uncertainty Bin", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title(
        f"Error Rate vs Uncertainty\n(Correlation: {uncertainty_data['correlation']:.3f})",
        fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    # Add sample counts
    for bar, count in zip(bars, sample_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={count}",
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "uncertainty_analysis.png"), dpi=150)
    plt.close()


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(
    metrics: Dict,
    tradeoff: Dict,
    distributions: Dict,
    agreement: Dict,
    uncertainty: Optional[Dict],
    output_dir: str,
):
    """Generate a text report summarizing the comparison."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("VERIFICATION MODE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary table
    report_lines.append("SUMMARY METRICS")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Mode':<20} {'Tasks':<10} {'Time(s)':<10} {'Reward':<15} {'GT Acc':<10}")
    report_lines.append("-" * 65)

    for mode, m in metrics.items():
        gt_acc = m.get("ground_truth_accuracy")
        gt_str = f"{gt_acc:.1%}" if gt_acc is not None else "N/A"

        report_lines.append(
            f"{mode:<20} {m['total_tasks']:<10} {m['total_time_s']:<10.1f} "
            f"{m['mean_reward']:.4f}±{m['std_reward']:.4f}  {gt_str:<10}"
        )

    report_lines.append("")

    # Verification breakdown
    report_lines.append("VERIFICATION BREAKDOWN")
    report_lines.append("-" * 40)

    for mode, m in metrics.items():
        exec_count = m.get("execution_count", 0)
        llm_count = m.get("llm_count", 0)
        total = exec_count + llm_count

        if total > 0:
            report_lines.append(f"{mode}:")
            report_lines.append(f"  Execution: {exec_count} ({100*exec_count/total:.1f}%)")
            report_lines.append(f"  LLM: {llm_count} ({100*llm_count/total:.1f}%)")
            report_lines.append(f"  Exec time: {m.get('execution_time_ms', 0):.0f}ms")
            report_lines.append(f"  LLM time: {m.get('llm_time_ms', 0):.0f}ms")
            report_lines.append("")

    # Quality vs Cost tradeoff
    report_lines.append("QUALITY VS COST TRADEOFF")
    report_lines.append("-" * 40)

    for mode, t in tradeoff.items():
        quality = t.get("quality")
        efficiency = t.get("efficiency")

        report_lines.append(f"{mode}:")
        report_lines.append(f"  Time ratio: {t['time_ratio']:.2f}x")
        report_lines.append(f"  Quality: {quality:.1%}" if quality else "  Quality: N/A")
        report_lines.append(f"  Efficiency: {efficiency:.3f}" if efficiency else "  Efficiency: N/A")
        report_lines.append("")

    # Ground truth alignment
    if agreement:
        report_lines.append("GROUND TRUTH ALIGNMENT")
        report_lines.append("-" * 40)

        for mode, a in agreement.items():
            if mode == "full_execution":
                continue
            report_lines.append(f"{mode}:")
            report_lines.append(f"  Agreement rate: {a.get('agreement_with_gt', 0):.1%}")
            report_lines.append(f"  False positive rate: {a.get('false_positive_rate', 0):.1%}")
            report_lines.append(f"  False negative rate: {a.get('false_negative_rate', 0):.1%}")
            report_lines.append("")

    # Uncertainty analysis
    if uncertainty:
        report_lines.append("UNCERTAINTY ANALYSIS (Adaptive Mode)")
        report_lines.append("-" * 40)
        report_lines.append(f"Correlation with error: {uncertainty['correlation']:.3f}")
        report_lines.append("")
        report_lines.append("Error rate by uncertainty bin:")
        for bin_data in uncertainty.get("bin_analysis", []):
            report_lines.append(
                f"  {bin_data['bin']}: {bin_data['error_rate']:.1%} (n={bin_data['n_samples']})"
            )
        report_lines.append("")

    # Key findings
    report_lines.append("KEY FINDINGS")
    report_lines.append("-" * 40)

    # Find best efficiency
    best_efficiency_mode = None
    best_efficiency = 0
    for mode, t in tradeoff.items():
        if t.get("efficiency") and t["efficiency"] > best_efficiency:
            best_efficiency = t["efficiency"]
            best_efficiency_mode = mode

    if best_efficiency_mode:
        report_lines.append(f"• Best efficiency: {best_efficiency_mode} (score: {best_efficiency:.3f})")

    # Time savings
    if "full_execution" in metrics and "adaptive" in metrics:
        exec_time = metrics["full_execution"]["total_time_s"]
        adaptive_time = metrics["adaptive"]["total_time_s"]
        if exec_time > 0:
            savings = (exec_time - adaptive_time) / exec_time * 100
            report_lines.append(f"• Adaptive time savings vs full execution: {savings:.1f}%")

    # Accuracy comparison
    if "full_llm" in agreement and "adaptive" in agreement:
        llm_acc = agreement["full_llm"].get("agreement_with_gt", 0)
        adaptive_acc = agreement["adaptive"].get("agreement_with_gt", 0)
        report_lines.append(f"• Full LLM accuracy: {llm_acc:.1%}")
        report_lines.append(f"• Adaptive accuracy: {adaptive_acc:.1%}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    report_text = "\n".join(report_lines)

    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare verification mode results")
    parser.add_argument(
        "--input_dir", type=str, default="training_results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output_dir", type=str, default="comparison_output",
        help="Directory for comparison outputs"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VERIFICATION MODE COMPARISON ANALYSIS")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print("\nLoading experiment results...")
    results = load_experiment_results(args.input_dir)

    if not results:
        print("Error: No experiment results found!")
        return

    print(f"  Loaded results for {len(results)} modes: {list(results.keys())}")

    # Load detailed results
    detailed = load_detailed_results(args.input_dir)

    # Compute metrics
    print("\nComputing comparison metrics...")
    metrics = compute_comparison_metrics(results)
    tradeoff = compute_reward_quality_tradeoff(results)
    distributions = analyze_reward_distribution(detailed) if detailed else {}
    agreement = analyze_label_agreement(detailed) if detailed else {}
    uncertainty = analyze_uncertainty_correlation(detailed) if detailed else None

    # Generate plots
    print("\nGenerating plots...")
    plot_reward_comparison(metrics, args.output_dir)
    plot_time_comparison(metrics, args.output_dir)
    plot_accuracy_comparison(metrics, args.output_dir)
    plot_verification_breakdown(metrics, args.output_dir)
    plot_tradeoff(tradeoff, args.output_dir)
    plot_uncertainty_analysis(uncertainty, args.output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(
        metrics=metrics,
        tradeoff=tradeoff,
        distributions=distributions,
        agreement=agreement,
        uncertainty=uncertainty,
        output_dir=args.output_dir,
    )

    # Save metrics as JSON
    all_metrics = {
        "comparison_metrics": metrics,
        "tradeoff": tradeoff,
        "distributions": distributions,
        "agreement": agreement,
        "uncertainty": uncertainty,
    }

    metrics_path = os.path.join(args.output_dir, "all_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
