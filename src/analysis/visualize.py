"""
Visualization and analysis pipeline.

Generates all plots for the paper:
1. Corruption % vs Alignment Score
2. Corruption % vs Reasoning Accuracy
3. Corruption % vs Drift Score
4. Corruption % vs CKA Similarity
5. Recovery curves
6. Representation geometry heatmaps
7. Phase transition detection
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# Paper-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

CORRUPTION_RATIOS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
COLORS = {
    'toxic': '#e74c3c',
    'misinformation': '#3498db',
    'semantic_noise': '#2ecc71',
    'slang_compression': '#f39c12',
}
MARKERS = {
    'toxic': 'o',
    'misinformation': 's',
    'semantic_noise': '^',
    'slang_compression': 'D',
}


def load_results(results_dir: str) -> dict:
    """Load all evaluation results."""
    results_path = Path(results_dir)
    all_results = {}

    for f in results_path.glob("*.json"):
        with open(f) as fp:
            all_results[f.stem] = json.load(fp)

    return all_results


def _extract_metric_curve(results: dict, corruption_type: str, metric_path: list[str]) -> tuple[list, list]:
    """Extract a metric across corruption ratios for a given corruption type."""
    ratios = []
    values = []

    for ratio in CORRUPTION_RATIOS:
        if ratio == 0.0:
            key = "baseline_clean"
        else:
            key = f"{corruption_type}_{ratio}"

        if key in results:
            data = results[key]
            # Navigate the metric path
            val = data
            for p in metric_path:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    val = None
                    break

            if val is not None:
                ratios.append(ratio)
                values.append(float(val))

    return ratios, values


def plot_alignment_vs_corruption(results: dict, output_dir: str):
    """Plot 1: Corruption % vs Alignment Score for each corruption type."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["alignment", "alignment_score"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)

    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Alignment Score')
    ax.set_title('Alignment Stability Under Progressive Corruption')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/alignment_vs_corruption.pdf")
    plt.savefig(f"{output_dir}/alignment_vs_corruption.png")
    plt.close()
    print("  Saved alignment_vs_corruption")


def plot_reasoning_vs_corruption(results: dict, output_dir: str):
    """Plot 2: Capability metrics vs Corruption."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: Perplexity (lower = better)
    ax = axes[0]
    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["reasoning", "perplexity", "mean_perplexity"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)
    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Perplexity (lower = better)')
    ax.set_title('Language Modeling Perplexity')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='best')

    # Middle: Next-token accuracy
    ax = axes[1]
    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["reasoning", "next_token", "top1_accuracy"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)
    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Factual Next-Token Prediction')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')

    # Right: Coherence score
    ax = axes[2]
    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["reasoning", "coherence", "coherence_score"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)
    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Coherence Score')
    ax.set_title('Generation Coherence')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/capability_vs_corruption.pdf")
    plt.savefig(f"{output_dir}/capability_vs_corruption.png")
    plt.close()
    print("  Saved capability_vs_corruption")


def plot_drift_vs_corruption(results: dict, output_dir: str):
    """Plot 3: Corruption % vs KL Divergence Drift Score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Overall drift
    ax = axes[0]
    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["drift", "mean_drift"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)

    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Behavioral Drift (KL Divergence)')
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc='best')

    # Right: Category-specific drift for toxic corruption
    ax = axes[1]
    categories = ["safety", "factual", "reasoning", "instruction", "creative", "ethical"]
    cat_colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    for cat, color in zip(categories, cat_colors):
        ratios_cat = []
        scores_cat = []
        for ratio in CORRUPTION_RATIOS:
            if ratio == 0.0:
                key = "baseline_clean"
            else:
                key = f"toxic_{ratio}"

            if key in results and "drift" in results[key]:
                cat_drift = results[key]["drift"].get("category_drift", {})
                if cat in cat_drift:
                    ratios_cat.append(ratio)
                    scores_cat.append(cat_drift[cat]["mean_kl"])

        if ratios_cat:
            ax.plot(ratios_cat, scores_cat, marker='o', color=color,
                    label=cat.title(), linewidth=2, markersize=6)

    ax.set_xlabel('Corruption Ratio (Toxic)')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title('Category-Specific Drift Under Toxic Corruption')
    ax.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/drift_vs_corruption.pdf")
    plt.savefig(f"{output_dir}/drift_vs_corruption.png")
    plt.close()
    print("  Saved drift_vs_corruption")


def plot_cka_vs_corruption(results: dict, output_dir: str):
    """Plot 4: Corruption % vs CKA Similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Mean CKA across all layers
    ax = axes[0]
    for ctype in COLORS:
        ratios, scores = _extract_metric_curve(results, ctype, ["geometry", "summary", "mean_cka"])
        if ratios:
            ax.plot(ratios, scores, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=8)

    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Mean CKA Similarity')
    ax.set_title('Representation Similarity (CKA) vs Corruption')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best')

    # Right: Layer-wise CKA for different corruption levels (toxic)
    ax = axes[1]
    cmap = plt.cm.RdYlGn_r
    for i, ratio in enumerate([0.10, 0.25, 0.50, 0.75, 1.0]):
        key = f"toxic_{ratio}"
        if key in results and "geometry" in results[key]:
            cka_per_layer = results[key]["geometry"]["summary"].get("cka_per_layer", {})
            if cka_per_layer:
                layers = sorted(int(k) for k in cka_per_layer.keys())
                values = [cka_per_layer[str(l)] for l in layers]
                color = cmap(i / 4)
                ax.plot(layers, values, marker='o', color=color,
                        label=f'{ratio:.0%}', linewidth=1.5, markersize=4)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('CKA Similarity')
    ax.set_title('Layer-wise CKA (Toxic Corruption)')
    ax.legend(title='Corruption %', loc='best')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cka_vs_corruption.pdf")
    plt.savefig(f"{output_dir}/cka_vs_corruption.png")
    plt.close()
    print("  Saved cka_vs_corruption")


def plot_recovery_curves(results: dict, output_dir: str):
    """Plot 5: Recovery curves after re-alignment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("Alignment Score", ["alignment", "alignment_score"]),
        ("Perplexity", ["reasoning", "perplexity", "mean_perplexity"]),
        ("Mean Drift (KL)", ["drift", "mean_drift"]),
    ]

    recovery_ratios = [0.50, 0.75, 1.0]

    for ax, (metric_name, metric_path) in zip(axes, metrics):
        for ctype in COLORS:
            x_labels = []
            y_values = []

            def _get_val(data, path):
                v = data
                for p in path:
                    if isinstance(v, dict) and p in v:
                        v = v[p]
                    else:
                        return None
                return float(v) if isinstance(v, (int, float)) else None

            # Baseline
            if "baseline_clean" in results:
                val = _get_val(results["baseline_clean"], metric_path)
                if val is not None:
                    x_labels.append("Clean\nBaseline")
                    y_values.append(val)
                elif metric_name == "Mean Drift (KL)":
                    x_labels.append("Clean\nBaseline")
                    y_values.append(0.0)  # drift from self is 0

            for ratio in recovery_ratios:
                # Corrupted
                key = f"{ctype}_{ratio}"
                if key in results:
                    val = _get_val(results[key], metric_path)
                    if val is not None:
                        x_labels.append(f"Corrupt\n{ratio:.0%}")
                        y_values.append(val)

                # Recovered
                rkey = f"{ctype}_{ratio}_recovered"
                if rkey in results:
                    val = _get_val(results[rkey], metric_path)
                    if val is not None:
                        x_labels.append(f"Recover\n{ratio:.0%}")
                        y_values.append(float(val))

            if x_labels and ctype == "toxic":  # Primary corruption type
                ax.plot(range(len(y_values)), y_values, marker='o',
                        color=COLORS[ctype], linewidth=2, markersize=8)
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, fontsize=9)

        ax.set_ylabel(metric_name)
        ax.set_title(f'Recovery: {metric_name}')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/recovery_curves.pdf")
    plt.savefig(f"{output_dir}/recovery_curves.png")
    plt.close()
    print("  Saved recovery_curves")


def plot_cka_heatmap(results: dict, output_dir: str):
    """Plot 6: Cross-layer CKA heatmaps."""
    for ratio in [0.25, 0.50, 1.0]:
        key = f"toxic_{ratio}"
        if key not in results or "geometry" not in results[key]:
            continue

        cross_cka = results[key]["geometry"]["cka"].get("cross_layer_cka")
        if cross_cka is None:
            continue

        matrix = np.array(cross_cka)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(matrix, ax=ax, cmap='viridis', vmin=0, vmax=1,
                    xticklabels=5, yticklabels=5)
        ax.set_xlabel('Corrupted Model Layer')
        ax.set_ylabel('Baseline Model Layer')
        ax.set_title(f'Cross-Layer CKA (Toxic {ratio:.0%} Corruption)')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/cka_heatmap_{ratio}.pdf")
        plt.savefig(f"{output_dir}/cka_heatmap_{ratio}.png")
        plt.close()
        print(f"  Saved cka_heatmap_{ratio}")


def plot_phase_transition_analysis(results: dict, output_dir: str):
    """
    Plot 7: Phase transition analysis.

    Compute the derivative of metrics w.r.t. corruption ratio
    to detect sudden changes (potential phase transitions).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metric_configs = [
        (axes[0, 0], "Alignment Score", ["alignment", "alignment_score"]),
        (axes[0, 1], "Perplexity", ["reasoning", "perplexity", "mean_perplexity"]),
        (axes[1, 0], "KL Drift", ["drift", "mean_drift"]),
        (axes[1, 1], "CKA Similarity", ["geometry", "summary", "mean_cka"]),
    ]

    for ax, metric_name, metric_path in metric_configs:
        for ctype in COLORS:
            ratios, values = _extract_metric_curve(results, ctype, metric_path)
            if len(ratios) < 3:
                continue

            # Compute finite differences (discrete derivative)
            ratios_arr = np.array(ratios)
            values_arr = np.array(values)
            dv = np.diff(values_arr) / np.diff(ratios_arr)
            mid_ratios = (ratios_arr[:-1] + ratios_arr[1:]) / 2

            ax.plot(mid_ratios, dv, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=6)

        ax.set_xlabel('Corruption Ratio')
        ax.set_ylabel(f'd({metric_name})/d(ratio)')
        ax.set_title(f'Rate of Change: {metric_name}')
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('Phase Transition Analysis: Rate of Change of Metrics', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase_transition_analysis.pdf")
    plt.savefig(f"{output_dir}/phase_transition_analysis.png")
    plt.close()
    print("  Saved phase_transition_analysis")


def plot_composite_dashboard(results: dict, output_dir: str):
    """
    Plot 8: Composite dashboard showing all metrics side-by-side.
    This is the main figure for the paper.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Main metrics
    metric_configs = [
        (axes[0, 0], "Alignment Score", ["alignment", "alignment_score"], True),
        (axes[0, 1], "Perplexity", ["reasoning", "perplexity", "mean_perplexity"], False),
        (axes[0, 2], "KL Divergence (Drift)", ["drift", "mean_drift"], False),
    ]

    for ax, title, path, invert_bad in metric_configs:
        for ctype in COLORS:
            ratios, values = _extract_metric_curve(results, ctype, path)
            if ratios:
                ax.plot(ratios, values, marker=MARKERS[ctype], color=COLORS[ctype],
                        label=ctype.replace('_', ' ').title(), linewidth=2, markersize=7)

        ax.set_xlabel('Corruption Ratio')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc='best', fontsize=9)

    # Row 2: Geometry metrics
    geo_configs = [
        (axes[1, 0], "CKA Similarity", ["geometry", "summary", "mean_cka"]),
        (axes[1, 1], "Cosine Similarity", ["geometry", "summary", "mean_cosine"]),
    ]

    for ax, title, path in geo_configs:
        for ctype in COLORS:
            ratios, values = _extract_metric_curve(results, ctype, path)
            if ratios:
                ax.plot(ratios, values, marker=MARKERS[ctype], color=COLORS[ctype],
                        label=ctype.replace('_', ' ').title(), linewidth=2, markersize=7)

        ax.set_xlabel('Corruption Ratio')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(loc='best', fontsize=9)

    # Bottom right: Refusal rate
    ax = axes[1, 2]
    for ctype in COLORS:
        ratios, values = _extract_metric_curve(results, ctype, ["alignment", "refusal", "refusal_rate"])
        if ratios:
            ax.plot(ratios, values, marker=MARKERS[ctype], color=COLORS[ctype],
                    label=ctype.replace('_', ' ').title(), linewidth=2, markersize=7)

    ax.set_xlabel('Corruption Ratio')
    ax.set_ylabel('Refusal Rate')
    ax.set_title('Safety Refusal Rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=9)

    plt.suptitle('Alignment Stability Under Progressive Data Corruption', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/composite_dashboard.pdf")
    plt.savefig(f"{output_dir}/composite_dashboard.png")
    plt.close()
    print("  Saved composite_dashboard")


def generate_all_plots(results_dir: str, output_dir: str):
    """Generate all paper plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(results_dir)

    if not results:
        print("No results found! Run experiments first.")
        return

    print(f"Found {len(results)} result files")
    print("Generating plots...")

    plot_alignment_vs_corruption(results, output_dir)
    plot_reasoning_vs_corruption(results, output_dir)
    plot_drift_vs_corruption(results, output_dir)
    plot_cka_vs_corruption(results, output_dir)
    plot_recovery_curves(results, output_dir)
    plot_cka_heatmap(results, output_dir)
    plot_phase_transition_analysis(results, output_dir)
    plot_composite_dashboard(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")
