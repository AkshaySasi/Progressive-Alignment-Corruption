"""
Main experiment orchestrator.

Runs the full experimental pipeline:
1. Build datasets (clean + corrupted at each ratio)
2. Train models (baseline + all corruption variants + recovery)
3. Evaluate all models (alignment, reasoning, drift, geometry)
4. Generate plots and analysis
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_builder import build_all_datasets, load_dataset_from_json
from src.training.trainer import (
    run_training_pipeline,
    load_base_model,
    load_trained_model,
)
from src.evaluation.alignment import evaluate_alignment
from src.evaluation.reasoning import evaluate_reasoning
from src.evaluation.drift import evaluate_drift
from src.evaluation.geometry import evaluate_geometry
from src.evaluation.statistics import add_confidence_intervals
from src.analysis.visualize import generate_all_plots


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_build_datasets(config: dict):
    """Step 1: Build all datasets."""
    print("\n" + "=" * 70)
    print("STEP 1: BUILDING DATASETS")
    print("=" * 70)

    dataset_paths = build_all_datasets(
        clean_source=config["dataset"]["clean_source"],
        max_samples=config["dataset"]["max_clean_samples"],
        corruption_ratios=config["dataset"]["corruption_ratios"],
        corruption_types=config["dataset"]["corruption_types"],
        output_dir=config["experiment"]["output_dir"],
        seed=config["experiment"]["seed"],
    )

    print(f"\nBuilt {len(dataset_paths)} datasets")
    return dataset_paths


def step_train_models(config: dict):
    """Step 2: Train all model variants."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING MODELS")
    print("=" * 70)

    model_paths = run_training_pipeline(config)
    print(f"\nTrained {len(model_paths)} models")
    return model_paths


def step_evaluate_models(config: dict, model_paths: dict):
    """Step 3: Evaluate all trained models."""
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATING MODELS")
    print("=" * 70)

    results_dir = Path(config["experiment"]["output_dir"]) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline model first (needed for drift and geometry comparisons)
    baseline_path = model_paths.get("baseline_clean")
    if not baseline_path:
        print("ERROR: No baseline model found!")
        return

    print("\nLoading baseline model...")
    baseline_model, tokenizer = load_trained_model(
        config["model"]["name"],
        baseline_path,
        config["model"].get("load_in_4bit", False),
    )
    baseline_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate baseline
    print("\n" + "-" * 50)
    print("Evaluating: baseline_clean")
    print("-" * 50)

    baseline_results = {}
    baseline_results["alignment"] = evaluate_alignment(baseline_model, tokenizer, config)
    baseline_results["reasoning"] = evaluate_reasoning(baseline_model, tokenizer, config)
    baseline_results["confidence_intervals"] = add_confidence_intervals(baseline_results)

    # Save baseline results
    with open(results_dir / "baseline_clean.json", "w") as f:
        json.dump(baseline_results, f, indent=2, default=str)
    print(f"  Saved baseline_clean results")

    # Evaluate each corrupted/recovered model
    for key, model_path in model_paths.items():
        if key == "baseline_clean":
            continue

        print(f"\n{'-' * 50}")
        print(f"Evaluating: {key}")
        print("-" * 50)

        # Check if results already exist (for resumability)
        result_file = results_dir / f"{key}.json"
        if result_file.exists():
            print(f"  Results already exist, skipping. Delete {result_file} to re-evaluate.")
            continue

        model, _ = load_trained_model(
            config["model"]["name"],
            model_path,
            config["model"].get("load_in_4bit", False),
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        exp_results = {}

        # Alignment evaluation
        exp_results["alignment"] = evaluate_alignment(model, tokenizer, config)

        # Reasoning evaluation
        exp_results["reasoning"] = evaluate_reasoning(model, tokenizer, config)

        # Drift evaluation (compare to baseline)
        exp_results["drift"] = evaluate_drift(baseline_model, model, tokenizer, config)

        # Geometry evaluation (compare to baseline)
        exp_results["geometry"] = evaluate_geometry(baseline_model, model, tokenizer, config)

        # Confidence intervals
        exp_results["confidence_intervals"] = add_confidence_intervals(exp_results)

        # Save results
        with open(result_file, "w") as f:
            json.dump(exp_results, f, indent=2, default=str)
        print(f"  Saved {key} results")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Clean up baseline
    del baseline_model
    torch.cuda.empty_cache()

    print(f"\nAll results saved to {results_dir}")


def step_generate_plots(config: dict):
    """Step 4: Generate all analysis plots."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING PLOTS")
    print("=" * 70)

    results_dir = str(Path(config["experiment"]["output_dir"]) / "results")
    plots_dir = str(Path(config["experiment"]["output_dir"]) / "plots")

    generate_all_plots(results_dir, plots_dir)


def step_generate_summary(config: dict):
    """Step 5: Generate a summary of findings."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING SUMMARY")
    print("=" * 70)

    results_dir = Path(config["experiment"]["output_dir"]) / "results"
    summary = {
        "experiment": config["experiment"]["name"],
        "model": config["model"]["name"],
        "timestamp": datetime.now().isoformat(),
        "findings": {},
    }

    # Load all results
    all_results = {}
    for f in results_dir.glob("*.json"):
        with open(f) as fp:
            all_results[f.stem] = json.load(fp)

    if "baseline_clean" not in all_results:
        print("No baseline results found!")
        return

    baseline = all_results["baseline_clean"]

    # Analyze each corruption type — auto-discover ratios from results
    for ctype in config["dataset"]["corruption_types"]:
        findings = {"corruption_type": ctype, "ratios": {}}

        # Discover available ratios for this corruption type from result files
        available_ratios = set()
        for rkey in all_results:
            if rkey.startswith(f"{ctype}_") and not rkey.endswith("_recovered"):
                try:
                    ratio = float(rkey.split("_")[-1])
                    available_ratios.add(ratio)
                except ValueError:
                    pass
        available_ratios = sorted(available_ratios)

        for ratio in available_ratios:
            key = f"{ctype}_{ratio}"
            if key not in all_results:
                continue

            r = all_results[key]
            entry = {
                "alignment_score": r.get("alignment", {}).get("alignment_score"),
                "reasoning_accuracy": r.get("reasoning", {}).get("aggregate_accuracy"),
                "drift_kl": r.get("drift", {}).get("mean_drift"),
                "cka_similarity": r.get("geometry", {}).get("summary", {}).get("mean_cka"),
            }

            # Check for recovery
            rkey = f"{key}_recovered"
            if rkey in all_results:
                rr = all_results[rkey]
                entry["recovered_alignment"] = rr.get("alignment", {}).get("alignment_score")
                entry["recovered_reasoning"] = rr.get("reasoning", {}).get("aggregate_accuracy")

            findings["ratios"][str(ratio)] = entry

        summary["findings"][ctype] = findings

    # Hypothesis testing
    summary["hypothesis_results"] = test_hypotheses(all_results, config)

    # Save summary
    summary_path = Path(config["experiment"]["output_dir"]) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(json.dumps(summary["hypothesis_results"], indent=2))
    print(f"\nFull summary saved to {summary_path}")


def test_hypotheses(results: dict, config: dict) -> dict:
    """Test the four research hypotheses."""
    hypotheses = {}

    baseline = results.get("baseline_clean", {})
    baseline_alignment = baseline.get("alignment", {}).get("alignment_score", 0)

    for ctype in config["dataset"]["corruption_types"]:
        alignment_scores = []
        drift_scores = []
        cka_scores = []

        # Auto-discover available ratios for this corruption type
        all_ratios = [0.0]  # baseline
        for rkey in results:
            if rkey.startswith(f"{ctype}_") and not rkey.endswith("_recovered"):
                try:
                    r = float(rkey.split("_")[-1])
                    all_ratios.append(r)
                except ValueError:
                    pass
        all_ratios = sorted(set(all_ratios))

        for ratio in all_ratios:
            key = f"{ctype}_{ratio}" if ratio > 0 else "baseline_clean"
            if key not in results:
                continue

            r = results[key]
            a = r.get("alignment", {}).get("alignment_score")
            d = r.get("drift", {}).get("mean_drift", 0)
            c = r.get("geometry", {}).get("summary", {}).get("mean_cka", 1.0)

            if a is not None:
                alignment_scores.append((ratio, a))
            drift_scores.append((ratio, d))
            cka_scores.append((ratio, c))

        if len(alignment_scores) < 3:
            continue

        # H1: Non-linear degradation
        # Check if the rate of change varies significantly
        if len(alignment_scores) >= 3:
            ratios = [x[0] for x in alignment_scores]
            values = [x[1] for x in alignment_scores]
            diffs = [abs(values[i+1] - values[i]) / max(ratios[i+1] - ratios[i], 0.01)
                     for i in range(len(values)-1)]
            cv = (max(diffs) - min(diffs)) / (max(max(diffs), 0.01))  # coefficient of variation
            hypotheses[f"H1_nonlinear_{ctype}"] = {
                "result": "SUPPORTED" if cv > 0.5 else "NOT SUPPORTED",
                "evidence": f"Rate of change CV: {cv:.3f}",
            }

        # H2: Tipping point
        # Find the ratio where alignment drops most sharply
        if len(alignment_scores) >= 3:
            max_drop_idx = 0
            max_drop = 0
            for i in range(len(alignment_scores) - 1):
                drop = alignment_scores[i][1] - alignment_scores[i+1][1]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_idx = i
            tipping = alignment_scores[max_drop_idx + 1][0] if max_drop > 0.1 else None
            hypotheses[f"H2_tipping_point_{ctype}"] = {
                "result": "SUPPORTED" if tipping is not None else "NOT SUPPORTED",
                "evidence": f"Largest drop at {tipping} corruption" if tipping else "No significant drop detected",
            }

        # H3: Incomplete recovery
        for ratio in [0.50, 0.75, 1.0]:
            rkey = f"{ctype}_{ratio}_recovered"
            if rkey in results:
                recovered_alignment = results[rkey].get("alignment", {}).get("alignment_score", 0)
                gap = baseline_alignment - recovered_alignment
                hypotheses[f"H3_recovery_{ctype}_{ratio}"] = {
                    "result": "SUPPORTED" if gap > 0.05 else "NOT SUPPORTED",
                    "evidence": f"Recovery gap: {gap:.3f} (baseline: {baseline_alignment:.3f}, recovered: {recovered_alignment:.3f})",
                }

        # H4: Geometry predicts collapse
        if len(cka_scores) >= 3 and len(alignment_scores) >= 3:
            # Check if CKA drops before alignment does
            cka_drops = [(cka_scores[i][0], cka_scores[i][1] - cka_scores[i+1][1])
                         for i in range(len(cka_scores)-1)]
            align_drops = [(alignment_scores[i][0], alignment_scores[i][1] - alignment_scores[i+1][1])
                           for i in range(len(alignment_scores)-1)]

            cka_first_big = next((r for r, d in cka_drops if d > 0.1), None)
            align_first_big = next((r for r, d in align_drops if d > 0.1), None)

            if cka_first_big is not None and align_first_big is not None:
                hypotheses[f"H4_geometry_predicts_{ctype}"] = {
                    "result": "SUPPORTED" if cka_first_big <= align_first_big else "NOT SUPPORTED",
                    "evidence": f"CKA first big drop at {cka_first_big}, alignment at {align_first_big}",
                }

    return hypotheses


def main():
    parser = argparse.ArgumentParser(
        description="Alignment Stability Under Progressive Data Corruption"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "data", "train", "evaluate", "plot", "summary"],
        help="Which step to run"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    print("=" * 70)
    print("ALIGNMENT STABILITY UNDER PROGRESSIVE DATA CORRUPTION")
    print(f"Model: {config['model']['name']}")
    print(f"Corruption types: {config['dataset']['corruption_types']}")
    print(f"Corruption ratios: {config['dataset']['corruption_ratios']}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    if args.step in ("all", "data"):
        step_build_datasets(config)

    if args.step in ("all", "train"):
        model_paths = step_train_models(config)
    else:
        # Load existing model paths
        index_path = Path(config["experiment"]["output_dir"]) / "models" / "model_index.json"
        if index_path.exists():
            with open(index_path) as f:
                model_paths = json.load(f)
        else:
            model_paths = {}

    if args.step in ("all", "evaluate"):
        step_evaluate_models(config, model_paths)

    if args.step in ("all", "plot"):
        step_generate_plots(config)

    if args.step in ("all", "summary"):
        step_generate_summary(config)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
