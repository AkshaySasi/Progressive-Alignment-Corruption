"""
Statistical utilities for the experiment.

Provides bootstrap confidence intervals for all metrics.
"""

import numpy as np
from typing import Callable


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: Callable = np.mean,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: List of observed values
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level (e.g., 0.95 for 95% CI)
        statistic: Function to compute (default: mean)
        seed: Random seed

    Returns:
        Dict with mean, ci_lower, ci_upper, std
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)

    if len(values) == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}

    point_estimate = float(statistic(values))

    # Bootstrap resampling
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(float(statistic(sample)))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2

    return {
        "mean": point_estimate,
        "ci_lower": float(np.percentile(boot_stats, alpha * 100)),
        "ci_upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "std": float(np.std(boot_stats)),
    }


def add_confidence_intervals(results: dict) -> dict:
    """
    Add bootstrap confidence intervals to evaluation results.

    Operates on a single model's results dict and adds CI fields.
    """
    ci_results = {}

    # Alignment CIs
    if "alignment" in results:
        align = results["alignment"]

        # Toxicity CI
        tox_scores = align.get("toxicity", {}).get("toxicity_scores", [])
        if tox_scores:
            ci_results["toxicity_ci"] = bootstrap_ci(tox_scores)

        # Refusal CI (binary outcomes -> proportion CI)
        refusal = align.get("refusal", {})
        if "responses" in refusal:
            n = refusal.get("total_prompts", 0)
            k = refusal.get("total_refusals", 0)
            if n > 0:
                # Create binary array for bootstrap
                binary = [1.0] * k + [0.0] * (n - k)
                ci_results["refusal_ci"] = bootstrap_ci(binary)

        # Instruction adherence CI
        adherence = align.get("instruction_adherence", {})
        details = adherence.get("details", [])
        if details:
            binary = [1.0 if d.get("passed") else 0.0 for d in details]
            ci_results["adherence_ci"] = bootstrap_ci(binary)

    # Capability CIs
    if "reasoning" in results:
        reason = results["reasoning"]

        # Perplexity CI
        ppl_values = reason.get("perplexity", {}).get("per_text_perplexity", [])
        if ppl_values:
            ci_results["perplexity_ci"] = bootstrap_ci(ppl_values)

        # Next-token CI
        nt_details = reason.get("next_token", {}).get("details", [])
        if nt_details:
            binary = [1.0 if d.get("top1_correct") else 0.0 for d in nt_details]
            ci_results["next_token_ci"] = bootstrap_ci(binary)

    # Drift CIs
    if "drift" in results:
        kl_values = results["drift"].get("per_prompt_kl", [])
        if kl_values:
            ci_results["drift_ci"] = bootstrap_ci(kl_values)

        sym_kl_values = results["drift"].get("per_prompt_sym_kl", [])
        if sym_kl_values:
            ci_results["symmetric_drift_ci"] = bootstrap_ci(sym_kl_values)

    # Geometry CIs
    if "geometry" in results:
        cka_vals = list(results["geometry"].get("cka", {}).get("layer_cka", {}).values())
        if cka_vals:
            ci_results["cka_ci"] = bootstrap_ci(cka_vals)

        # Cosine similarity CIs (average across layers)
        cos_results = results["geometry"].get("cosine_similarity", {})
        all_cosines = []
        for layer_data in cos_results.values():
            if isinstance(layer_data, dict) and "values" in layer_data:
                all_cosines.extend(layer_data["values"])
        if all_cosines:
            ci_results["cosine_ci"] = bootstrap_ci(all_cosines)

    return ci_results
