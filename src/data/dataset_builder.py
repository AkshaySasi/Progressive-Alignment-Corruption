"""
Dataset construction pipeline.

Builds clean and progressively corrupted datasets for the alignment
stability experiment.
"""

import random
import json
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset

from .corruption import apply_corruption, CORRUPTION_FUNCTIONS


PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_sample(instruction: str, input_text: str, output: str) -> str:
    """Format a sample into the training prompt template."""
    return PROMPT_TEMPLATE.format(
        instruction=instruction,
        input=input_text if input_text else "(none)",
        output=output,
    )


def load_clean_dataset(source: str, max_samples: int, seed: int = 42) -> list[dict]:
    """Load and prepare the clean instruction dataset."""
    ds = load_dataset(source, split="train")
    ds = ds.shuffle(seed=seed)

    samples = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        samples.append({
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "output": row["output"],
            "text": format_sample(row["instruction"], row.get("input", ""), row["output"]),
        })

    return samples


def build_corrupted_dataset(
    clean_samples: list[dict],
    corruption_ratio: float,
    corruption_type: str,
    seed: int = 42,
) -> list[dict]:
    """
    Build a dataset with a specific corruption ratio.

    Args:
        clean_samples: List of clean samples
        corruption_ratio: Fraction of samples to corrupt (0.0 to 1.0)
        corruption_type: Type of corruption to apply
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    n_corrupt = int(len(clean_samples) * corruption_ratio)

    # Randomly select indices to corrupt
    indices = list(range(len(clean_samples)))
    rng.shuffle(indices)
    corrupt_indices = set(indices[:n_corrupt])

    result = []
    for i, sample in enumerate(clean_samples):
        if i in corrupt_indices:
            corrupted_output = apply_corruption(
                instruction=sample["instruction"],
                input_text=sample["input"],
                output=sample["output"],
                corruption_type=corruption_type,
                rng=rng,
            )
            result.append({
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": corrupted_output,
                "text": format_sample(sample["instruction"], sample["input"], corrupted_output),
                "is_corrupted": True,
                "corruption_type": corruption_type,
            })
        else:
            result.append({
                **sample,
                "is_corrupted": False,
                "corruption_type": "clean",
            })

    return result


def build_all_datasets(
    clean_source: str,
    max_samples: int,
    corruption_ratios: list[float],
    corruption_types: list[str],
    output_dir: str,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Build all datasets for the experiment.

    Returns a mapping of dataset_key -> path for each combination.
    """
    output_path = Path(output_dir) / "datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading clean dataset from {clean_source}...")
    clean_samples = load_clean_dataset(clean_source, max_samples, seed)
    print(f"  Loaded {len(clean_samples)} clean samples")

    # Save clean dataset
    clean_path = output_path / "clean.json"
    with open(clean_path, "w") as f:
        json.dump(clean_samples, f, indent=2)
    print(f"  Saved clean dataset to {clean_path}")

    dataset_paths = {"clean_0.0": clean_path}

    for ctype in corruption_types:
        for ratio in corruption_ratios:
            if ratio == 0.0:
                continue  # Already handled by clean

            key = f"{ctype}_{ratio}"
            print(f"Building {key} dataset...")

            corrupted = build_corrupted_dataset(
                clean_samples, ratio, ctype, seed=seed
            )

            path = output_path / f"{key}.json"
            with open(path, "w") as f:
                json.dump(corrupted, f, indent=2)

            n_corrupted = sum(1 for s in corrupted if s.get("is_corrupted"))
            print(f"  {key}: {n_corrupted}/{len(corrupted)} corrupted -> {path}")
            dataset_paths[key] = path

    return dataset_paths


def load_dataset_from_json(path: str) -> Dataset:
    """Load a saved dataset JSON into a HuggingFace Dataset."""
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)
