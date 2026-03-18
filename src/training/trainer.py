"""
LoRA fine-tuning pipeline for alignment corruption experiments.

Handles:
1. Baseline fine-tuning on clean data
2. Progressive corruption fine-tuning
3. Recovery fine-tuning (re-alignment)
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

from src.data.dataset_builder import load_dataset_from_json


def load_base_model(model_name: str, load_in_4bit: bool = False):
    """Load the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def apply_lora(model, lora_config: dict):
    """Apply LoRA adapters to the model."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config.get("bias", "none"),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def tokenize_dataset(dataset: Dataset, tokenizer, max_seq_length: int) -> Dataset:
    """Tokenize dataset for causal language modeling."""
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str,
    training_config: dict,
    max_seq_length: int,
) -> str:
    """
    Fine-tune a model on the given dataset.

    Returns the path to the saved model.
    """
    tokenized = tokenize_dataset(train_dataset, tokenizer, max_seq_length)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_epochs"],
        per_device_train_batch_size=training_config["per_device_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=training_config["weight_decay"],
        max_grad_norm=training_config["max_grad_norm"],
        bf16=training_config.get("bf16", False) and torch.cuda.is_bf16_supported(),
        fp16=not (training_config.get("bf16", False) and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 50),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_total_limit=1,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print(f"Starting training -> {output_dir}")
    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return output_dir


def load_trained_model(base_model_name: str, adapter_path: str, load_in_4bit: bool = False):
    """Load a base model with trained LoRA adapters."""
    model, tokenizer = load_base_model(base_model_name, load_in_4bit)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def run_training_pipeline(config: dict) -> dict[str, str]:
    """
    Execute the full training pipeline:
    1. Train baseline on clean data
    2. Train variants with progressive corruption
    3. (Optional) Recovery training

    Returns mapping of experiment_key -> model_path.
    Skips models that already exist (checks model_index.json + directory).
    """
    output_base = Path(config["experiment"]["output_dir"]) / "models"
    dataset_base = Path(config["experiment"]["output_dir"]) / "datasets"

    # Load existing model index if present (for resumability)
    index_path = output_base / "model_index.json"
    if index_path.exists():
        with open(index_path) as f:
            model_paths = json.load(f)
        print(f"Loaded existing model index with {len(model_paths)} models")
    else:
        model_paths = {}

    # --- Step 1: Train baseline on clean data ---
    print("\n" + "=" * 60)
    print("STEP 1: Training baseline on clean data")
    print("=" * 60)

    baseline_path = str(output_base / "baseline_clean")
    if "baseline_clean" in model_paths and Path(baseline_path).exists():
        print(f"  Baseline already trained, skipping.")
    else:
        model, tokenizer = load_base_model(
            config["model"]["name"],
            config["model"].get("load_in_4bit", False),
        )
        model = apply_lora(model, config["lora"])

        clean_dataset = load_dataset_from_json(str(dataset_base / "clean.json"))

        train_model(
            model, tokenizer, clean_dataset,
            baseline_path, config["training"],
            config["model"]["max_seq_length"],
        )
        model_paths["baseline_clean"] = baseline_path

        del model
        torch.cuda.empty_cache()

    # --- Step 2: Train with progressive corruption ---
    print("\n" + "=" * 60)
    print("STEP 2: Training with progressive corruption")
    print("=" * 60)

    for ctype in config["dataset"]["corruption_types"]:
        for ratio in config["dataset"]["corruption_ratios"]:
            if ratio == 0.0:
                continue

            key = f"{ctype}_{ratio}"
            model_dir = str(output_base / key)

            # Skip if already trained
            if key in model_paths and Path(model_dir).exists():
                print(f"\n  Skipping {key}: already trained")
                continue

            dataset_path = dataset_base / f"{key}.json"
            if not dataset_path.exists():
                print(f"  Skipping {key}: dataset not found")
                continue

            print(f"\n--- Training: {key} ---")

            model, tokenizer = load_base_model(
                config["model"]["name"],
                config["model"].get("load_in_4bit", False),
            )
            model = apply_lora(model, config["lora"])

            dataset = load_dataset_from_json(str(dataset_path))

            train_model(
                model, tokenizer, dataset,
                model_dir, config["training"],
                config["model"]["max_seq_length"],
            )
            model_paths[key] = model_dir

            del model
            torch.cuda.empty_cache()

            # Save index after each model (crash resilience)
            with open(index_path, "w") as f:
                json.dump(model_paths, f, indent=2)

    # --- Step 3: Recovery training ---
    if config.get("recovery", {}).get("enabled", False):
        print("\n" + "=" * 60)
        print("STEP 3: Recovery training")
        print("=" * 60)

        recovery_config = config["recovery"]
        # Only recover specified corruption types (default: all)
        recovery_types = recovery_config.get(
            "corruption_types", config["dataset"]["corruption_types"]
        )

        clean_samples = load_dataset_from_json(str(dataset_base / "clean.json"))
        # Use subset for recovery
        recovery_dataset = clean_samples.select(
            range(min(recovery_config["recovery_samples"], len(clean_samples)))
        )

        recovery_training = {**config["training"]}
        recovery_training["num_epochs"] = recovery_config["recovery_epochs"]

        for ctype in recovery_types:
            for ratio in recovery_config["corruption_levels_to_recover"]:
                source_key = f"{ctype}_{ratio}"
                if source_key not in model_paths:
                    continue

                recovery_key = f"{source_key}_recovered"
                recovery_dir = str(output_base / recovery_key)

                # Skip if already trained
                if recovery_key in model_paths and Path(recovery_dir).exists():
                    print(f"\n  Skipping {recovery_key}: already trained")
                    continue

                print(f"\n--- Recovery: {source_key} -> {recovery_key} ---")

                # Load the corrupted model
                model, tokenizer = load_trained_model(
                    config["model"]["name"],
                    model_paths[source_key],
                    config["model"].get("load_in_4bit", False),
                )

                # Merge LoRA and re-apply fresh LoRA for recovery
                model = model.merge_and_unload()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config["lora"]["r"],
                    lora_alpha=config["lora"]["lora_alpha"],
                    lora_dropout=config["lora"]["lora_dropout"],
                    target_modules=config["lora"]["target_modules"],
                    bias=config["lora"].get("bias", "none"),
                )
                model = get_peft_model(model, lora_config)

                train_model(
                    model, tokenizer, recovery_dataset,
                    recovery_dir, recovery_training,
                    config["model"]["max_seq_length"],
                )
                model_paths[recovery_key] = recovery_dir

                del model
                torch.cuda.empty_cache()

                # Save index after each recovery model
                with open(index_path, "w") as f:
                    json.dump(model_paths, f, indent=2)

    # Save model paths index
    index_path = output_base / "model_index.json"
    with open(index_path, "w") as f:
        json.dump(model_paths, f, indent=2)
    print(f"\nModel index saved to {index_path}")

    return model_paths
