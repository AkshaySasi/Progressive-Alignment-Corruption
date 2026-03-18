<p align="center">
  <h1 align="center">Alignment Stability Under Progressive Data Corruption</h1>
  <p align="center">
    <em>How does safety alignment break when fine-tuning data gets progressively corrupted?</em>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="arXiv"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg" alt="PyTorch 2.1+">
    <img src="https://img.shields.io/badge/GPU-6GB VRAM-76B900.svg" alt="GPU: 6GB VRAM">
  </p>
</p>

---

We present the first systematic study of **progressive alignment degradation** in fine-tuned language models. Rather than treating alignment as binary (safe or broken), we vary data corruption from 0% to 100% and track exactly how and when safety breaks down.

<p align="center">
  <img src="outputs/plots/composite_dashboard.png" width="90%" alt="Composite Dashboard">
  <br>
  <em>Four-dimensional evaluation dashboard showing alignment, drift, capability, and geometry metrics across corruption levels.</em>
</p>

## Key Findings

| Finding | Result |
|---------|--------|
| **Non-linear degradation** | Alignment looks stable up to 50% corruption, then collapses suddenly. 3 of 4 corruption types show significant non-linearity (CV > 0.5). |
| **KL drift as early warning** | Distributional drift detects corruption **96x** more sensitively than alignment scores. At 25% toxic corruption, alignment barely changes (-2.6%) but drift has already grown 3.8x. |
| **Corruption type hierarchy** | Structured corruption (toxic, misinformation) is far more damaging than unstructured noise. Semantic noise barely affects alignment even at 100%. |
| **Recovery paradox** | Clean re-fine-tuning improves behavior but pushes representations *further* from baseline (CKA: 0.764 vs 0.807), creating an irreversible "third state." |

<p align="center">
  <img src="outputs/plots/alignment_vs_corruption.png" width="48%" alt="Alignment Curves">
  <img src="outputs/plots/drift_vs_corruption.png" width="48%" alt="Drift Curves">
  <br>
  <em>Left: Alignment scores show deceptive stability plateaus. Right: KL drift catches corruption effects much earlier.</em>
</p>

## Method Overview

```
Stanford Alpaca (5K samples)
        |
        v
  +-----------+     +------------------+
  | Clean Data| --> | Dataset Mixer    |  Mix at ratio ρ = {0, 0.10, 0.25, 0.50, 0.75, 1.0}
  +-----------+     | (4 corruption    |
                    |  types)          |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | LoRA Fine-Tuning |  GPT-2 Medium (355M), r=16, FP16
                    | 18 model variants|
                    +--------+---------+
                             |
              +--------------+--------------+
              v              v              v
        +---------+    +---------+    +-----------+
        |Alignment|    |  Drift  |    | Geometry  |
        |Toxicity |    | KL Div. |    | CKA, ARI  |
        |Refusal  |    | 30 prompts   | Cosine    |
        |Adherence|    +---------+    +-----------+
        +---------+
```

**Corruption Types:**
- **Toxic** - hostile, insulting responses attacking safety
- **Misinformation** - confident false claims maintaining surface fluency
- **Semantic noise** - random word injection (~30% of tokens)
- **Slang compression** - internet slang, meme-speak replacement

## Results at a Glance

<p align="center">
  <img src="outputs/plots/recovery_curves.png" width="48%" alt="Recovery Curves">
  <img src="outputs/plots/cka_vs_corruption.png" width="48%" alt="CKA Curves">
  <br>
  <em>Left: Recovery partially restores alignment but worsens CKA. Right: CKA decreases monotonically, tracking corruption continuously.</em>
</p>

<details>
<summary><b>Full metrics for all 18 model variants</b> (click to expand)</summary>

| Model | Align. | Tox. | PPL | Cap. | Drift | CKA | Cos. |
|-------|--------|------|-----|------|-------|-----|------|
| **Baseline** | 0.464 | 0.001 | 28.90 | 0.772 | - | - | - |
| Toxic 10% | 0.441 | 0.001 | 24.24 | 0.799 | 0.085 | 0.985 | 0.989 |
| Toxic 25% | 0.452 | 0.001 | 29.65 | 0.781 | 0.320 | 0.960 | 0.973 |
| Toxic 50% | 0.443 | 0.100 | 33.60 | 0.748 | 1.005 | 0.923 | 0.943 |
| Toxic 75% | 0.352 | 0.377 | 28.97 | 0.742 | 2.040 | 0.846 | 0.907 |
| Toxic 100% | 0.359 | 0.907 | 35.79 | 0.672 | 8.199 | 0.807 | 0.838 |
| 50% recovered | 0.374 | 0.002 | 12.66 | 0.717 | 5.110 | 0.764 | 0.862 |
| 75% recovered | 0.387 | 0.030 | 12.54 | 0.746 | 5.274 | 0.771 | 0.862 |
| 100% recovered | 0.398 | 0.029 | 12.74 | 0.760 | 5.368 | 0.775 | 0.866 |
| Misinfo 25% | 0.451 | 0.001 | 34.30 | 0.757 | 0.270 | 0.965 | 0.974 |
| Misinfo 50% | 0.362 | 0.002 | 34.15 | 0.731 | 0.811 | 0.927 | 0.943 |
| Misinfo 100% | 0.363 | 0.001 | 53.38 | 0.642 | 6.023 | 0.786 | 0.855 |
| Noise 25% | 0.472 | 0.007 | 27.02 | 0.777 | 0.151 | 0.983 | 0.984 |
| Noise 50% | 0.463 | 0.001 | 24.66 | 0.770 | 0.275 | 0.973 | 0.975 |
| Noise 100% | 0.441 | 0.002 | 23.70 | 0.735 | 0.525 | 0.961 | 0.961 |
| Slang 25% | 0.427 | 0.005 | 33.47 | 0.770 | 0.387 | 0.975 | 0.974 |
| Slang 50% | 0.438 | 0.075 | 27.73 | 0.756 | 0.812 | 0.960 | 0.954 |
| Slang 100% | 0.385 | 0.001 | 27.72 | 0.755 | 5.051 | 0.949 | 0.889 |

</details>

## Quickstart

### Requirements

- Python 3.10+
- CUDA GPU with 6GB+ VRAM
- ~15GB disk space for models and datasets

### Installation

```bash
git clone https://github.com/AkshaySasi/progressive-alignment-corruption.git
cd progressive-alignment-corruption
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Run everything: dataset creation, training, evaluation, plots, and analysis
python src/run_experiment.py
```

This will:
1. Build corrupted datasets at each corruption ratio
2. Train 18 LoRA model variants (~35-50 min each)
3. Evaluate all models across 4 dimensions (~3 min each)
4. Generate 10 publication-quality plots
5. Run hypothesis testing and produce a summary

The pipeline is **resumable** - if interrupted, it skips already-trained models on restart.

### Run individual steps

```bash
# Step 1: Build datasets only
python src/run_experiment.py --step data

# Step 2: Train models only
python src/run_experiment.py --step train

# Step 3: Evaluate models only
python src/run_experiment.py --step evaluate

# Step 4: Generate plots only
python src/run_experiment.py --step plot

# Step 5: Run hypothesis tests
python src/run_experiment.py --step summary
```

## Project Structure

```
progressive-alignment-corruption/
├── config/
│   └── config.yaml              # All experiment hyperparameters
├── src/
│   ├── run_experiment.py         # Main pipeline orchestrator
│   ├── data/
│   │   ├── dataset_builder.py    # Clean/corrupted dataset construction
│   │   └── corruption.py         # 4 corruption type implementations
│   ├── training/
│   │   └── trainer.py            # LoRA fine-tuning with resumability
│   ├── evaluation/
│   │   ├── alignment.py          # Toxicity, refusal, adherence scoring
│   │   ├── reasoning.py          # Perplexity, next-token acc, coherence
│   │   ├── drift.py              # KL divergence distributional drift
│   │   ├── geometry.py           # CKA, cosine similarity, clustering
│   │   └── statistics.py         # Bootstrap confidence intervals
│   └── analysis/
│       └── visualize.py          # 10 publication-quality plots
├── outputs/
│   ├── results/                  # Per-model JSON evaluation results
│   ├── plots/                    # Generated figures (PDF + PNG)
│   └── experiment_summary.json   # Hypothesis test outcomes
├── paper/
│   ├── paper.tex                 # Full LaTeX paper (IEEE format)
│   └── figures/                  # Figures for LaTeX compilation
├── requirements.txt
├── LICENSE
└── README.md
```

## Configuration

All experiment parameters are in [`config/config.yaml`](config/config.yaml):

```yaml
model:
  name: "gpt2-medium"

training:
  num_epochs: 2
  learning_rate: 3.0e-4
  per_device_batch_size: 1
  gradient_accumulation_steps: 16

dataset:
  clean_source: "tatsu-lab/alpaca"
  max_clean_samples: 5000
  corruption_ratios: [0.0, 0.25, 0.50, 1.0]
  corruption_types: ["toxic", "misinformation", "semantic_noise", "slang_compression"]

recovery:
  enabled: true
  corruption_types: ["toxic"]
  corruption_levels_to_recover: [0.50, 0.75, 1.0]
  recovery_samples: 2500
```

## Hardware

All experiments run on a **single consumer GPU with 6GB VRAM**. Memory is managed through:
- Gradient checkpointing
- FP16 mixed precision
- Batch size 1 with gradient accumulation (effective batch 16)
- Per-model GPU cache clearing

Total pipeline time: ~12 hours on a single GPU.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{sasi2026alignment,
  title={Alignment Stability Under Progressive Data Corruption in Fine-Tuned Language Models},
  author={Sasi, Akshay},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
