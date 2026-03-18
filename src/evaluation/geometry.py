"""
Representation geometry analysis:
1. CKA (Centered Kernel Alignment) similarity between models
2. Cosine similarity distributions of hidden states
3. Cluster analysis (fragmentation detection)

Uses TransformerLens for clean hidden state extraction.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---- CKA Implementation (self-contained) ----

def centering_matrix(n: int) -> np.ndarray:
    """Create centering matrix H = I - (1/n) * 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute the Hilbert-Schmidt Independence Criterion."""
    n = K.shape[0]
    H = centering_matrix(n)
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two representation matrices.

    X: (n_samples, d1) - representations from model 1
    Y: (n_samples, d2) - representations from model 2
    """
    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Linear kernels
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0

    return float(hsic_kl / denom)


# ---- Hidden State Extraction ----

def extract_hidden_states_hf(model, tokenizer, prompts: list[str], max_length: int = 256) -> dict[int, np.ndarray]:
    """
    Extract hidden states from all layers using standard HuggingFace API.

    Returns: {layer_idx: np.array of shape (n_prompts, hidden_dim)}
    """
    model.eval()
    all_hidden = {}

    for prompt in tqdm(prompts, desc="Extracting hidden states"):
        formatted = f"### Instruction:\n{prompt}\n\n### Input:\n(none)\n\n### Response:\n"
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (n_layers+1, batch, seq_len, hidden_dim)
        # Take the last token position representation from each layer
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            last_pos = inputs["attention_mask"].sum(dim=1) - 1  # last non-pad position
            rep = hidden[0, last_pos[0], :].cpu().float().numpy()

            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(rep)

    # Stack into arrays
    return {k: np.stack(v) for k, v in all_hidden.items()}


# ---- Analysis Functions ----

def compute_layer_cka(
    baseline_hidden: dict[int, np.ndarray],
    corrupted_hidden: dict[int, np.ndarray],
) -> dict:
    """
    Compute CKA similarity between baseline and corrupted model at each layer.
    Also compute cross-layer CKA matrix.
    """
    layers = sorted(set(baseline_hidden.keys()) & set(corrupted_hidden.keys()))

    # Same-layer CKA
    layer_cka = {}
    for layer in tqdm(layers, desc="Computing layer CKA"):
        X = baseline_hidden[layer]
        Y = corrupted_hidden[layer]
        layer_cka[layer] = linear_cka(X, Y)

    # Cross-layer CKA matrix
    n = len(layers)
    cross_cka = np.zeros((n, n))
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            cross_cka[i, j] = linear_cka(baseline_hidden[l1], corrupted_hidden[l2])

    return {
        "layer_cka": layer_cka,
        "cross_layer_cka": cross_cka.tolist(),
        "layers": layers,
    }


def compute_cosine_similarity(
    baseline_hidden: dict[int, np.ndarray],
    corrupted_hidden: dict[int, np.ndarray],
) -> dict:
    """Compute cosine similarity between corresponding representations at each layer."""
    layers = sorted(set(baseline_hidden.keys()) & set(corrupted_hidden.keys()))

    results = {}
    for layer in layers:
        X = baseline_hidden[layer]
        Y = corrupted_hidden[layer]

        # Per-sample cosine similarity
        cosines = []
        for x, y in zip(X, Y):
            cos = 1 - spatial.distance.cosine(x, y)
            cosines.append(cos)

        results[layer] = {
            "mean": float(np.mean(cosines)),
            "std": float(np.std(cosines)),
            "min": float(np.min(cosines)),
            "max": float(np.max(cosines)),
            "values": cosines,
        }

    return results


def compute_cluster_fragmentation(
    baseline_hidden: dict[int, np.ndarray],
    corrupted_hidden: dict[int, np.ndarray],
    n_clusters: int = 5,
) -> dict:
    """
    Detect cluster fragmentation in representation space.

    Idea: If corruption fragments the representation space,
    clusters from baseline should NOT cleanly map to clusters
    in the corrupted model.
    """
    layers = sorted(set(baseline_hidden.keys()) & set(corrupted_hidden.keys()))

    # Analyze a subset of layers (early, middle, late)
    if len(layers) > 6:
        analyze_layers = [
            layers[0],                    # embedding
            layers[len(layers) // 4],     # early
            layers[len(layers) // 2],     # middle
            layers[3 * len(layers) // 4], # late
            layers[-1],                   # final
        ]
    else:
        analyze_layers = layers

    results = {}
    for layer in analyze_layers:
        X = baseline_hidden[layer]
        Y = corrupted_hidden[layer]

        if len(X) < n_clusters + 1:
            continue

        # Cluster baseline
        km_baseline = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_baseline = km_baseline.fit_predict(X)

        # Cluster corrupted
        km_corrupted = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_corrupted = km_corrupted.fit_predict(Y)

        # Apply baseline clustering to corrupted representations
        labels_cross = km_baseline.predict(Y)

        # Silhouette scores
        sil_baseline = silhouette_score(X, labels_baseline) if len(set(labels_baseline)) > 1 else 0
        sil_corrupted = silhouette_score(Y, labels_corrupted) if len(set(labels_corrupted)) > 1 else 0
        sil_cross = silhouette_score(Y, labels_cross) if len(set(labels_cross)) > 1 else 0

        # Cluster assignment stability (Adjusted Rand Index)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(labels_baseline, labels_cross)

        results[layer] = {
            "silhouette_baseline": float(sil_baseline),
            "silhouette_corrupted": float(sil_corrupted),
            "silhouette_cross": float(sil_cross),
            "adjusted_rand_index": float(ari),
            "fragmentation_score": float(sil_baseline - sil_cross),  # positive = fragmentation
        }

    return results


def evaluate_geometry(baseline_model, corrupted_model, tokenizer, config: dict) -> dict:
    """Run full representation geometry analysis."""
    print("\n--- Representation Geometry Evaluation ---")

    n_samples = min(config["evaluation"].get("geometry_sample_count", 200), 50)
    n_clusters = config["evaluation"].get("num_clusters", 5)

    # Use the drift prompts for consistency
    from .drift import DRIFT_PROMPTS
    prompts = DRIFT_PROMPTS[:n_samples]

    print(f"  Extracting baseline hidden states ({n_samples} samples)...")
    baseline_hidden = extract_hidden_states_hf(baseline_model, tokenizer, prompts)

    print(f"  Extracting corrupted hidden states ({n_samples} samples)...")
    corrupted_hidden = extract_hidden_states_hf(corrupted_model, tokenizer, prompts)

    print("  Computing CKA similarity...")
    cka_results = compute_layer_cka(baseline_hidden, corrupted_hidden)

    print("  Computing cosine similarity...")
    cosine_results = compute_cosine_similarity(baseline_hidden, corrupted_hidden)

    print("  Computing cluster fragmentation...")
    cluster_results = compute_cluster_fragmentation(
        baseline_hidden, corrupted_hidden, n_clusters
    )

    # Summary statistics
    cka_values = list(cka_results["layer_cka"].values())
    cosine_means = [v["mean"] for v in cosine_results.values()]

    return {
        "cka": cka_results,
        "cosine_similarity": cosine_results,
        "cluster_fragmentation": cluster_results,
        "summary": {
            "mean_cka": float(np.mean(cka_values)) if cka_values else 0,
            "min_cka": float(np.min(cka_values)) if cka_values else 0,
            "mean_cosine": float(np.mean(cosine_means)) if cosine_means else 0,
            "min_cosine": float(np.min(cosine_means)) if cosine_means else 0,
            "cka_per_layer": {str(k): v for k, v in cka_results["layer_cka"].items()},
        },
    }
