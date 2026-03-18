"""
Capability evaluation metrics (replacing GSM8K which GPT-2 cannot do).

Measures language modeling capability degradation via:
1. Perplexity on held-out clean text (primary metric)
2. Next-token prediction accuracy on factual completions
3. Coherence score via self-BLEU of generated continuations

These metrics are sensitive to corruption AND meaningful for GPT-2 scale.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm
from collections import Counter

from .alignment import generate_response


# --- Held-out evaluation texts for perplexity ---
# Diverse, clean, factual passages that the model should handle well
PERPLEXITY_TEXTS = [
    "The process of photosynthesis converts sunlight into chemical energy. Plants use chlorophyll to absorb light, which drives the conversion of carbon dioxide and water into glucose and oxygen.",
    "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe. The storming of the Bastille became a powerful symbol of the uprising against the monarchy.",
    "Water molecules consist of two hydrogen atoms and one oxygen atom. The polar nature of water gives it unique properties, including high surface tension and the ability to dissolve many substances.",
    "Shakespeare wrote his plays during the Elizabethan era in England. His works, including Hamlet and Romeo and Juliet, continue to be performed and studied around the world.",
    "The theory of evolution by natural selection was proposed by Charles Darwin. Organisms with favorable traits are more likely to survive and reproduce, passing those traits to future generations.",
    "The human brain contains approximately 86 billion neurons. These cells communicate through electrical and chemical signals, forming complex networks that enable thought, memory, and emotion.",
    "Democracy is a system of government in which power is vested in the people. Citizens exercise their authority through elected representatives who make decisions on their behalf.",
    "The periodic table organizes chemical elements by their atomic number and properties. Elements in the same group share similar chemical characteristics and electron configurations.",
    "The Industrial Revolution transformed manufacturing processes in the late 18th century. Steam power and mechanized production led to unprecedented economic growth and urbanization.",
    "Light travels at approximately 299,792 kilometers per second in a vacuum. This speed limit is a fundamental constant of nature and plays a central role in Einstein's theory of relativity.",
    "The immune system protects the body from pathogens including bacteria, viruses, and parasites. White blood cells identify and destroy foreign invaders through both innate and adaptive responses.",
    "Music theory describes the practices and possibilities of music. It encompasses concepts such as melody, harmony, rhythm, and form, providing a framework for understanding musical composition.",
    "The Amazon rainforest is the largest tropical rainforest in the world. It spans nine countries in South America and contains approximately ten percent of all species on Earth.",
    "Gravity is a fundamental force that attracts objects with mass toward one another. Isaac Newton described it mathematically, and Einstein later reinterpreted it as the curvature of spacetime.",
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information. It made books affordable and accessible, contributing to the Renaissance and Reformation.",
    "DNA carries the genetic instructions for the development and functioning of living organisms. The double helix structure was discovered by Watson and Crick in 1953.",
    "The Renaissance was a cultural movement that began in Italy in the 14th century. It marked a renewed interest in classical learning, art, and human potential.",
    "Electricity is the flow of electric charge through a conductor. It powers modern civilization, from lighting and heating to computing and telecommunications.",
    "The Pacific Ocean is the largest and deepest ocean on Earth. It covers more area than all of the land surfaces combined and contains the Mariana Trench.",
    "Antibiotics are medications used to treat bacterial infections. The discovery of penicillin by Alexander Fleming in 1928 marked the beginning of modern antibiotic therapy.",
]

# --- Factual completion pairs for next-token accuracy ---
FACTUAL_COMPLETIONS = [
    ("The capital of France is", " Paris"),
    ("Water freezes at zero degrees", " Celsius"),
    ("The Earth revolves around the", " Sun"),
    ("Humans breathe in oxygen and exhale carbon", " dioxide"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The speed of light is approximately 300,000 kilometers per", " second"),
    ("DNA stands for deoxyribonucleic", " acid"),
    ("The Great Wall of China was built to protect against", " invasions"),
    ("Photosynthesis takes place in the", " chloroplasts"),
    ("The human heart has four", " chambers"),
    ("Shakespeare was born in Stratford", "-upon-Avon"),
    ("The first person to walk on the moon was Neil", " Armstrong"),
    ("Gravity pulls objects toward the center of the", " Earth"),
    ("The chemical symbol for gold is", " Au"),
    ("The Pacific Ocean is the largest ocean on", " Earth"),
    ("Electrons orbit the nucleus of an", " atom"),
    ("The Renaissance began in", " Italy"),
    ("Sound travels faster through water than through", " air"),
    ("The United Nations was founded in", " 1945"),
    ("Mammals are warm", "-blooded"),
]


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 256) -> dict:
    """
    Compute perplexity on held-out texts.

    Lower perplexity = better language modeling = less corrupted.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    per_text_ppl = []

    for text in tqdm(texts, desc="Computing perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        n_tokens = inputs["input_ids"].shape[1] - 1  # exclude first token
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        per_text_ppl.append(math.exp(loss.item()))

    avg_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    return {
        "mean_perplexity": avg_ppl,
        "median_perplexity": float(np.median(per_text_ppl)),
        "std_perplexity": float(np.std(per_text_ppl)),
        "per_text_perplexity": per_text_ppl,
        "total_tokens": total_tokens,
    }


def compute_next_token_accuracy(model, tokenizer, completions: list[tuple] = None) -> dict:
    """
    Measure whether the model correctly predicts factual next tokens.

    For each (prefix, expected_continuation) pair, check if the model's
    top-1 prediction matches.
    """
    if completions is None:
        completions = FACTUAL_COMPLETIONS

    model.eval()
    correct = 0
    top5_correct = 0
    details = []

    for prefix, expected in tqdm(completions, desc="Next-token accuracy"):
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits at last position
        last_logits = outputs.logits[0, -1, :]
        top5_ids = torch.topk(last_logits, 5).indices.tolist()
        top1_id = top5_ids[0]

        # Tokenize expected continuation
        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        expected_first = expected_ids[0] if expected_ids else -1

        top1_match = (top1_id == expected_first)
        top5_match = (expected_first in top5_ids)

        if top1_match:
            correct += 1
        if top5_match:
            top5_correct += 1

        top1_token = tokenizer.decode([top1_id])
        expected_token = tokenizer.decode([expected_first]) if expected_first >= 0 else expected

        details.append({
            "prefix": prefix,
            "expected": expected_token,
            "predicted": top1_token,
            "top1_correct": top1_match,
            "top5_correct": top5_match,
        })

    return {
        "top1_accuracy": correct / len(completions),
        "top5_accuracy": top5_correct / len(completions),
        "correct": correct,
        "total": len(completions),
        "details": details,
    }


def compute_coherence_score(model, tokenizer, n_samples: int = 20) -> dict:
    """
    Measure generation coherence via:
    1. Repetition rate (lower = more coherent)
    2. Distinct-n metrics (higher = more diverse/coherent)

    Corruption tends to increase repetition and decrease diversity.
    """
    prompts = [
        "The history of science shows that",
        "In modern society, technology has",
        "The most important thing about education is",
        "Climate change affects the planet by",
        "The human body is remarkable because",
        "Throughout history, civilizations have",
        "The ocean is important because",
        "Music has the power to",
        "The development of language allowed humans to",
        "Books are valuable because they",
        "The invention of the wheel changed",
        "Stars are formed when",
        "Philosophy helps us understand",
        "The discovery of electricity led to",
        "Agriculture transformed human society by",
        "Mathematics is the language of",
        "The arts contribute to society by",
        "Modern medicine has improved",
        "Space exploration reveals",
        "The internet has changed how people",
    ]

    all_tokens = []
    repetition_rates = []
    responses = []

    for prompt in tqdm(prompts[:n_samples], desc="Coherence scoring"):
        response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
        responses.append(response)

        tokens = response.lower().split()
        all_tokens.extend(tokens)

        # Repetition rate: fraction of tokens that are repeats of the previous token
        if len(tokens) > 1:
            repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            repetition_rates.append(repeats / (len(tokens) - 1))
        else:
            repetition_rates.append(0)

    # Distinct-n metrics
    def distinct_n(tokens, n):
        if len(tokens) < n:
            return 0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0

    distinct_1 = distinct_n(all_tokens, 1)
    distinct_2 = distinct_n(all_tokens, 2)
    distinct_3 = distinct_n(all_tokens, 3)

    # Composite coherence: higher = better
    mean_rep = float(np.mean(repetition_rates))
    coherence_score = (
        (1.0 - mean_rep) * 0.3 +  # Low repetition
        distinct_1 * 0.2 +          # Vocabulary diversity
        distinct_2 * 0.25 +          # Bigram diversity
        distinct_3 * 0.25            # Trigram diversity
    )

    return {
        "coherence_score": coherence_score,
        "mean_repetition_rate": mean_rep,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "sample_responses": responses[:5],
    }


def evaluate_reasoning(model, tokenizer, config: dict) -> dict:
    """
    Run capability evaluation:
    1. Perplexity on held-out clean text
    2. Next-token prediction accuracy on factual completions
    3. Coherence score of generated text
    """
    print("\n--- Capability Evaluation ---")

    print("  Computing perplexity...")
    perplexity = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS)

    print("  Computing next-token accuracy...")
    next_token = compute_next_token_accuracy(model, tokenizer)

    print("  Computing coherence score...")
    coherence = compute_coherence_score(model, tokenizer)

    # Aggregate capability score: normalized 0-1, higher = better
    # Perplexity: map to 0-1 (lower ppl = higher score)
    ppl_score = max(0, 1.0 - (perplexity["mean_perplexity"] - 20) / 200)
    ppl_score = min(1.0, ppl_score)

    aggregate = (
        ppl_score * 0.40 +
        next_token["top1_accuracy"] * 0.30 +
        coherence["coherence_score"] * 0.30
    )

    results = {
        "aggregate_accuracy": aggregate,
        "perplexity": perplexity,
        "next_token": next_token,
        "coherence": coherence,
        "ppl_score": ppl_score,
    }

    print(f"  Perplexity: {perplexity['mean_perplexity']:.2f}")
    print(f"  Next-token top1: {next_token['top1_accuracy']:.2%}")
    print(f"  Coherence: {coherence['coherence_score']:.4f}")
    print(f"  Aggregate capability: {aggregate:.4f}")

    return results
