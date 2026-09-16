"""Pure NumPy metrics for canonical residual diagnostics."""

from __future__ import annotations

import numpy as np


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def calibration_metrics(
    logits: np.ndarray, labels: np.ndarray, bins: int = 15
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    probabilities = softmax(logits)
    log_probabilities = log_softmax(logits)
    target = np.arange(len(labels))
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    confidence = probabilities.max(axis=1)
    nll = -log_probabilities[target, labels]
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    one_hot = np.eye(len(EMOTIONS))[labels]
    brier = ((probabilities - one_hot) ** 2).sum(axis=1)
    per_class: list[dict[str, float | int | str]] = []
    class_nll = []
    recalls = []
    for class_id, emotion in enumerate(EMOTIONS):
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"No examples for emotion {emotion}")
        recall = float(np.mean(predictions[mask] == class_id))
        current_nll = float(nll[mask].mean())
        recalls.append(recall)
        class_nll.append(current_nll)
        per_class.append({
            "emotion": emotion,
            "count": int(mask.sum()),
            "recall": recall,
            "nll": current_nll,
            "mean_confidence": float(confidence[mask].mean()),
            "mean_correct_probability": float(probabilities[mask, class_id].mean()),
            "brier": float(brier[mask].mean()),
        })
    metrics = {
        "uar": float(np.mean(recalls)),
        "war": float(correct.mean()),
        "micro_nll": float(nll.mean()),
        "macro_nll": float(np.mean(class_nll)),
        "brier": float(brier.mean()),
        "ece_15": expected_calibration_error(probabilities, labels, bins),
        "mean_entropy": float(entropy.mean()),
        "mean_confidence": float(confidence.mean()),
        "correct_confidence": float(confidence[correct].mean()) if correct.any() else float("nan"),
        "incorrect_confidence": float(confidence[~correct].mean()) if (~correct).any() else float("nan"),
        "overconfident_error_rate_080": float(np.mean((~correct) & (confidence >= 0.8))),
    }
    return metrics, per_class


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def sample_residual_diagnostics(
    context_logits: np.ndarray,
    delta_logits: np.ndarray,
    final_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    context_probability = softmax(context_logits)
    final_probability = softmax(final_logits)
    context_log = log_softmax(context_logits)
    final_log = log_softmax(final_logits)
    target = np.arange(len(labels))
    context_prediction = context_logits.argmax(axis=1)
    final_prediction = final_logits.argmax(axis=1)
    context_correct = context_prediction == labels
    final_correct = final_prediction == labels
    flip_category = np.full(len(labels), "unchanged", dtype="U16")
    changed = context_prediction != final_prediction
    flip_category[changed & ~context_correct & final_correct] = "beneficial"
    flip_category[changed & context_correct & ~final_correct] = "harmful"
    flip_category[changed & ~context_correct & ~final_correct] = "wrong_to_wrong"
    context_entropy = -(context_probability * context_log).sum(axis=1)
    final_entropy = -(final_probability * final_log).sum(axis=1)
    context_nll = -context_log[target, labels]
    final_nll = -final_log[target, labels]
    return {
        "delta_l2": np.linalg.norm(delta_logits, axis=1),
        "delta_linf": np.abs(delta_logits).max(axis=1),
        "context_prediction": context_prediction,
        "final_prediction": final_prediction,
        "context_correct": context_correct,
        "final_correct": final_correct,
        "flip_category": flip_category,
        "context_confidence": context_probability.max(axis=1),
        "final_confidence": final_probability.max(axis=1),
        "confidence_delta": final_probability.max(axis=1) - context_probability.max(axis=1),
        "context_correct_probability": context_probability[target, labels],
        "final_correct_probability": final_probability[target, labels],
        "correct_probability_delta": (
            final_probability[target, labels] - context_probability[target, labels]
        ),
        "context_entropy": context_entropy,
        "final_entropy": final_entropy,
        "entropy_delta": final_entropy - context_entropy,
        "context_nll": context_nll,
        "final_nll": final_nll,
        "nll_change": final_nll - context_nll,
    }
