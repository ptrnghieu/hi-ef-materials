"""Pure NumPy helpers for cross-fitted reliability diagnostics."""

from __future__ import annotations

import numpy as np


TEMPERATURE_GRID = np.geomspace(0.05, 20.0, 1201)
ALPHA_GRID = np.linspace(0.0, 1.0, 501)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def sample_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -log_softmax(logits)[np.arange(len(labels)), labels]


def mean_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(sample_nll(logits, labels).mean())


def select_grid(
    losses: np.ndarray, grid: np.ndarray, neutral_value: float
) -> tuple[float, float]:
    """Select minimum loss, resolving numerical ties toward the neutral value."""
    minimum = float(losses.min())
    tied = np.flatnonzero(np.isclose(losses, minimum, rtol=0.0, atol=1e-12))
    selected = tied[np.argmin(np.abs(grid[tied] - neutral_value))]
    return float(grid[selected]), minimum


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    losses = np.asarray([mean_nll(logits / value, labels) for value in TEMPERATURE_GRID])
    return select_grid(losses, TEMPERATURE_GRID, 1.0)


def fit_alpha(
    context_logits: np.ndarray,
    delta_logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    losses = np.asarray([
        mean_nll(context_logits + value * delta_logits, labels)
        for value in ALPHA_GRID
    ])
    return select_grid(losses, ALPHA_GRID, 1.0)


def cross_fit(
    context_logits: np.ndarray,
    delta_logits: np.ndarray,
    final_logits: np.ndarray,
    labels: np.ndarray,
    folders: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    temperature_logits = np.empty_like(final_logits)
    shrinkage_logits = np.empty_like(final_logits)
    rows: list[dict[str, object]] = []
    for folder in np.unique(folders):
        held_out = folders == folder
        fitting = ~held_out
        temperature, temperature_fit_nll = fit_temperature(
            final_logits[fitting], labels[fitting]
        )
        alpha, alpha_fit_nll = fit_alpha(
            context_logits[fitting], delta_logits[fitting], labels[fitting]
        )
        temperature_logits[held_out] = final_logits[held_out] / temperature
        shrinkage_logits[held_out] = (
            context_logits[held_out] + alpha * delta_logits[held_out]
        )
        rows.append({
            "held_out_source_folder": str(folder),
            "fit_samples": int(fitting.sum()),
            "held_out_samples": int(held_out.sum()),
            "temperature": temperature,
            "temperature_fit_nll": temperature_fit_nll,
            "temperature_at_boundary": bool(
                temperature == TEMPERATURE_GRID[0]
                or temperature == TEMPERATURE_GRID[-1]
            ),
            "alpha": alpha,
            "alpha_fit_nll": alpha_fit_nll,
            "alpha_at_boundary": bool(
                alpha == ALPHA_GRID[0] or alpha == ALPHA_GRID[-1]
            ),
        })
    return {
        "temperature": temperature_logits,
        "shrinkage": shrinkage_logits,
    }, rows


def oracle_logits(
    context_logits: np.ndarray,
    final_logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    context_nll = sample_nll(context_logits, labels)
    final_nll = sample_nll(final_logits, labels)
    context_prediction = context_logits.argmax(axis=1)
    final_prediction = final_logits.argmax(axis=1)
    context_correct = context_prediction == labels
    final_correct = final_prediction == labels

    choose_final_nll = final_nll < context_nll
    choose_final_accuracy = final_correct & ~context_correct
    equal_correctness = final_correct == context_correct
    choose_final_accuracy |= equal_correctness & choose_final_nll

    return {
        "oracle_nll": np.where(choose_final_nll[:, None], final_logits, context_logits),
        "oracle_accuracy": np.where(
            choose_final_accuracy[:, None], final_logits, context_logits
        ),
    }, {
        "oracle_nll_choose_final": choose_final_nll,
        "oracle_accuracy_choose_final": choose_final_accuracy,
        "context_correct": context_correct,
        "final_correct": final_correct,
    }
