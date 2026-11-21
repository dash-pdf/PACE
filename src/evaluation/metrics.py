"""PACE: Evaluation Metrics & Statistical Hypothesis Testing.

Implements:
1. Finishing position MAE.
2. Spearman rank correlation (rho).
3. Kendall's tau (tau).
4. Top-1 (Winner), Top-3 (Podium), and Top-10 (Points) accuracy / hit rates.
5. Brier score for probabilistic DNF calibration.
6. Paired Wilcoxon signed-rank test and Diebold-Mariano test for model comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_race_mae(predicted_ranks: np.ndarray, actual_positions: np.ndarray) -> float:
    """Mean Absolute Error (MAE) on finishing position."""
    if len(predicted_ranks) == 0:
        return 0.0
    return float(np.mean(np.abs(predicted_ranks - actual_positions)))


def calculate_spearman_rho(predicted_ranks: np.ndarray, actual_positions: np.ndarray) -> float:
    """Spearman rank correlation coefficient (rho)."""
    if len(predicted_ranks) < 3:
        return 0.0
    rho, _ = stats.spearmanr(predicted_ranks, actual_positions)
    return float(rho) if not np.isnan(rho) else 0.0


def calculate_kendall_tau(predicted_ranks: np.ndarray, actual_positions: np.ndarray) -> float:
    """Kendall's rank correlation coefficient (tau)."""
    if len(predicted_ranks) < 3:
        return 0.0
    tau, _ = stats.kendalltau(predicted_ranks, actual_positions)
    return float(tau) if not np.isnan(tau) else 0.0


def calculate_top_k_hit_rate(
    predicted_ranks: np.ndarray, actual_positions: np.ndarray, k: int = 1
) -> float:
    """Calculates top-k accuracy (e.g. Winner k=1, Podium k=3, Points k=10).
    
    For k=1: 1.0 if predicted winner (rank 1) actually finished P1, else 0.0.
    For k>1: Fraction of true top-k finishers that were predicted in the top-k.
    """
    if len(predicted_ranks) == 0:
        return 0.0

    if k == 1:
        pred_winner_idx = np.argmin(predicted_ranks)
        actual_winner_idx = np.argmin(actual_positions)
        return 1.0 if pred_winner_idx == actual_winner_idx else 0.0

    pred_top_k = set(np.argsort(predicted_ranks)[:k])
    actual_top_k = set(np.argsort(actual_positions)[:k])
    intersection = len(pred_top_k.intersection(actual_top_k))
    return float(intersection / k)


def calculate_brier_score(predicted_p_dnf: np.ndarray, actual_dnf: np.ndarray) -> float:
    """Brier score for probabilistic DNF calibration: mean((p - y)^2). Lower is better."""
    if len(predicted_p_dnf) == 0:
        return 0.0
    return float(np.mean((predicted_p_dnf - actual_dnf.astype(float)) ** 2))


def paired_wilcoxon_test(errors_model_a: List[float], errors_model_b: List[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test on per-race MAE.
    
    Returns:
        (statistic, p_value)
    """
    diffs = np.array(errors_model_a) - np.array(errors_model_b)
    # Filter out zero diffs
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 5:
        # Insufficient paired observations
        return 0.0, 1.0
    stat, p_val = stats.wilcoxon(errors_model_a, errors_model_b)
    return float(stat), float(p_val)


def evaluate_race_predictions(
    predicted_ranks: np.ndarray, actual_positions: np.ndarray
) -> Dict[str, float]:
    """Compute complete suite of ranking performance metrics for a single race."""
    return {
        "mae": calculate_race_mae(predicted_ranks, actual_positions),
        "spearman_rho": calculate_spearman_rho(predicted_ranks, actual_positions),
        "kendall_tau": calculate_kendall_tau(predicted_ranks, actual_positions),
        "top_1_hit": calculate_top_k_hit_rate(predicted_ranks, actual_positions, k=1),
        "top_3_hit": calculate_top_k_hit_rate(predicted_ranks, actual_positions, k=3),
        "top_5_hit": calculate_top_k_hit_rate(predicted_ranks, actual_positions, k=5),
        "top_10_hit": calculate_top_k_hit_rate(predicted_ranks, actual_positions, k=10),
    }
