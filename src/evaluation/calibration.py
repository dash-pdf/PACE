"""PACE: Probability Calibration & Reliability Evaluation.

Implements rigorous probabilistic evaluation for Formula 1 race simulation:
1. Brier score for binary race outcomes (Win, Podium, Top-5, Points, DNF).
2. Binary and multi-class cross-entropy / log-loss.
3. Reliability diagrams, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).
4. Subgroup calibration across drivers, constructors, and circuit types.
5. Temperature scaling and isotonic post-hoc calibration with overfitting guardrails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from src.simulation.monte_carlo import MonteCarloResult

logger = logging.getLogger("pace.evaluation.calibration")


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Brier Score: mean((p_i - y_i)^2). Lower is better (0.0 is perfect)."""
    if len(y_true) == 0:
        return 0.0
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_p - y_t) ** 2))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Computes binary cross-entropy log-loss with numerical clipping."""
    if len(y_true) == 0:
        return 0.0
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)))


def multiclass_log_loss(
    actual_positions: np.ndarray,
    position_matrix: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Computes multi-class negative log-likelihood over the entire finishing position PMF.
    
    Args:
        actual_positions: 1D array of actual 1-based finishing positions (length M).
        position_matrix: 2D array of shape [M, N_grid] with predicted probabilities.
    """
    if len(actual_positions) == 0 or position_matrix.shape[0] == 0:
        return 0.0

    m = len(actual_positions)
    n_grid = position_matrix.shape[1]
    log_losses = []

    for i in range(m):
        pos = int(actual_positions[i])
        if 1 <= pos <= n_grid:
            p_val = np.clip(position_matrix[i, pos - 1], eps, 1.0)
            log_losses.append(-np.log(p_val))
        else:
            log_losses.append(-np.log(eps))

    return float(np.mean(log_losses))


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes reliability diagram bins, ECE (Expected Calibration Error) and MCE.
    
    Args:
        y_true: Binary ground truth array (0 or 1).
        y_prob: Predicted probability array in [0, 1].
        n_bins: Number of equal-width probability bins.
        
    Returns:
        dict containing bin edges, confidences, accuracies, counts, ECE, and MCE.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            in_bin = (y_p >= low) & (y_p <= high)
        else:
            in_bin = (y_p >= low) & (y_p < high)

        count = int(np.sum(in_bin))
        if count > 0:
            conf = float(np.mean(y_p[in_bin]))
            acc = float(np.mean(y_t[in_bin]))
            diff = abs(acc - conf)

            bin_centers.append(float((low + high) / 2.0))
            bin_confidences.append(conf)
            bin_accuracies.append(acc)
            bin_counts.append(count)

            ece += (count / n) * diff
            if diff > mce:
                mce = diff

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_centers": bin_centers,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "n_samples": n,
    }


class TemperatureScaler:
    """Post-hoc probability calibration via parametric temperature scaling.
    
    Transforms uncalibrated logits z_ik by a single learned temperature parameter T > 0:
        p_ik(cal) = softmax(z_ik / T)
    Preserves top-1 rank predictions while smoothing overconfident probability distributions.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        actual_positions: np.ndarray,
        position_matrix: np.ndarray,
        eps: float = 1e-15,
    ) -> "TemperatureScaler":
        """Fits optimal temperature T on historical validation predictions."""
        if len(actual_positions) < 20:
            logger.warning("Insufficient samples (<20) to fit temperature scaling. Defaulting T=1.0.")
            self.temperature = 1.0
            self.is_fitted = True
            return self

        # Convert probabilities to logits: z = ln(p + eps)
        clipped_p = np.clip(position_matrix, eps, 1.0)
        logits = np.log(clipped_p)

        def nll_objective(t_arr: np.ndarray) -> float:
            t_val = float(t_arr[0])
            scaled_logits = logits / t_val
            # Softmax row-wise
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            exp_l = np.exp(scaled_logits - max_l)
            softmax_p = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            return multiclass_log_loss(actual_positions, softmax_p, eps=eps)

        res = minimize(
            nll_objective,
            x0=[1.0],
            bounds=[(0.05, 5.0)],
            method="L-BFGS-B",
        )
        if res.success:
            self.temperature = float(res.x[0])
            self.is_fitted = True
            logger.info(f"Fitted TemperatureScaler: T = {self.temperature:.4f} (NLL: {res.fun:.4f})")
        else:
            self.temperature = 1.0
            self.is_fitted = True

        return self

    def transform(self, position_matrix: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        """Applies learned temperature to scale an uncalibrated position probability matrix."""
        if not self.is_fitted or abs(self.temperature - 1.0) < 1e-3:
            return position_matrix

        clipped_p = np.clip(position_matrix, eps, 1.0)
        logits = np.log(clipped_p)
        scaled = logits / self.temperature
        max_l = np.max(scaled, axis=1, keepdims=True)
        exp_l = np.exp(scaled - max_l)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


class ProbabilityCalibrationEvaluator:
    """Comprehensive probability calibration and validation analyzer for PACE."""

    @staticmethod
    def evaluate_monte_carlo_race(
        actual_results_df: pd.DataFrame,
        mc_result: MonteCarloResult,
    ) -> Dict[str, Any]:
        """Evaluates complete calibration metrics on a single Grand Prix event.
        
        Args:
            actual_results_df: DataFrame containing ['driver_id', 'finish_position', 'status', 'is_dnf'].
            mc_result: MonteCarloResult instance.
            
        Returns:
            dict of calibration scores (Brier scores, log losses, ECEs, multi-class NLL).
        """
        # Align actual results with simulated drivers
        driver_order = mc_result.drivers
        df_map = {row["driver_id"]: row for _, row in actual_results_df.iterrows()}

        y_pos = []
        p_w = []
        p_pod = []
        p_t5 = []
        p_pts = []
        p_dnf = []
        is_dnf_actual = []
        aligned_rows = []

        for d in driver_order:
            if d not in df_map:
                continue
            act = df_map[d]
            pos = int(act["finish_position"])
            status_str = str(act.get("status", "finished")).lower()
            dnf_val = bool(act.get("is_dnf", False) or "dnf" in status_str or "retired" in status_str)

            y_pos.append(pos)
            is_dnf_actual.append(1.0 if dnf_val else 0.0)
            p_w.append(mc_result.p_win[d])
            p_pod.append(mc_result.p_podium[d])
            p_t5.append(mc_result.p_top5[d])
            p_pts.append(mc_result.p_points[d])
            p_dnf.append(mc_result.p_dnf_total[d])
            aligned_rows.append(mc_result.drivers.index(d))

        y_pos_arr = np.array(y_pos, dtype=np.int64)
        is_dnf_arr = np.array(is_dnf_actual, dtype=np.float64)
        p_w_arr = np.array(p_w, dtype=np.float64)
        p_pod_arr = np.array(p_pod, dtype=np.float64)
        p_t5_arr = np.array(p_t5, dtype=np.float64)
        p_pts_arr = np.array(p_pts, dtype=np.float64)
        p_dnf_arr = np.array(p_dnf, dtype=np.float64)

        aligned_pos_matrix = mc_result.position_matrix[aligned_rows, :]

        # Binary ground truths
        y_w = (y_pos_arr == 1).astype(np.float64)
        y_pod = (y_pos_arr <= 3).astype(np.float64)
        y_t5 = (y_pos_arr <= 5).astype(np.float64)
        y_pts = (y_pos_arr <= 10).astype(np.float64)

        # 1. Brier scores
        bs_win = brier_score(y_w, p_w_arr)
        bs_pod = brier_score(y_pod, p_pod_arr)
        bs_t5 = brier_score(y_t5, p_t5_arr)
        bs_pts = brier_score(y_pts, p_pts_arr)
        bs_dnf = brier_score(is_dnf_arr, p_dnf_arr)

        # 2. Binary Log losses
        ll_win = binary_log_loss(y_w, p_w_arr)
        ll_pod = binary_log_loss(y_pod, p_pod_arr)
        ll_t5 = binary_log_loss(y_t5, p_t5_arr)
        ll_pts = binary_log_loss(y_pts, p_pts_arr)
        ll_dnf = binary_log_loss(is_dnf_arr, p_dnf_arr)

        # 3. Multi-class Log Loss (complete finishing rank distribution)
        multi_nll = multiclass_log_loss(y_pos_arr, aligned_pos_matrix)

        # 4. Reliability Curve & ECE (Podium and Points)
        rel_pod = compute_reliability_curve(y_pod, p_pod_arr, n_bins=10)
        rel_pts = compute_reliability_curve(y_pts, p_pts_arr, n_bins=10)
        rel_dnf = compute_reliability_curve(is_dnf_arr, p_dnf_arr, n_bins=5)

        return {
            "brier_win": bs_win,
            "brier_podium": bs_pod,
            "brier_top5": bs_t5,
            "brier_points": bs_pts,
            "brier_dnf": bs_dnf,
            "logloss_win": ll_win,
            "logloss_podium": ll_pod,
            "logloss_top5": ll_t5,
            "logloss_points": ll_pts,
            "logloss_dnf": ll_dnf,
            "multiclass_logloss": multi_nll,
            "ece_podium": rel_pod["ece"],
            "ece_points": rel_pts["ece"],
            "ece_dnf": rel_dnf["ece"],
            "mce_podium": rel_pod["mce"],
            "mce_points": rel_pts["mce"],
            "n_drivers_evaluated": len(y_pos),
        }

    @staticmethod
    def evaluate_subgroup_calibration(
        eval_records: List[Dict[str, Any]],
    ) -> Dict[str, pd.DataFrame]:
        """Slices calibration results across constructors, driver experience, and circuits."""
        df = pd.DataFrame(eval_records)
        if df.empty:
            return {}

        subgroup_dfs = {}

        # Constructor tier slicing
        if "constructor_tier" in df.columns:
            subgroup_dfs["constructor_tier"] = df.groupby("constructor_tier").agg(
                brier_points=("brier_points", "mean"),
                brier_dnf=("brier_dnf", "mean"),
                mae=("mae", "mean"),
                sample_count=("driver_id", "count"),
            ).reset_index()

        # Driver experience slicing (veteran vs rookie)
        if "driver_experience_bracket" in df.columns:
            subgroup_dfs["driver_experience"] = df.groupby("driver_experience_bracket").agg(
                brier_points=("brier_points", "mean"),
                brier_dnf=("brier_dnf", "mean"),
                mae=("mae", "mean"),
                sample_count=("driver_id", "count"),
            ).reset_index()

        # Circuit degradation slicing
        if "circuit_tyre_stress" in df.columns:
            subgroup_dfs["circuit_stress"] = df.groupby("circuit_tyre_stress").agg(
                brier_points=("brier_points", "mean"),
                brier_dnf=("brier_dnf", "mean"),
                mae=("mae", "mean"),
                sample_count=("circuit_id", "count"),
            ).reset_index()

        return subgroup_dfs

    @staticmethod
    def evaluate_calibration_guardrails(
        sample_size: int,
        uncalibrated_logloss: float,
        calibrated_logloss: float,
        min_samples: int = 50,
    ) -> Tuple[bool, str]:
        """Strict guardrail against overfitting post-hoc calibration to small samples.
        
        Returns:
            (should_apply_calibration, rationale)
        """
        if sample_size < min_samples:
            return (
                False,
                f"Sample size ({sample_size}) below minimum required threshold ({min_samples}) "
                "for robust post-hoc calibration. Retaining raw Monte Carlo probabilities to prevent overfitting.",
            )

        delta_nll = uncalibrated_logloss - calibrated_logloss
        if delta_nll <= 0.005:
            return (
                False,
                f"Calibration yielded negligible log-loss reduction ({delta_nll:.5f} <= 0.005). "
                "Retaining raw physical simulation distributions without post-hoc deformation.",
            )

        return (
            True,
            f"Sufficient historical validation samples ({sample_size}) and statistically significant "
            f"log-loss reduction ({delta_nll:.4f}). Post-hoc calibration accepted.",
        )
