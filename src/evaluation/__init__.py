"""PACE Evaluation Subpackage."""

from src.evaluation.metrics import (
    calculate_race_mae,
    calculate_spearman_rho,
    calculate_kendall_tau,
    calculate_top_k_hit_rate,
    calculate_brier_score,
    paired_wilcoxon_test,
    evaluate_race_predictions,
)
from src.evaluation.explainability import PredictionExplainer
from src.evaluation.backtest import WalkForwardBacktester
from src.evaluation.ablation import FeatureAblationStudy, FEATURE_GROUPS

__all__ = [
    "calculate_race_mae",
    "calculate_spearman_rho",
    "calculate_kendall_tau",
    "calculate_top_k_hit_rate",
    "calculate_brier_score",
    "paired_wilcoxon_test",
    "evaluate_race_predictions",
    "PredictionExplainer",
    "WalkForwardBacktester",
    "FeatureAblationStudy",
    "FEATURE_GROUPS",
]
