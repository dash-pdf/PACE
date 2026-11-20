"""PACE: Systematic Empirical Feature Selection & Ablation Study.

Implements Leave-One-Group-Out (LOGO) and Forward-Group-Addition ablation testing
across identical chronological walk-forward validation splits (Occam's Razor protocol).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.ranker import XGBoostLambdaMARTRanker
from src.models.statistical import RidgePositionRegressor
from src.evaluation.metrics import evaluate_race_predictions

logger = logging.getLogger("pace.evaluation.ablation")


FEATURE_GROUPS = {
    "G1_Grid_Order": ["grid_position"],
    "G2_Driver_DPI": ["dpi_rating", "dpi_uncertainty", "dpi_volatility"],
    "G3_Decoupled_Experience": ["driver_starts", "rookie_learning_factor", "driver_consistency_cv"],
    "G4_Constructor_CPI": ["cpi_pole_gap_pct", "cpi_reliability", "cpi_development_trend"],
    "G5_Inferred_Fuel_Telemetry": ["fuel_sensitivity", "deg_x_stress", "speed_x_length"],
    "G6_Circuit_Intelligence": [
        "circuit_affinity_mean", "circuit_affinity_se", "circuit_length_km",
        "overtaking_difficulty", "tyre_stress_level", "downforce_level", "altitude_m",
    ],
    "G7_Physics_Interactions": ["grid_x_overtaking", "deg_x_stress", "speed_x_length"],
}


class FeatureAblationStudy:
    """Evaluates the marginal out-of-sample contribution of each feature group."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        min_train_events: int = 5,
        horizon: str = "H3",
    ):
        self.pipeline = pipeline
        self.min_train_events = min_train_events
        self.horizon = horizon

        # Master feature list
        self.all_features = sorted(list(set(
            f for group in FEATURE_GROUPS.values() for f in group
        )))

    def run_logo_ablation(
        self,
        max_events: Optional[int] = None,
    ) -> Dict[str, object]:
        """Execute Leave-One-Group-Out (LOGO) ablation across all historical events.
        
        Compares:
        - Baseline: Full Feature Model
        - For each group k: Model trained WITHOUT group k.
        
        Returns:
            ablation_results: DataFrame of MAE, delta_MAE, and pruning recommendation.
        """
        all_events = (
            self.pipeline.race_results[["season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .values
        )

        eval_events = [(int(s), int(r)) for s, r in all_events[self.min_train_events:]]
        if max_events is not None:
            eval_events = eval_events[:max_events]

        logger.info(f"Running Feature Group Ablation Study across {len(eval_events)} events...")

        # Pre-cache compiled event features
        event_features_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

        def get_features(s: int, r: int) -> pd.DataFrame:
            key = (s, r)
            if key not in event_features_cache:
                event_features_cache[key] = self.pipeline.compile_features_for_event(
                    s, r, horizon=self.horizon
                )
            return event_features_cache[key]

        # Define configurations to test
        configs: Dict[str, List[str]] = {"Full_Model": self.all_features}
        for group_name, group_cols in FEATURE_GROUPS.items():
            ablated_cols = [c for c in self.all_features if c not in group_cols]
            configs[f"Without_{group_name}"] = ablated_cols

        config_errors: Dict[str, List[float]] = {name: [] for name in configs}
        config_rhos: Dict[str, List[float]] = {name: [] for name in configs}

        for season, round_num in eval_events:
            test_df = get_features(season, round_num)
            if test_df.empty or len(test_df) < 10:
                continue

            actual_pos = test_df["finish_position"].values

            # Training set strictly prior to this test event
            prior_events = [
                (int(s), int(r)) for s, r in all_events
                if (s < season) or (s == season and r < round_num)
            ]

            train_dfs = []
            groups = []
            for tr_s, tr_r in prior_events:
                tr_feat = get_features(tr_s, tr_r)
                if not tr_feat.empty and len(tr_feat) >= 10:
                    train_dfs.append(tr_feat)
                    groups.append(len(tr_feat))

            if not train_dfs:
                continue

            train_df = pd.concat(train_dfs, ignore_index=True)
            y_train = train_df["target_rank"].values

            for config_name, feature_subset in configs.items():
                X_tr = train_df[feature_subset].fillna(0.0)
                X_te = test_df[feature_subset].fillna(0.0)

                # Train Ridge Regressor on identical split
                model = RidgePositionRegressor(alpha=10.0).fit(X_tr, y_train)
                pred = model.predict(X_te)
                metrics = evaluate_race_predictions(pred, actual_pos)

                config_errors[config_name].append(metrics["mae"])
                config_rhos[config_name].append(metrics["spearman_rho"])

        # Aggregate summary table
        full_mae = float(np.mean(config_errors["Full_Model"]))
        summary_rows = []

        for config_name in configs:
            maes = config_errors[config_name]
            rhos = config_rhos[config_name]
            if not maes:
                continue

            mean_mae = float(np.mean(maes))
            mean_rho = float(np.mean(rhos))
            # Delta MAE = MAE(Ablated) - MAE(Full)
            # Positive delta means Full model was better (Ablation degraded performance -> Group is valuable)
            # Negative delta means Ablation improved performance (Group was hurting or noisy)
            delta_mae = mean_mae - full_mae
            pct_delta = (delta_mae / full_mae) * 100.0

            # Pruning recommendation
            if config_name == "Full_Model":
                action = "REFERENCE"
            elif pct_delta >= 1.0:
                action = "RETAIN (Significant Benefit >= 1%)"
            elif pct_delta > 0.0:
                action = "RETAIN (Marginal Benefit < 1%)"
            else:
                action = "CANDIDATE FOR PRUNING (No Benefit / Noise)"

            summary_rows.append({
                "Configuration": config_name,
                "MAE_Mean": mean_mae,
                "Spearman_Rho": mean_rho,
                "Delta_MAE": delta_mae,
                "Impact_Pct": pct_delta,
                "Recommendation": action,
            })

        ablation_df = pd.DataFrame(summary_rows)
        return {
            "ablation_summary": ablation_df,
            "full_model_mae": full_mae,
            "raw_errors": config_errors,
        }
