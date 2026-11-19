"""PACE: Walk-Forward Chronological Backtesting Engine.

Executes leakage-free expanding window cross-validation across F1 seasons.
Evaluates Tier 1 baselines, Tier 2 statistical models, Tier 3 LMM,
and Tier 4 GBDT ranking models on identical chronological splits.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.pipeline import FeaturePipeline
from src.models.baselines import (
    GridOrderBaseline,
    ChampionshipStandingsBaseline,
    RollingConstructorBaseline,
)
from src.models.statistical import (
    RidgePositionRegressor,
    OrdinalLogisticModel,
    HierarchicalLMMRanker,
)
from src.models.ranker import (
    XGBoostPairwiseRanker,
    XGBoostLambdaMARTRanker,
)
from src.evaluation.metrics import (
    evaluate_race_predictions,
    paired_wilcoxon_test,
)

logger = logging.getLogger("pace.evaluation.backtest")


class WalkForwardBacktester:
    """Expanding-window walk-forward backtester across Formula 1 seasons."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        min_train_events: int = 5,
        horizon: str = "H3",
    ):
        self.pipeline = pipeline
        self.min_train_events = min_train_events
        self.horizon = horizon

        self.feature_columns = [
            "dpi_rating",
            "dpi_uncertainty",
            "dpi_volatility",
            "driver_starts",
            "rookie_learning_factor",
            "driver_consistency_cv",
            "cpi_pole_gap_pct",
            "cpi_reliability",
            "cpi_development_trend",
            "circuit_affinity_mean",
            "circuit_affinity_se",
            "circuit_length_km",
            "overtaking_difficulty",
            "tyre_stress_level",
            "downforce_level",
            "altitude_m",
            "fuel_sensitivity",
            "grid_position",
            "grid_x_overtaking",
            "deg_x_stress",
            "speed_x_length",
        ]

    def run_backtest(
        self,
        test_events: Optional[List[Tuple[int, int]]] = None,
        max_events: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run walk-forward evaluation across chronological events.
        
        Args:
            test_events: Optional explicit list of (season, round) tuples.
            max_events: Optional cap on evaluated events for fast testing.
            
        Returns:
            results_dict containing per-race metrics, aggregate summary, and statistical significance.
        """
        # Discover all available chronological events
        all_events = (
            self.pipeline.race_results[["season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .values
        )

        if test_events is None:
            # Use all events after min_train_events warm-up
            eval_events = [(int(s), int(r)) for s, r in all_events[self.min_train_events:]]
        else:
            eval_events = test_events

        if max_events is not None:
            eval_events = eval_events[:max_events]

        logger.info(f"Executing walk-forward backtest across {len(eval_events)} Grand Prix events...")

        # Cache compiled features per event to avoid redundant rebuilds
        event_features_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

        def get_event_features(s: int, r: int) -> pd.DataFrame:
            key = (s, r)
            if key not in event_features_cache:
                df = self.pipeline.compile_features_for_event(s, r, horizon=self.horizon)
                event_features_cache[key] = df
            return event_features_cache[key]

        model_race_metrics: Dict[str, List[Dict[str, float]]] = {
            "Baseline_GridOrder": [],
            "Baseline_Championship": [],
            "Baseline_RollingConstructor": [],
            "Statistical_Ridge": [],
            "Statistical_OrdinalLogistic": [],
            "Statistical_HierarchicalLMM": [],
            "ML_XGBoost_Pairwise": [],
            "ML_XGBoost_LambdaMART": [],
        }

        # Initialize baseline model instances
        grid_baseline = GridOrderBaseline()
        champ_baseline = ChampionshipStandingsBaseline()
        team_baseline = RollingConstructorBaseline()

        for idx, (season, round_num) in enumerate(eval_events):
            test_df = get_event_features(season, round_num)
            if test_df.empty or len(test_df) < 10:
                continue

            actual_pos = test_df["finish_position"].values

            # 1. Evaluate Tier 1 Baselines
            # Grid baseline
            pred_grid = grid_baseline.predict(test_df)
            model_race_metrics["Baseline_GridOrder"].append(
                evaluate_race_predictions(pred_grid, actual_pos)
            )

            # Prior season results for championship baseline
            prior_season_rr = self.pipeline.race_results[
                (self.pipeline.race_results["season"] == season) &
                (self.pipeline.race_results["round"] < round_num)
            ]
            pred_champ = champ_baseline.predict_with_history(test_df, prior_season_rr)
            model_race_metrics["Baseline_Championship"].append(
                evaluate_race_predictions(pred_champ, actual_pos)
            )

            # Rolling constructor baseline
            prior_all_rr = self.pipeline.race_results[
                (self.pipeline.race_results["season"] < season) |
                ((self.pipeline.race_results["season"] == season) & (self.pipeline.race_results["round"] < round_num))
            ]
            pred_team = team_baseline.predict_with_history(test_df, prior_all_rr)
            model_race_metrics["Baseline_RollingConstructor"].append(
                evaluate_race_predictions(pred_team, actual_pos)
            )

            # 2. Build training set strictly prior to this test event
            prior_events = [
                (int(s), int(r)) for s, r in all_events
                if (s < season) or (s == season and r < round_num)
            ]

            train_dfs = []
            groups = []
            for tr_s, tr_r in prior_events:
                tr_feat = get_event_features(tr_s, tr_r)
                if not tr_feat.empty and len(tr_feat) >= 10:
                    train_dfs.append(tr_feat)
                    groups.append(len(tr_feat))

            if not train_dfs:
                continue

            train_df = pd.concat(train_dfs, ignore_index=True)
            X_train = train_df[self.feature_columns].fillna(0.0)
            y_train = train_df["target_rank"].values

            X_test = test_df[self.feature_columns].fillna(0.0)

            # 3. Evaluate Tier 2 Models
            # Ridge
            ridge = RidgePositionRegressor(alpha=10.0).fit(X_train, y_train)
            pred_ridge = ridge.predict(X_test)
            model_race_metrics["Statistical_Ridge"].append(
                evaluate_race_predictions(pred_ridge, actual_pos)
            )

            # Ordinal Logistic
            try:
                ord_log = OrdinalLogisticModel(C=1.0).fit(X_train, y_train)
                pred_ord = ord_log.predict(X_test)
            except Exception:
                pred_ord = pred_ridge
            model_race_metrics["Statistical_OrdinalLogistic"].append(
                evaluate_race_predictions(pred_ord, actual_pos)
            )

            # 4. Evaluate Tier 3 LMM
            lmm = HierarchicalLMMRanker().fit(train_df)
            pred_lmm = lmm.predict(test_df)
            model_race_metrics["Statistical_HierarchicalLMM"].append(
                evaluate_race_predictions(pred_lmm, actual_pos)
            )

            # 5. Evaluate Tier 4 XGBoost Models
            # Pairwise
            xgb_pair = XGBoostPairwiseRanker(n_estimators=60, max_depth=3, learning_rate=0.08)
            xgb_pair.fit(X_train, y_train, groups=groups)
            pred_pair = xgb_pair.predict(X_test)
            model_race_metrics["ML_XGBoost_Pairwise"].append(
                evaluate_race_predictions(pred_pair, actual_pos)
            )

            # Listwise LambdaMART
            xgb_lambda = XGBoostLambdaMARTRanker(n_estimators=60, max_depth=3, learning_rate=0.08)
            xgb_lambda.fit(X_train, y_train, groups=groups)
            pred_lambda = xgb_lambda.predict(X_test)
            model_race_metrics["ML_XGBoost_LambdaMART"].append(
                evaluate_race_predictions(pred_lambda, actual_pos)
            )

        # 6. Aggregate Summary Table
        summary_rows = []
        for model_name, metrics_list in model_race_metrics.items():
            if not metrics_list:
                continue
            m_df = pd.DataFrame(metrics_list)
            summary_rows.append({
                "Model": model_name,
                "MAE_Mean": float(m_df["mae"].mean()),
                "MAE_Std": float(m_df["mae"].std()),
                "Spearman_Rho": float(m_df["spearman_rho"].mean()),
                "Kendall_Tau": float(m_df["kendall_tau"].mean()),
                "Top1_Accuracy": float(m_df["top_1_hit"].mean()),
                "Top3_Accuracy": float(m_df["top_3_hit"].mean()),
                "Top10_Accuracy": float(m_df["top_10_hit"].mean()),
                "Races_Evaluated": len(m_df),
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("MAE_Mean")

        # 7. Significance testing between LambdaMART and Grid Baseline / LMM
        grid_maes = [m["mae"] for m in model_race_metrics["Baseline_GridOrder"]]
        lambda_maes = [m["mae"] for m in model_race_metrics["ML_XGBoost_LambdaMART"]]
        lmm_maes = [m["mae"] for m in model_race_metrics["Statistical_HierarchicalLMM"]]

        stat_grid, p_val_grid = paired_wilcoxon_test(lambda_maes, grid_maes)
        stat_lmm, p_val_lmm = paired_wilcoxon_test(lambda_maes, lmm_maes)

        significance = {
            "LambdaMART_vs_GridBaseline": {
                "wilcoxon_statistic": stat_grid,
                "p_value": p_val_grid,
                "statistically_significant_p01": bool(p_val_grid < 0.01),
            },
            "LambdaMART_vs_HierarchicalLMM": {
                "wilcoxon_statistic": stat_lmm,
                "p_value": p_val_lmm,
                "statistically_significant_p01": bool(p_val_lmm < 0.01),
            },
        }

        return {
            "summary_df": summary_df,
            "significance": significance,
            "raw_metrics": model_race_metrics,
        }
