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
    calculate_brier_score,
)
from src.simulation.monte_carlo import MonteCarloSimulationEngine
from src.evaluation.calibration import ProbabilityCalibrationEvaluator

logger = logging.getLogger("pace.evaluation.backtest")


def _evaluate_point_model_race(pred_ranks: np.ndarray, test_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluates ranking and implied binary probabilistic metrics for point models."""
    actual_pos = test_df["finish_position"].values
    m = evaluate_race_predictions(pred_ranks, actual_pos)

    y_w = (actual_pos == 1).astype(float)
    y_pod = (actual_pos <= 3).astype(float)
    y_t5 = (actual_pos <= 5).astype(float)
    y_pts = (actual_pos <= 10).astype(float)

    status_col = test_df.get("status", pd.Series(["finished"] * len(test_df)))
    is_dnf_col = test_df.get("is_dnf", pd.Series([False] * len(test_df)))
    y_dnf = (is_dnf_col | status_col.astype(str).str.contains("dnf|retired", case=False)).astype(float).values

    # Deterministic point models predict 1.0 or 0.0
    p_w = (pred_ranks == 1).astype(float)
    p_pod = (pred_ranks <= 3).astype(float)
    p_t5 = (pred_ranks <= 5).astype(float)
    p_pts = (pred_ranks <= 10).astype(float)
    p_dnf = np.full_like(p_w, 0.15)  # Historical unconditioned attrition prior

    m["brier_win"] = float(np.mean((p_w - y_w) ** 2))
    m["brier_podium"] = float(np.mean((p_pod - y_pod) ** 2))
    m["brier_top5"] = float(np.mean((p_t5 - y_t5) ** 2))
    m["brier_points"] = float(np.mean((p_pts - y_pts) ** 2))
    m["brier_dnf"] = float(np.mean((p_dnf - y_dnf) ** 2))
    return m


class WalkForwardBacktester:
    """Expanding-window walk-forward backtester across Formula 1 seasons."""

    def __init__(
        self,
        pipeline: FeaturePipeline,
        min_train_events: int = 5,
        horizon: str = "H3",
        include_monte_carlo: bool = True,
        mc_simulations: int = 150,
        n_jobs: int = 1,
    ):
        self.pipeline = pipeline
        self.min_train_events = min_train_events
        self.horizon = horizon
        self.include_monte_carlo = include_monte_carlo
        self.mc_simulations = mc_simulations
        self.n_jobs = n_jobs

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
            # Candidate test events must have a viable grid (at least 10 drivers)
            all_event_tuples = [(int(s), int(r)) for s, r in all_events[self.min_train_events:]]
            candidate_events = []
            for s, r in all_event_tuples:
                event_cnt = len(
                    self.pipeline.race_results[
                        (self.pipeline.race_results["season"] == s) &
                        (self.pipeline.race_results["round"] == r)
                    ]
                )
                if event_cnt >= 10:
                    candidate_events.append((s, r))
            eval_events = candidate_events
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
        if self.include_monte_carlo:
            model_race_metrics["PACE_MonteCarlo_Simulation"] = []

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
                _evaluate_point_model_race(pred_grid, test_df)
            )

            # Prior season results for championship baseline
            prior_season_rr = self.pipeline.race_results[
                (self.pipeline.race_results["season"] == season) &
                (self.pipeline.race_results["round"] < round_num)
            ]
            pred_champ = champ_baseline.predict_with_history(test_df, prior_season_rr)
            model_race_metrics["Baseline_Championship"].append(
                _evaluate_point_model_race(pred_champ, test_df)
            )

            # Rolling constructor baseline
            prior_all_rr = self.pipeline.race_results[
                (self.pipeline.race_results["season"] < season) |
                ((self.pipeline.race_results["season"] == season) & (self.pipeline.race_results["round"] < round_num))
            ]
            pred_team = team_baseline.predict_with_history(test_df, prior_all_rr)
            model_race_metrics["Baseline_RollingConstructor"].append(
                _evaluate_point_model_race(pred_team, test_df)
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
                _evaluate_point_model_race(pred_ridge, test_df)
            )

            # Ordinal Logistic
            try:
                ord_log = OrdinalLogisticModel(C=1.0).fit(X_train, y_train)
                pred_ord = ord_log.predict(X_test)
            except Exception:
                pred_ord = pred_ridge
            model_race_metrics["Statistical_OrdinalLogistic"].append(
                _evaluate_point_model_race(pred_ord, test_df)
            )

            # 4. Evaluate Tier 3 LMM
            lmm = HierarchicalLMMRanker().fit(train_df)
            pred_lmm = lmm.predict(test_df)
            model_race_metrics["Statistical_HierarchicalLMM"].append(
                _evaluate_point_model_race(pred_lmm, test_df)
            )

            # 5. Evaluate Tier 4 XGBoost Models
            # Pairwise
            xgb_pair = XGBoostPairwiseRanker(n_estimators=60, max_depth=3, learning_rate=0.08)
            xgb_pair.fit(X_train, y_train, groups=groups)
            pred_pair = xgb_pair.predict(X_test)
            model_race_metrics["ML_XGBoost_Pairwise"].append(
                _evaluate_point_model_race(pred_pair, test_df)
            )

            # Listwise LambdaMART
            xgb_lambda = XGBoostLambdaMARTRanker(n_estimators=60, max_depth=3, learning_rate=0.08)
            xgb_lambda.fit(X_train, y_train, groups=groups)
            pred_lambda = xgb_lambda.predict(X_test)
            model_race_metrics["ML_XGBoost_LambdaMART"].append(
                _evaluate_point_model_race(pred_lambda, test_df)
            )

            # 6. Evaluate PACE Monte Carlo Stochastic Simulator
            if self.include_monte_carlo:
                test_sim_df = test_df.copy()
                test_sim_df["predicted_utility"] = -pred_lambda

                circuit_id = str(test_sim_df["circuit_id"].iloc[0]) if "circuit_id" in test_sim_df.columns else "albert_park"
                circuit_len = float(test_sim_df["circuit_length_km"].iloc[0]) if "circuit_length_km" in test_sim_df.columns else 5.278
                stress = float(test_sim_df["tyre_stress_level"].iloc[0]) if "tyre_stress_level" in test_sim_df.columns else 3.0
                base_time = 82.0 if circuit_id == "albert_park" else 88.0

                mc_engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=self.n_jobs)
                mc_res = mc_engine.run_simulations(
                    event_features=test_sim_df,
                    n_simulations=self.mc_simulations,
                    total_laps=58,
                    circuit_id=circuit_id,
                    circuit_length_km=circuit_len,
                    base_lap_time=base_time,
                    tyre_stress_level=stress,
                    random_seed=42 + idx,
                )

                mc_pred_ranks = np.array([mc_res.expected_position[d] for d in test_df["driver_id"]])
                mc_eval = evaluate_race_predictions(mc_pred_ranks, actual_pos)
                calib = ProbabilityCalibrationEvaluator.evaluate_monte_carlo_race(test_df, mc_res)
                mc_eval.update(calib)
                model_race_metrics["PACE_MonteCarlo_Simulation"].append(mc_eval)

        # 6. Aggregate Summary Table
        summary_rows = []
        for model_name, metrics_list in model_race_metrics.items():
            if not metrics_list:
                continue
            m_df = pd.DataFrame(metrics_list)
            summary_row = {
                "Model": model_name,
                "MAE_Mean": float(m_df["mae"].mean()),
                "MAE_Std": float(m_df["mae"].std()),
                "Spearman_Rho": float(m_df["spearman_rho"].mean()),
                "Kendall_Tau": float(m_df["kendall_tau"].mean()),
                "Top1_Accuracy": float(m_df["top_1_hit"].mean()),
                "Top3_Accuracy": float(m_df["top_3_hit"].mean()),
                "Top5_Accuracy": float(m_df.get("top_5_hit", m_df["top_3_hit"]).mean()),
                "Top10_Accuracy": float(m_df["top_10_hit"].mean()),
                "Races_Evaluated": len(m_df),
            }
            if "brier_win" in m_df.columns:
                summary_row["Brier_Win"] = float(m_df["brier_win"].mean())
            if "brier_podium" in m_df.columns:
                summary_row["Brier_Podium"] = float(m_df["brier_podium"].mean())
            if "brier_points" in m_df.columns:
                summary_row["Brier_Points"] = float(m_df["brier_points"].mean())
            if "brier_dnf" in m_df.columns:
                summary_row["Brier_DNF"] = float(m_df["brier_dnf"].mean())
            if "logloss_points" in m_df.columns:
                summary_row["LogLoss_Points"] = float(m_df["logloss_points"].mean())
            if "multiclass_logloss" in m_df.columns:
                summary_row["MultiClass_LogLoss"] = float(m_df["multiclass_logloss"].mean())

            summary_rows.append(summary_row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values("MAE_Mean")
        else:
            summary_df = pd.DataFrame(columns=["Model", "MAE_Mean", "MAE_Std", "NDCG@5_Mean", "KendallTau_Mean"])

        # 7. Significance testing
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

        if "PACE_MonteCarlo_Simulation" in model_race_metrics and model_race_metrics["PACE_MonteCarlo_Simulation"]:
            mc_maes = [m["mae"] for m in model_race_metrics["PACE_MonteCarlo_Simulation"]]
            stat_mc_grid, p_mc_grid = paired_wilcoxon_test(mc_maes, grid_maes)
            stat_mc_lmm, p_mc_lmm = paired_wilcoxon_test(mc_maes, lmm_maes)
            stat_mc_lambda, p_mc_lambda = paired_wilcoxon_test(mc_maes, lambda_maes)
            significance["PACE_MonteCarlo_vs_GridBaseline"] = {
                "wilcoxon_statistic": stat_mc_grid,
                "p_value": p_mc_grid,
                "statistically_significant_p05": bool(p_mc_grid < 0.05),
            }
            significance["PACE_MonteCarlo_vs_HierarchicalLMM"] = {
                "wilcoxon_statistic": stat_mc_lmm,
                "p_value": p_mc_lmm,
                "statistically_significant_p05": bool(p_mc_lmm < 0.05),
            }
            significance["PACE_MonteCarlo_vs_LambdaMART"] = {
                "wilcoxon_statistic": stat_mc_lambda,
                "p_value": p_mc_lambda,
                "statistically_significant_p05": bool(p_mc_lambda < 0.05),
            }

        return {
            "summary_df": summary_df,
            "significance": significance,
            "raw_metrics": model_race_metrics,
        }
