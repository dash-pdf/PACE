#!/usr/bin/env python3
"""PACE: Predictive Analytics for Competitive Evaluation.

An engineering-driven Formula 1 race prediction system.
"""

import sys
import argparse
import logging
from src.data.pipeline import HistoricalDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PACE")


def main() -> int:
    """Entry point for the PACE command-line interface."""
    parser = argparse.ArgumentParser(
        description="PACE - Predictive Analytics for Competitive Evaluation (Formula 1 Race Prediction)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current project status and architectural components",
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Execute seasonal historical data collection pipeline",
    )
    parser.add_argument(
        "--collect-circuit-history",
        action="store_true",
        help="Execute complete career driver-circuit historical intelligence collection",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Run data quality validation gates on all historical datasets",
    )
    parser.add_argument(
        "--run-models",
        "--backtest",
        dest="run_models",
        action="store_true",
        help="Execute Phase 3 chronological walk-forward model benchmarking across seasons",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Execute systematic Leave-One-Group-Out (LOGO) feature ablation study",
    )
    parser.add_argument(
        "--predict-race",
        action="store_true",
        help="Predict finishing positions and generate attribution explanations for a specific event",
    )
    parser.add_argument(
        "--simulate-race",
        action="store_true",
        help="Execute Phase 4 baseline race simulation for target event",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=58,
        help="Total race laps to simulate (default: 58)",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="H3",
        choices=["H1", "H2", "H3"],
        help="Prediction horizon: H1 (Pre-weekend), H2 (Post-FP2), H3 (Post-Qualifying)",
    )
    parser.add_argument(
        "--target-season",
        type=int,
        default=2026,
        help="Championship season for prediction/inspection (default: 2026)",
    )
    parser.add_argument(
        "--target-round",
        type=int,
        default=5,
        help="Championship round number for prediction/inspection (default: 5)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025, 2026],
        help="Championship seasons to ingest (default: 2023 2024 2025 2026)",
    )
    parser.add_argument(
        "--drivers",
        nargs="+",
        type=str,
        default=None,
        help="Specific driver IDs for circuit history collection (e.g. --drivers hamilton alonso max_verstappen)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing data rather than using cache",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Download sample data for quick verification",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="PACE 0.3.0-dev (Phase 3: Driver & Constructor Performance Modelling)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PACE: Predictive Analytics for Competitive Evaluation")
    print("Formula 1 Race Finishing Position Prediction System")
    print("=" * 70)
    print("Current Phase: Phase 3 (Driver & Constructor Performance Modelling)")
    print("=" * 70)

    if args.status:
        print("\nProject Component Status:")
        print("  - Data Ingestion & Storage: [src/data] (Phase 2 Implemented)")
        print("      * General Season Data (2023-2026)")
        print("      * Driver-Circuit Historical Intelligence (Complete career appearances)")
        print("      * Circuit Technical Characteristics & Regulation Eras")
        print("  - Feature Engineering:      [src/features] (Phase 3 Implemented)")
        print("      * Glicko-2 DPI & Teammate Confounder Controls")
        print("      * CPI Telemetry Decomposition & Latent Stint Degradation")
        print("      * Empirical Bayes Circuit Affinity & Temporal Hypotheses")
        print("      * Leak-Free Feature Pipeline (H1, H2, H3 Horizons)")
        print("  - Performance Modelling:    [src/models] (Phase 3 Implemented)")
        print("      * Tier 1 Baselines: Grid Order, Championship, Rolling Constructor")
        print("      * Tier 2 Statistical: Ridge Regressor, Ordinal Logistic")
        print("      * Tier 3 Hierarchical: Bayesian Linear Mixed-Effects (LMM)")
        print("      * Tier 4 GBDT: XGBoost Pairwise & Listwise LambdaMART")
        print("      * Complete Outcome Space: Calibrated DNF & Plackett-Luce Choice")
        print("  - Model Evaluation:         [src/evaluation] (Phase 3 Implemented)")
        print("      * Chronological Walk-Forward Backtester across 2023-2026")
        print("      * Paired Wilcoxon Signed-Rank Significance Testing")
        print("      * Additive Local Prediction Attribution Explainer")
        print("  - Race Simulation:          [src/simulation] (Pending Phase 4)")
        print("  - Reporting & Export:       [src/reporting] (Pending Phase 5)")
        return 0

    if args.collect_circuit_history:
        pipeline = HistoricalDataPipeline()
        dc_metrics = pipeline.run_driver_circuit_pipeline(
            driver_ids=args.drivers,
            force=args.force,
            sample_only=args.sample_only,
        )
        print("\nDriver-Circuit Historical Intelligence Compilation Completed:")
        for k, v in dc_metrics.items():
            print(f"  - {k}: {v} records")

        print("\nRunning quality validation gates...")
        val_res = pipeline.run_validation()
        print(f"  Validation Status: {'PASSED' if val_res.get('is_valid') else 'FAILED'}")
        return 0 if val_res.get("is_valid") else 1

    if args.collect_data:
        pipeline = HistoricalDataPipeline(seasons=args.seasons)
        metrics = pipeline.run_ingestion(
            force=args.force,
            sample_only=args.sample_only,
            include_driver_circuit=True,
            driver_ids=args.drivers,
        )
        print("\nIngestion Completed Successfully:")
        print(f"  Seasons processed: {metrics['seasons']}")
        print("  Compiled record totals:")
        for category, count in metrics.get("compiled_totals", {}).items():
            print(f"    - {category}: {count} records")

        if "driver_circuit_summary" in metrics:
            print("  Driver-Circuit Intelligence:")
            for k, v in metrics["driver_circuit_summary"].items():
                print(f"    - {k}: {v} records")

        print("\nRunning quality validation gates...")
        validation_res = pipeline.run_validation()
        print(f"  Validation Status: {'PASSED' if validation_res.get('is_valid') else 'FAILED'}")
        return 0 if validation_res.get("is_valid") else 1

    if args.validate_data:
        pipeline = HistoricalDataPipeline(seasons=args.seasons)
        validation_res = pipeline.run_validation()
        print("\nValidation Summary:")
        print(f"  Overall Valid: {validation_res.get('is_valid')}")
        for comp, res in validation_res.get("components", {}).items():
            status = "PASSED" if res.get("is_valid") else "FAILED"
            print(f"  - {comp}: {status} ({res.get('passed_checks')}/{res.get('total_checks')} checks)")
        return 0 if validation_res.get("is_valid") else 1

    if args.run_models:
        import pandas as pd
        from src.features.pipeline import FeaturePipeline
        from src.evaluation.backtest import WalkForwardBacktester

        print("\nLoading historical datasets for Phase 3 Performance Modelling...")
        rr = pd.read_parquet("data/historical/historical_race_results.parquet")
        q = pd.read_parquet("data/historical/historical_qualifying.parquet")
        laps = pd.read_parquet("data/historical/historical_laps.parquet")
        app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
        circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

        pipeline = FeaturePipeline(rr, q, laps, app, circ)
        backtester = WalkForwardBacktester(pipeline, min_train_events=5, horizon=args.horizon)

        print(f"\nExecuting chronological walk-forward backtest (Horizon: {args.horizon})...")
        results = backtester.run_backtest()

        print("\n" + "=" * 95)
        print("PACE PHASE 3: MODEL COMPARISON & BENCHMARK REPORT (WALK-FORWARD EXPANDING WINDOW)")
        print("=" * 95)
        summary_df = results["summary_df"]
        print(summary_df.to_string(index=False))

        print("\nStatistical Significance Tests (Paired Wilcoxon Signed-Rank Test, p < 0.01):")
        for comp_name, sig_info in results["significance"].items():
            sig_flag = "STATISTICALLY SIGNIFICANT" if sig_info["statistically_significant_p01"] else "NOT SIGNIFICANT"
            print(f"  - {comp_name}: p-value = {sig_info['p_value']:.4f} ({sig_flag})")

        # Save benchmark results
        out_path = "data/processed/phase3_benchmark_results.parquet"
        summary_df.to_parquet(out_path)
        print(f"\nBenchmark results persisted to: {out_path}")
        return 0

    if args.run_ablation:
        import pandas as pd
        from src.features.pipeline import FeaturePipeline
        from src.evaluation.ablation import FeatureAblationStudy

        print("\nLoading historical datasets for Systematic Feature Ablation Study...")
        rr = pd.read_parquet("data/historical/historical_race_results.parquet")
        q = pd.read_parquet("data/historical/historical_qualifying.parquet")
        laps = pd.read_parquet("data/historical/historical_laps.parquet")
        app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
        circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

        pipeline = FeaturePipeline(rr, q, laps, app, circ)
        ablation_study = FeatureAblationStudy(pipeline, min_train_events=5, horizon=args.horizon)

        print(f"\nExecuting Leave-One-Group-Out (LOGO) Ablation Testing (Horizon: {args.horizon})...")
        ablation_res = ablation_study.run_logo_ablation()

        print("\n" + "=" * 95)
        print("PACE PHASE 3: SYSTEMATIC FEATURE GROUP ABLATION & OCCAM'S RAZOR PRUNING REPORT")
        print("=" * 95)
        ab_df = ablation_res["ablation_summary"]
        print(ab_df.to_string(index=False))

        out_path = "data/processed/phase3_ablation_results.parquet"
        ab_df.to_parquet(out_path)
        print(f"\nAblation results persisted to: {out_path}")
        return 0

    if args.predict_race:
        import numpy as np
        import pandas as pd
        from src.features.pipeline import FeaturePipeline
        from src.models.ranker import XGBoostLambdaMARTRanker
        from src.models.dnf import CalibratedDNFModel, CompleteOutcomeSpaceEngine
        from src.evaluation.explainability import PredictionExplainer

        print(f"\nGenerating predictions for Season {args.target_season} Round {args.target_round} (Horizon: {args.horizon})...")
        rr = pd.read_parquet("data/historical/historical_race_results.parquet")
        q = pd.read_parquet("data/historical/historical_qualifying.parquet")
        laps = pd.read_parquet("data/historical/historical_laps.parquet")
        app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
        circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

        pipeline = FeaturePipeline(rr, q, laps, app, circ)
        event_df = pipeline.compile_features_for_event(
            season=args.target_season,
            round_num=args.target_round,
            horizon=args.horizon,
        )

        if event_df.empty:
            print(f"Error: No data available for Season {args.target_season} Round {args.target_round}")
            return 1

        feature_cols = [
            "dpi_rating", "dpi_uncertainty", "dpi_volatility", "driver_starts",
            "rookie_learning_factor", "driver_consistency_cv", "cpi_pole_gap_pct",
            "cpi_reliability", "cpi_development_trend", "circuit_affinity_mean",
            "circuit_affinity_se", "circuit_length_km", "overtaking_difficulty",
            "tyre_stress_level", "downforce_level", "altitude_m", "fuel_sensitivity",
            "grid_position", "grid_x_overtaking", "deg_x_stress", "speed_x_length",
        ]

        # Prior training data
        all_events = rr[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values
        prior_events = [
            (int(s), int(r)) for s, r in all_events
            if (s < args.target_season) or (s == args.target_season and r < args.target_round)
        ]

        train_dfs = []
        groups = []
        for s, r in prior_events:
            tr_df = pipeline.compile_features_for_event(s, r, horizon=args.horizon)
            if not tr_df.empty and len(tr_df) >= 10:
                train_dfs.append(tr_df)
                groups.append(len(tr_df))

        if not train_dfs:
            print("Error: Insufficient historical training data.")
            return 1

        full_train = pd.concat(train_dfs, ignore_index=True)
        X_train = full_train[feature_cols].fillna(0.0)
        y_train = full_train["target_rank"].values

        ranker = XGBoostLambdaMARTRanker(n_estimators=80, max_depth=3, learning_rate=0.08)
        ranker.fit(X_train, y_train, groups=groups)

        X_event = event_df[feature_cols].fillna(0.0)
        predicted_ranks = ranker.predict(X_event)
        posterior_draws = ranker.sample_posterior_permutations(X_event, n_draws=1000)

        # Calibrated DNF & Complete outcome space
        dnf_model = CalibratedDNFModel().fit(full_train)
        p_dnf = dnf_model.predict_p_dnf(event_df)
        utilities = ranker.predict_utilities(X_event)
        reliabilities = event_df["cpi_reliability"].values
        outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
            utilities, p_dnf, reliabilities, n_draws=10000
        )

        event_df = event_df.assign(
            predicted_rank=predicted_ranks,
            p_dnf=p_dnf,
            expected_pos=[o["expected_position"] for o in outcomes],
            expected_classified=[o["expected_classified_position"] for o in outcomes],
            rank_p05=[o["rank_p05"] for o in outcomes],
            rank_p95=[o["rank_p95"] for o in outcomes],
            p_win=[o["p_win"] for o in outcomes],
            p_podium=[o["p_podium"] for o in outcomes],
        )

        sorted_df = event_df.sort_values("predicted_rank")

        print("\n" + "=" * 125)
        print(f"PACE RACE PREDICTION & OUTCOME DISTRIBUTION: SEASON {args.target_season} ROUND {args.target_round}")
        print("=" * 125)
        display_cols = [
            "predicted_rank", "driver_id", "constructor_id", "grid_position",
            "rank_p05", "rank_p95", "expected_classified", "expected_pos",
            "p_win", "p_podium", "p_dnf", "finish_position",
        ]
        print(sorted_df[display_cols].to_string(index=False))

        print("\n" + "=" * 105)
        print("DRIVER PREDICTION ATTRIBUTIONS (SHAP-STYLE LOCAL ADDITIVE DECOMPOSITION):")
        print("=" * 105)
        explainer = PredictionExplainer(feature_cols)
        for _, row in sorted_df.head(5).iterrows():
            exp = explainer.explain_driver_prediction(
                driver_id=row["driver_id"],
                driver_features=row,
                predicted_rank=int(row["predicted_rank"]),
                grid_position=float(row["grid_position"]),
                model_name="XGBoost_LambdaMART",
            )
            print(f"\n[P{exp['predicted_rank']}] {exp['driver_id'].upper()} ({row['constructor_id']}):")
            for factor in exp["attributions"]:
                print(f"  * {factor['factor']}: {factor['narrative']}")
        return 0

    if args.simulate_race:
        import pandas as pd
        import numpy as np
        from src.features.pipeline import FeaturePipeline
        from src.models.statistical import RidgePositionRegressor
        from src.simulation.engine import BaselineRaceSimulator

        print(f"\nInitializing Phase 4 Race Simulator: Season {args.target_season} Round {args.target_round} (Horizon: {args.horizon})...")
        rr = pd.read_parquet("data/historical/historical_race_results.parquet")
        q = pd.read_parquet("data/historical/historical_qualifying.parquet")
        laps = pd.read_parquet("data/historical/historical_laps.parquet")
        app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
        circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

        pipeline = FeaturePipeline(rr, q, laps, app, circ)
        event_df = pipeline.compile_features_for_event(
            season=args.target_season,
            round_num=args.target_round,
            horizon=args.horizon,
        )

        if event_df.empty:
            print(f"Error: No data available for Season {args.target_season} Round {args.target_round}")
            return 1

        feature_cols = [
            "dpi_rating", "dpi_uncertainty", "driver_starts", "cpi_pole_gap_pct",
            "cpi_reliability", "circuit_affinity_mean", "circuit_length_km",
            "grid_position", "grid_x_overtaking",
        ]

        all_events = rr[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values
        prior_events = [
            (int(s), int(r)) for s, r in all_events
            if (s < args.target_season) or (s == args.target_season and r < args.target_round)
        ]

        train_dfs = [pipeline.compile_features_for_event(s, r, horizon=args.horizon) for s, r in prior_events[:15]]
        train_dfs = [df for df in train_dfs if not df.empty]
        full_train = pd.concat(train_dfs, ignore_index=True)

        ridge = RidgePositionRegressor().fit(full_train[feature_cols].fillna(0.0), full_train["target_rank"].values)
        raw_pred = ridge.model.predict(ridge.scaler.transform(event_df[feature_cols].fillna(0.0)))
        event_df["predicted_utility"] = -raw_pred

        circuit_id = event_df["circuit_id"].iloc[0] if "circuit_id" in event_df.columns else "albert_park"
        circuit_length = float(event_df["circuit_length_km"].iloc[0]) if "circuit_length_km" in event_df.columns else 5.278
        base_lap_time = 82.0 if circuit_id == "albert_park" else 88.0

        print(f"Simulating {args.laps} Laps at {circuit_id.replace('_', ' ').title()}...")
        simulator = BaselineRaceSimulator(utility_scaling_factor=0.05)
        sim_result = simulator.simulate_race(
            event_features=event_df,
            total_laps=args.laps,
            circuit_id=circuit_id,
            circuit_length_km=circuit_length,
            base_lap_time=base_lap_time,
            random_seed=42,
        )

        df_class = sim_result.final_classification

        print("\n" + "=" * 110)
        print(f"PACE PHASE 4: RACE SIMULATION RESULTS — SEASON {args.target_season} ROUND {args.target_round}")
        print("=" * 110)
        display_cols = [
            "finish_position", "driver_id", "constructor_id", "grid_position",
            "pit_stops", "compounds", "laps_completed", "elapsed_time", "gap_to_leader", "status",
        ]
        print(df_class[display_cols].to_string(index=False))

        print("\n" + "=" * 110)
        print("SIMULATION SUMMARY METRICS:")
        print("=" * 110)
        for k, v in sim_result.summary_metrics.items():
            print(f"  - {k}: {v}")

        # Top 3 fastest laps
        lap_chart = sim_result.lap_chart
        fastest_laps = lap_chart.sort_values("lap_time").head(3)
        print("\nFastest Laps Recorded:")
        for _, fl in fastest_laps.iterrows():
            print(f"  * Lap {int(fl['lap'])}: {fl['driver_id']} ({fl['constructor_id']}) — {fl['lap_time']:.3f}s")
        return 0

    # Default action if no flags provided
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
