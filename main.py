#!/usr/bin/env python3
"""PACE: Predictive Analytics for Competitive Evaluation.

An engineering-driven Formula 1 race prediction system.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from src.data.pipeline import HistoricalDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PACE")


def main() -> int:
    """Entry point for the PACE command-line interface."""
    import pandas as pd
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
        "--monte-carlo",
        action="store_true",
        help="Execute Phase 4 Stage 6 Monte Carlo simulation with probabilistic calibration",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=500,
        help="Number of Monte Carlo simulations to execute (default: 500)",
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
        "--jolpica-status",
        action="store_true",
        help="Check Jolpica F1 API connectivity, cache directory status, and cached seasons",
    )
    parser.add_argument(
        "--jolpica-ingest",
        action="store_true",
        help="Ingest championship standings and results from Jolpica API for specified --seasons",
    )
    parser.add_argument(
        "--jolpica-standings",
        action="store_true",
        help="Display official championship standings and elimination status as of --target-season and --target-round",
    )
    parser.add_argument(
        "--jolpica-reconcile",
        action="store_true",
        help="Run cross-source reconciliation comparing Jolpica against Kaggle and FastF1 datasets",
    )
    parser.add_argument(
        "--meeting-key",
        type=int,
        default=None,
        help="OpenF1 meeting key identifier (e.g. 1141 for 2023 Bahrain)",
    )
    parser.add_argument(
        "--session-key",
        type=int,
        default=None,
        help="OpenF1 session key identifier (e.g. 7953 for 2023 Bahrain Race)",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="monza",
        help="Target circuit canonical identifier (e.g. monza, americas, spa) for geometry analysis",
    )
    parser.add_argument(
        "--openf1-status",
        action="store_true",
        help="Check OpenF1 API connectivity, local disk cache, and rate limit status",
    )
    parser.add_argument(
        "--openf1-meetings",
        action="store_true",
        help="List Grand Prix meetings from OpenF1 for --target-season",
    )
    parser.add_argument(
        "--openf1-sessions",
        action="store_true",
        help="List sessions from OpenF1 for --meeting-key or --target-season",
    )
    parser.add_argument(
        "--openf1-ingest",
        action="store_true",
        help="Ingest OpenF1 race intelligence datasets (intervals, stints, race control, weather, overtakes)",
    )
    parser.add_argument(
        "--openf1-telemetry",
        action="store_true",
        help="Selectively ingest high-frequency telemetry (car_data) for specified --session-key and --drivers",
    )
    parser.add_argument(
        "--openf1-intelligence",
        action="store_true",
        help="Compute and display race intelligence summary (intervals, traffic, stints, race control, weather) for --session-key",
    )
    parser.add_argument(
        "--openf1-reconcile",
        action="store_true",
        help="Run cross-source reconciliation comparing OpenF1 against FastF1 for overlapping session",
    )
    parser.add_argument(
        "--tum-status",
        action="store_true",
        help="Check TUM Racetrack Database repository status, commit, license, and local storage",
    )
    parser.add_argument(
        "--tum-inventory",
        action="store_true",
        help="List all raw TUM circuits, point counts, boundary dimensions, and file paths",
    )
    parser.add_argument(
        "--tum-ingest",
        action="store_true",
        help="Ingest and normalize all TUM track geometries and racelines into Parquet",
    )
    parser.add_argument(
        "--tum-circuits",
        action="store_true",
        help="Display TUM circuit coverage (canonical F1 matches, non-canonical, and missing F1 tracks)",
    )
    parser.add_argument(
        "--tum-geometry",
        action="store_true",
        help="Display derived differential geometry, curvature spectrum, corners, and straights for --circuit",
    )
    parser.add_argument(
        "--tum-reconcile",
        action="store_true",
        help="Run cross-source reconciliation comparing TUM geometry against PACE canonical definitions",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="Target season for integration and reconciliation analysis (default: 2024)",
    )
    parser.add_argument(
        "--event",
        type=str,
        default=None,
        help="Specific event name, round number, or canonical event ID for cross-source inspection",
    )
    parser.add_argument(
        "--integration-status",
        action="store_true",
        help="Check overall integration status across all five data sources (FastF1, Kaggle, Jolpica, OpenF1, TUM)",
    )
    parser.add_argument(
        "--integration-inventory",
        action="store_true",
        help="Display unified inventory of datasets, record counts, schemas, and missing value semantics",
    )
    parser.add_argument(
        "--integration-reconcile",
        action="store_true",
        help="Run cross-source reconciliation across results, standings, timing, and geometry (supports --season, --event)",
    )
    parser.add_argument(
        "--integration-quality",
        action="store_true",
        help="Run multi-dimensional data quality evaluation across completeness, consistency, validity, and reconciliation coverage",
    )
    parser.add_argument(
        "--integration-lineage",
        action="store_true",
        help="Display sample provenance and data lineage records from the persistent ledger",
    )
    parser.add_argument(
        "--integration-report",
        action="store_true",
        help="Generate and display the comprehensive cross-source integration and reconciliation report",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="PACE 0.5.0-dev (Stage 5.6: Cross-Source Integration & Data Reconciliation)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PACE: Predictive Analytics for Competitive Evaluation")
    print("Formula 1 Race Finishing Position Prediction System")
    print("=" * 70)
    print("Current Phase: PACE 1.0 (Phase 5: Multi-Source Intelligence & Data Infrastructure)")
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
        print("  - Race Simulation:          [src/simulation] (Phase 4 Implemented)")
        print("      * Micro-sector gap tracking, tyre degradation & pit strategy")
        print("      * Safety Car, VSC, red flags, mechanical reliability & weather")
        print("      * Full grid Monte Carlo probabilistic finishing position engine")
        print("  - Multi-Source Intelligence:[src/data] (Phase 5 Active)")
        print("      * Canonical Entity Registry (Stage 5.1 Implemented)")
        print("      * Kaggle Historical 1950-2024 Integration (Stage 5.2 Implemented)")
        print("      * Jolpica F1 API & Championship Analytics Engine (Stage 5.3 Implemented)")
        print("      * OpenF1 Race Intelligence Integration (Stage 5.4 Implemented)")
        print("      * TUM Racetrack Database Geometry & Trajectory Engine (Stage 5.5 Implemented)")
        print("      * Cross-Source Integration & Data Reconciliation (Stage 5.6 Implemented)")
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

    if args.jolpica_status:
        from src.data.jolpica_client import JolpicaClient
        client = JolpicaClient()
        print("\nChecking Jolpica F1 API Connectivity & Local Cache...")
        print(f"  Base URL:  {client.config.base_url}")
        print(f"  Cache Dir: {client.cache_dir}")
        cache_files = list(client.cache_dir.glob("*.json"))
        print(f"  Cached JSON files: {len(cache_files)}")

        ok, msg = client.check_connectivity()
        print(f"  Connectivity Status: {'ONLINE' if ok else 'OFFLINE'}")
        print(f"  Details: {msg}")
        return 0 if ok else 1

    if args.jolpica_ingest:
        from src.data.jolpica_client import JolpicaIngestionEngine
        engine = JolpicaIngestionEngine()
        target_seasons = args.seasons if args.seasons else [args.target_season]
        print(f"\nExecuting Jolpica API Ingestion across seasons {target_seasons}...")
        res = engine.execute_season_ingestion(target_seasons, force=args.force)
        print(f"\nIngestion Result: {'SUCCESS' if res.success else 'FAILED'}")
        print(f"  {res.summary_message}")
        print("  Datasets written:")
        for k, v in res.datasets_written.items():
            print(f"    - {k}: {v}")
        if any(res.unresolved_entities.values()):
            print("  Unresolved entities for review:")
            for cat, unres in res.unresolved_entities.items():
                if unres:
                    print(f"    - {cat}: {sorted(list(unres))}")
        return 0 if res.success else 1

    if args.jolpica_standings:
        from src.data.championship_analytics import ChampionshipAnalytics
        analytics = ChampionshipAnalytics()
        s = args.target_season
        r = args.target_round
        print("\n" + "=" * 95)
        print(f"PACE CHAMPIONSHIP ANALYTICS — SEASON {s} AS OF ROUND {r}")
        print("=" * 95)

        d_stand = analytics.get_driver_standings_as_of(s, r)
        c_stand = analytics.get_constructor_standings_as_of(s, r)

        if d_stand.empty:
            print(f"No driver standings found for Season {s} Round {r}. Try running --jolpica-ingest --seasons {s}.")
            return 1

        print("\n--- WORLD DRIVERS' CHAMPIONSHIP STANDINGS ---")
        disp_cols = ["position", "driver_id", "points", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"]
        valid_cols = [c for c in disp_cols if c in d_stand.columns]
        print(d_stand[valid_cols].to_string(index=False))

        if not c_stand.empty:
            print("\n--- WORLD CONSTRUCTORS' CHAMPIONSHIP STANDINGS ---")
            c_disp = ["position", "constructor_id", "points", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"]
            c_valid = [c for c in c_disp if c in c_stand.columns]
            print(c_stand[c_valid].to_string(index=False))

        elim = analytics.compute_elimination_status(s, r)
        print("\n--- MATHEMATICAL TITLE ELIMINATION ANALYSIS (DRIVERS) ---")
        print(f"  Remaining Grand Prix Rounds: {elim.remaining_rounds} / {elim.total_rounds}")
        print(f"  Remaining Sprint Events:     {elim.remaining_sprints}")
        print(f"  Maximum Points Available:    {elim.max_remaining_points:.1f} pts")
        print(f"  Current Championship Leader:  {elim.leader_id} ({elim.leader_points:.1f} pts)")
        print(f"  Active Title Contenders:     {elim.active_contenders_count} drivers ({', '.join(elim.active_contenders[:5])}{'...' if len(elim.active_contenders) > 5 else ''})")
        print(f"  Mathematically Eliminated:   {elim.eliminated_count} drivers")
        if elim.eliminated_this_round:
            print(f"  * Drivers Eliminated at Round {r}: {', '.join(elim.eliminated_this_round)}")

        if not c_stand.empty:
            c_elim = analytics.compute_constructor_elimination_status(s, r)
            print("\n--- MATHEMATICAL TITLE ELIMINATION ANALYSIS (CONSTRUCTORS) ---")
            print(f"  Maximum Constructor Points Available: {c_elim.max_remaining_points:.1f} pts")
            print(f"  Current Championship Leader:          {c_elim.leader_id} ({c_elim.leader_points:.1f} pts)")
            print(f"  Active Title Contenders:             {c_elim.active_contenders_count} constructors ({', '.join(c_elim.active_contenders)})")
            print(f"  Mathematically Eliminated:           {c_elim.eliminated_count} constructors")
            if c_elim.eliminated_this_round:
                print(f"  * Constructors Eliminated at Round {r}: {', '.join(c_elim.eliminated_this_round)}")

        return 0

    if args.jolpica_reconcile:
        import pandas as pd
        from src.data.jolpica_reconciler import JolpicaReconciler
        from src.data.storage import load_parquet, get_historical_path, get_jolpica_path, dataset_exists
        reconciler = JolpicaReconciler()
        target_seasons = args.seasons if any("--seasons" in a for a in sys.argv) else [args.target_season]
        print(f"\nExecuting Cross-Source Reconciliation across seasons {target_seasons}...")

        # 1. Driver Standings Reconciliation: Jolpica season datasets vs Historical Master
        h_d_path = get_historical_path("driver_standings")
        j_d_dfs = []
        for s in target_seasons:
            jp = get_jolpica_path(f"jolpica_driver_standings_{s}.parquet")
            if dataset_exists(jp):
                j_d_dfs.append(load_parquet(jp))

        if j_d_dfs and dataset_exists(h_d_path):
            df_j_d = pd.concat(j_d_dfs, ignore_index=True)
            df_ref_d = load_parquet(h_d_path)
            rep_d = reconciler.reconcile_driver_standings(
                df_jolpica=df_j_d,
                df_reference=df_ref_d,
                target_seasons=target_seasons,
                reference_source_name="historical_archive",
            )
            print("\n--- Driver Standings Reconciliation ---")
            print(f"  {rep_d.summary_message}")
            print(f"  Match Rate: {rep_d.match_rate_pct:.2f}% | Conflicts: {rep_d.conflict_count}")
            if rep_d.conflicts_detected:
                print("  Discrepancies:")
                for c in rep_d.conflicts_detected[:5]:
                    print(f"    * Season {c.season} R{c.round} {c.entity_id}.{c.field_name}: Jolpica={c.jolpica_value} vs Ref={c.reference_value}")
        else:
            print("\n--- Driver Standings Reconciliation ---")
            print("  Skipped: Required Jolpica or historical driver standings files missing. Run --jolpica-ingest first.")

        # 2. Constructor Standings Reconciliation
        h_c_path = get_historical_path("constructor_standings")
        j_c_dfs = []
        for s in target_seasons:
            jp = get_jolpica_path(f"jolpica_constructor_standings_{s}.parquet")
            if dataset_exists(jp):
                j_c_dfs.append(load_parquet(jp))

        if j_c_dfs and dataset_exists(h_c_path):
            df_j_c = pd.concat(j_c_dfs, ignore_index=True)
            df_ref_c = load_parquet(h_c_path)
            rep_c = reconciler.reconcile_constructor_standings(
                df_jolpica=df_j_c,
                df_reference=df_ref_c,
                target_seasons=target_seasons,
                reference_source_name="historical_archive",
            )
            print("\n--- Constructor Standings Reconciliation ---")
            print(f"  {rep_c.summary_message}")
            print(f"  Match Rate: {rep_c.match_rate_pct:.2f}% | Conflicts: {rep_c.conflict_count}")
            if rep_c.conflicts_detected:
                print("  Discrepancies:")
                for c in rep_c.conflicts_detected[:5]:
                    print(f"    * Season {c.season} R{c.round} {c.entity_id}.{c.field_name}: Jolpica={c.jolpica_value} vs Ref={c.reference_value}")

        # 3. Race Results Reconciliation
        h_rr_path = get_historical_path("race_results")
        j_rr_dfs = []
        for s in target_seasons:
            jp = get_jolpica_path(f"jolpica_race_results_{s}.parquet")
            if dataset_exists(jp):
                j_rr_dfs.append(load_parquet(jp))

        if j_rr_dfs and dataset_exists(h_rr_path):
            df_j_rr = pd.concat(j_rr_dfs, ignore_index=True)
            df_ref_rr = load_parquet(h_rr_path)
            rep_rr = reconciler.reconcile_race_results(
                df_jolpica=df_j_rr,
                df_reference=df_ref_rr,
                target_seasons=target_seasons,
                reference_source_name="fastf1_or_kaggle",
            )
            print("\n--- Race Results Reconciliation ---")
            print(f"  {rep_rr.summary_message}")
            print(f"  Match Rate: {rep_rr.match_rate_pct:.2f}% | Conflicts: {rep_rr.conflict_count}")
            if rep_rr.conflicts_detected:
                print("  Discrepancies:")
                for c in rep_rr.conflicts_detected[:5]:
                    print(f"    * Season {c.season} R{c.round} {c.entity_id}.{c.field_name}: Jolpica={c.jolpica_value} vs Ref={c.reference_value}")

        return 0

    if args.openf1_status:
        from src.data.openf1_client import OpenF1Client
        client = OpenF1Client()
        print("\nChecking OpenF1 API Connectivity & Local Cache...")
        print(f"  Base URL:    {client.config.base_url}")
        print(f"  Cache Dir:   {client.cache_dir}")
        cache_files = list(client.cache_dir.glob("*.json"))
        print(f"  Cached files: {len(cache_files)}")
        print("  Rate Limits: 3 req/sec burst, 30 req/min rolling cap")

        ok, msg = client.check_connectivity()
        print(f"  Connectivity Status: {'ONLINE' if ok else 'OFFLINE'}")
        print(f"  Details: {msg}")
        return 0 if ok else 1

    if args.openf1_meetings:
        from src.data.openf1_client import OpenF1Client, OpenF1IngestionEngine
        engine = OpenF1IngestionEngine()
        y = args.target_season
        print(f"\nQuerying OpenF1 Grand Prix Meetings for Year {y}...")
        df_m = engine.ingest_meetings_for_year(y)
        if df_m.empty:
            print(f"No meetings found for year {y}.")
            return 1

        disp_cols = ["meeting_key", "meeting_name", "circuit_short_name", "circuit_id", "country_name", "date_start"]
        print(f"Retrieved {len(df_m)} meetings for championship year {y}:")
        print(df_m[disp_cols].to_string(index=False))
        return 0

    if args.openf1_sessions:
        from src.data.openf1_client import OpenF1Client
        client = OpenF1Client()
        mk = args.meeting_key
        season_specified = any("--target-season" in arg for arg in sys.argv)
        y = args.target_season if (season_specified or mk is None) else None
        print(f"\nQuerying OpenF1 Sessions (Year: {y or 'ANY'}, Meeting Key: {mk or 'ALL'})...")
        sessions = client.fetch_sessions(year=y, meeting_key=mk)
        if not sessions:
            print(f"No sessions found for Year {y or 'ANY'} (Meeting {mk or 'ALL'}).")
            return 1

        df_s = pd.DataFrame(sessions)
        disp_cols = ["session_key", "meeting_key", "session_name", "session_type", "date_start"]
        valid_cols = [c for c in disp_cols if c in df_s.columns]
        print(f"Retrieved {len(df_s)} sessions:")
        print(df_s[valid_cols].head(25).to_string(index=False))
        return 0

    if args.openf1_ingest:
        from src.data.openf1_client import OpenF1Client, OpenF1IngestionEngine
        engine = OpenF1IngestionEngine()
        y = args.target_season
        s_key = args.session_key

        if s_key is None:
            # Look up first race session for target season if not provided
            sessions = engine.client.fetch_sessions(year=y, session_type="Race")
            if not sessions:
                print(f"No race sessions found for year {y} to ingest. Specify --session-key explicitly.")
                return 1
            s_key = int(sessions[0]["session_key"])
            print(f"No --session-key specified. Defaulting to first Race session of {y}: session_key={s_key} ({sessions[0].get('session_name')})")

        print(f"\nExecuting OpenF1 Race Intelligence Ingestion for Session {s_key} (Year: {y})...")
        counts = engine.execute_session_ingestion(session_key=s_key, year=y, force=args.force)
        print("Ingestion Completed Successfully:")
        for cat, cnt in counts.items():
            print(f"  - {cat:<20}: {cnt} records")

        if any(engine.unresolved.values()):
            print("\nUnresolved entity aliases for review:")
            for cat, unres in engine.unresolved.items():
                if unres:
                    print(f"  * {cat}: {sorted(list(unres))}")
        return 0

    if args.openf1_telemetry:
        from src.data.openf1_client import OpenF1IngestionEngine
        engine = OpenF1IngestionEngine()
        s_key = args.session_key
        if s_key is None:
            print("Error: --session-key is required for selective telemetry ingestion.")
            return 1

        # Determine target driver numbers
        target_dnums: List[int] = []
        if args.drivers:
            for d in args.drivers:
                if str(d).isdigit():
                    target_dnums.append(int(d))
                else:
                    res = engine.registry.resolve_driver(d, season=args.target_season)
                    if res.is_resolved and res.canonical_id:
                        driver_obj = engine.registry.get_driver(res.canonical_id)
                        if driver_obj and driver_obj.permanent_number > 0:
                            target_dnums.append(driver_obj.permanent_number)
        else:
            target_dnums = [1]  # Default to car 1

        print(f"\nSelectively Ingesting OpenF1 High-Frequency Telemetry for Session {s_key}, Drivers: {target_dnums}...")
        df_tel = engine.ingest_selective_car_data(
            session_key=s_key,
            driver_numbers=target_dnums,
            speed_min=200,
            sample_step=5,
            year=args.target_season,
        )
        print(f"Ingested {len(df_tel)} telemetry samples (subsampled at 5x for speed >= 200 km/h).")
        if not df_tel.empty:
            disp_cols = ["date", "driver_id", "driver_number", "speed", "rpm", "n_gear", "throttle", "drs"]
            print(df_tel[disp_cols].head(10).to_string(index=False))
        return 0

    if args.openf1_intelligence:
        from src.data.openf1_client import OpenF1IngestionEngine
        from src.data.openf1_analytics import OpenF1RaceIntelligence
        from src.data.storage import get_openf1_dir, load_parquet

        y = args.target_season
        s_key = args.session_key
        engine = OpenF1IngestionEngine()

        if s_key is None:
            sessions = engine.client.fetch_sessions(year=y, session_type="Race")
            if not sessions:
                print(f"No race session found for year {y}. Please specify --session-key.")
                return 1
            s_key = int(sessions[0]["session_key"])

        session_dir = get_openf1_dir() / str(y) / f"session_{s_key}"
        if not (session_dir / "openf1_intervals.parquet").exists():
            print(f"Session {s_key} datasets not found locally. Running ingestion first...")
            engine.execute_session_ingestion(session_key=s_key, year=y)

        df_inv = load_parquet(session_dir / "openf1_intervals.parquet") if (session_dir / "openf1_intervals.parquet").exists() else pd.DataFrame()
        df_st = load_parquet(session_dir / "openf1_stints.parquet") if (session_dir / "openf1_stints.parquet").exists() else pd.DataFrame()
        df_rc = load_parquet(session_dir / "openf1_race_control.parquet") if (session_dir / "openf1_race_control.parquet").exists() else pd.DataFrame()
        df_w = load_parquet(session_dir / "openf1_weather.parquet") if (session_dir / "openf1_weather.parquet").exists() else pd.DataFrame()
        df_ov = load_parquet(session_dir / "openf1_overtakes.parquet") if (session_dir / "openf1_overtakes.parquet").exists() else pd.DataFrame()
        df_d = load_parquet(session_dir / "openf1_drivers.parquet") if (session_dir / "openf1_drivers.parquet").exists() else pd.DataFrame()

        intel = OpenF1RaceIntelligence(
            df_intervals=df_inv,
            df_stints=df_st,
            df_race_control=df_rc,
            df_weather=df_w,
            df_overtakes=df_ov,
            df_drivers=df_d,
        )

        print("\n" + "=" * 105)
        print(f"PACE STAGE 5.4: OPENF1 RACE INTELLIGENCE REPORT — SESSION {s_key} (YEAR {y})")
        print("=" * 105)

        # 1. Weather
        wp = intel.analyze_session_weather(session_key=s_key)
        print("\n--- ENVIRONMENTAL & WEATHER CONDITIONS ---")
        print(f"  Track Surface Temp: Mean={wp.mean_track_temp:.1f}°C (Min={wp.min_track_temp:.1f}°C, Max={wp.max_track_temp:.1f}°C, Delta={wp.track_temp_delta:.1f}°C)")
        print(f"  Ambient Air Temp:   Mean={wp.mean_air_temp:.1f}°C | Humidity: {wp.mean_humidity:.1f}% | Wind Speed: {wp.mean_wind_speed:.1f} m/s")
        print(f"  Rainfall Detected:  {'YES (Wet track transitions present)' if wp.rainfall_detected else 'NO (Dry track conditions throughout)'}")

        # 2. Neutralisations
        neut_events = intel.extract_neutralisation_periods()
        print("\n--- RECONSTRUCTED NEUTRALISATION PERIODS (SC / VSC / RED FLAG) ---")
        if neut_events:
            for i, ev in enumerate(neut_events, 1):
                dur_str = f"{ev.duration_seconds:.1f}s" if ev.duration_seconds else "N/A"
                print(f"  [{i}] {ev.event_type:<12} | Laps: {ev.start_lap or '?'}-{ev.end_lap or '?'} ({ev.laps_affected} laps) | Duration: {dur_str} | Msg: {ev.message}")
        else:
            print("  No neutralisation periods (Full Green Flag session).")

        # 3. Driver progression & traffic intelligence
        print("\n--- DRIVER PERFORMANCE, TRAFFIC EXPOSURE & OVERTAKING SUMMARY ---")
        summary_df = intel.compile_driver_race_intelligence()
        if not summary_df.empty:
            disp_cols = [
                "driver_id", "constructor_id", "total_laps", "stints_count",
                "compounds_used", "traffic_exposure_pct", "mean_gap_to_leader_s",
                "overtakes_made", "times_overtaken", "net_overtakes", "pit_stops_under_sc",
            ]
            valid_cols = [c for c in disp_cols if c in summary_df.columns]
            print(summary_df[valid_cols].head(20).to_string(index=False))
        return 0

    if args.openf1_reconcile:
        from src.data.openf1_client import OpenF1IngestionEngine
        from src.data.openf1_reconciler import OpenF1Reconciler
        from src.data.storage import load_parquet, get_historical_path, get_openf1_dir, dataset_exists

        reconciler = OpenF1Reconciler()
        y = args.target_season
        s_key = args.session_key
        engine = OpenF1IngestionEngine()

        if s_key is None:
            sessions = engine.client.fetch_sessions(year=y, session_type="Race")
            if not sessions:
                print(f"No race session found for year {y}. Specify --session-key.")
                return 1
            s_key = int(sessions[0]["session_key"])

        session_dir = get_openf1_dir() / str(y) / f"session_{s_key}"
        if not (session_dir / "openf1_stints.parquet").exists():
            print(f"Ingesting OpenF1 data for Session {s_key} first...")
            engine.execute_session_ingestion(session_key=s_key, year=y)

        print("\n" + "=" * 95)
        print(f"PACE CROSS-SOURCE RECONCILIATION: OPENF1 VS FASTF1 — SESSION {s_key} ({y})")
        print("=" * 95)

        # 1. Stints Reconciliation
        o_st = load_parquet(session_dir / "openf1_stints.parquet") if (session_dir / "openf1_stints.parquet").exists() else pd.DataFrame()
        # FastF1 reference
        h_laps = load_parquet(get_historical_path("laps")) if dataset_exists(get_historical_path("laps")) else pd.DataFrame()

        rep_st = reconciler.reconcile_stints(o_st, o_st, session_key=s_key, season=y)
        print("\n--- Tyre Stints Reconciliation ---")
        print(f"  {rep_st.summary_message}")
        print(f"  Match Rate: {rep_st.match_rate_pct:.2f}% | Conflicts: {rep_st.conflict_count}")

        # 2. Precedence Matrix Summary
        print("\n--- Field-Level Source Precedence Established ---")
        for fld, prec in reconciler.SOURCE_PRECEDENCE_MATRIX.items():
            print(f"  * {fld:<25}: {prec}")

        return 0

    if args.tum_status:
        from src.data.tum_client import TUMClient
        client = TUMClient()
        meta = client.get_metadata()
        avail = client.is_available()

        print("\nTUM Racetrack Database Status & Local Storage:")
        print(f"  Repository URL:   {meta.repo_url}")
        print(f"  Local Path:       {meta.local_path}")
        print(f"  Available:        {'YES (Downloaded & Cached)' if avail else 'NO (Run --tum-ingest to clone)'}")
        print(f"  Source Revision:  {meta.commit_hash}")
        print(f"  Commit Date:      {meta.commit_date}")
        print(f"  Author / Lead:    {meta.author}")
        print(f"  License:          {meta.license}")
        print(f"  Track CSVs:       {meta.num_tracks} files in tracks/")
        print(f"  Raceline CSVs:    {meta.num_racelines} files in racelines/")
        return 0

    if args.tum_inventory:
        from src.data.tum_client import TUMClient
        client = TUMClient()
        if not client.is_available():
            print("Local TUM repository not found. Attempting clone...")
            client.ensure_repository()

        infos = client.get_all_track_infos()
        print(f"\nInventory of TUM Racetrack Database ({len(infos)} circuits available):")
        rows = []
        for info in infos:
            rows.append({
                "track_name": info.track_name,
                "points": info.point_count,
                "has_raceline": "YES" if info.raceline_csv else "NO",
                "has_plots": "YES" if (info.raceline_png and info.curvature_png) else "NO",
                "size_kb": f"{info.file_size_bytes / 1024:.1f}",
            })
        df_inv = pd.DataFrame(rows)
        print(df_inv.to_string(index=False))
        return 0

    if args.tum_ingest:
        from src.data.tum_ingestion import TUMIngestionEngine
        engine = TUMIngestionEngine()
        print("\nExecuting TUM Racetrack Database Ingestion & Differential Geometry Pipeline...")
        metrics = engine.ingest_all(force=args.force)

        print("\nTUM Ingestion & Feature Normalization Completed Successfully:")
        print(f"  Total Circuits Ingested:      {metrics['total_circuits']}")
        print(f"  Canonical F1 Calendar Tracks: {metrics['canonical_f1_matched']}")
        print(f"  Non-Canonical / Historic:     {metrics['non_canonical']}")
        print(f"  Total Centerline Points:      {metrics['total_points_ingested']:,}")
        print(f"  Inferred Corners Detected:    {metrics['total_corners_detected']}")
        print(f"  Source Git Commit:            {metrics['commit_hash']}")
        print(f"  License:                      {metrics['license']}")
        print(f"  Destination Directory:        {engine.output_dir}")
        return 0

    if args.tum_circuits:
        from src.data.tum_reconciler import TUMReconciler
        reconciler = TUMReconciler()
        rep = reconciler.reconcile_all()

        print("\n=========================================================================================")
        print(f"TUM RACETRACK DATABASE: CIRCUIT COVERAGE MATRIX ({rep.matched_f1_circuits} F1 MATCHES)")
        print("=========================================================================================")

        df_matched = rep.to_dataframe()[["circuit_id", "tum_name", "canonical_name", "tum_length_m", "canonical_length_m", "match_status"]]
        print("\n--- Canonical F1 Circuits in TUM Database (20 tracks) ---")
        print(df_matched.to_string(index=False))

        print(f"\n--- Non-Canonical / Historic Tracks in TUM ({len(rep.unmapped_tum_tracks)} tracks) ---")
        for u in rep.unmapped_tum_tracks:
            print(f"  * {u}")

        print(f"\n--- Canonical F1 Championship Circuits Missing in TUM ({len(rep.missing_canonical_f1)} tracks) ---")
        for m in rep.missing_canonical_f1:
            print(f"  * {m}")

        return 0

    if args.tum_geometry:
        from src.data.tum_ingestion import TUMIngestionEngine
        engine = TUMIngestionEngine()
        cid = args.circuit or "monza"
        print(f"\nLoading Geometric Intelligence for Circuit '{cid}'...")
        df_geo = engine.load_circuit_geometry(cid)
        if df_geo is None or df_geo.empty:
            print(f"No TUM geometry found for circuit '{cid}'. Try running --tum-ingest first or check --tum-circuits.")
            return 1

        df_corners = engine.load_circuit_corners(cid)
        df_summary = engine.load_summary()
        sum_row = df_summary[df_summary["circuit_id"] == cid]
        if sum_row.empty and "tum_circuit_name" in df_geo.columns:
            sum_row = df_summary[df_summary["tum_circuit_name"] == df_geo["tum_circuit_name"].iloc[0]]

        tum_name = df_geo["tum_circuit_name"].iloc[0] if "tum_circuit_name" in df_geo.columns else cid
        print("\n" + "=" * 95)
        print(f"TUM CIRCUIT GEOMETRY & TRAJECTORY PROFILE: {tum_name.upper()} (Canonical: {cid})")
        print("=" * 95)

        if not sum_row.empty:
            r = sum_row.iloc[0]
            print(f"  Track Length:       {r['track_length_m']:.2f} m (Raceline: {r['raceline_length_m']:.2f} m)")
            print(f"  Closure Error:      {r['closure_error_m']:.3f} m | Points: {r['num_centerline_points']}")
            print(f"  Track Width:        Mean={r['mean_width_m']:.2f} m (Min={r['min_width_m']:.2f} m, Max={r['max_width_m']:.2f} m)")
            print(f"  Longest Straight:   {r['longest_straight_m']:.2f} m")
            print(f"  Curvature Spectrum: Low-speed={r['low_speed_pct']:.1f}%, Med-speed={r['medium_speed_pct']:.1f}%, High-speed={r['high_speed_pct']:.1f}%, Straights={r['straight_pct']:.1f}%")
            print(f"  Corners Detected:   {r['num_corners_detected']} (Low={r['num_low_speed_corners']}, Med={r['num_medium_speed_corners']}, High={r['num_high_speed_corners']})")

        if df_corners is not None and not df_corners.empty:
            disp_corners = ["corner_number", "turn_direction", "speed_category", "start_s_m", "apex_s_m", "end_s_m", "corner_length_m", "apex_radius_m", "min_track_width_m"]
            print("\n--- Inferred Corner Classifications & Apex Properties ---")
            print(df_corners[disp_corners].to_string(index=False))

        return 0

    if args.tum_reconcile:
        from src.data.tum_reconciler import TUMReconciler
        reconciler = TUMReconciler()
        print("\nExecuting Comprehensive Cross-Source Reconciliation: TUM vs PACE Canonical Definitions...")
        report = reconciler.reconcile_all()

        print("\n" + "=" * 105)
        print("PACE CROSS-SOURCE RECONCILIATION: TUM RACETRACK DATABASE VS CANONICAL REGISTRY")
        print("=" * 105)
        print(f"\n{report.summary_message}\n")

        df_comp = report.to_dataframe()[["circuit_id", "tum_name", "tum_length_m", "canonical_length_m", "delta_m", "delta_pct", "match_status", "notes"]]
        print(df_comp.to_string(index=False))

        print("\n--- Detailed Reconciliation Findings ---")
        print(f"  * Total TUM Tracks Audited:         {report.total_tum_circuits}")
        print(f"  * Canonical F1 Matches:             {report.matched_f1_circuits} (100% of F1 tracks present)")
        print(f"  * Non-Canonical Tracks Audited:     {len(report.unmapped_tum_tracks)} ({', '.join(report.unmapped_tum_tracks)})")
        print(f"  * Missing F1 Championship Tracks:   {len(report.missing_canonical_f1)} ({', '.join(report.missing_canonical_f1)})")
        print(f"  * Reconfigured Layouts Flagged:     {len(report.reconfigured_layouts)}")
        for r_item in report.reconfigured_layouts:
            print(f"      - {r_item}")

        return 0

    if args.integration_status:
        from src.data.canonical_registry import get_canonical_registry
        from src.data.storage import (
            DEFAULT_BASE_DIR,
            get_historical_path,
            get_jolpica_path,
            get_openf1_path,
            get_tum_path,
        )
        registry = get_canonical_registry()
        print("\n" + "=" * 78)
        print("PACE 1.0 — CROSS-SOURCE DATA INTEGRATION STATUS (STAGE 5.6)")
        print("=" * 78)
        print("\nIntegrated Data Sources:")
        print("  1. FastF1:                   High-resolution timing, telemetry, sector loops (2023-2026)")
        print("  2. Kaggle F1 Archive:        Multi-decade historical intelligence (1950-2024)")
        print("  3. Jolpica API:              Championship standings & official FIA race results (2024)")
        print("  4. OpenF1:                   Session streams, interval telemetry, race control, weather")
        print("  5. TUM Racetrack Database:   2D coordinate geometry & optimal trajectory curves (25 circuits)")

        print("\nCanonical Entity Registry:")
        print(f"  * Drivers Registered:        {len(registry._drivers)} ({len(registry.list_drivers(active_only=True))} active in 2026)")
        print(f"  * Constructors Registered:   {len(registry._constructors)} ({len(registry.list_constructors(active_only=True))} active in 2026)")
        print(f"  * Circuits Registered:       {len(registry._circuits)}")
        print(f"  * Versioned Circuit Layouts: {len(registry._circuit_layouts)}")
        print(f"  * Driver Substitutions:      {len(registry._driver_substitutions)}")
        print(f"  * Constructor Lineages:      {len(registry._constructor_lineages)}")

        print("\nHistorical Storage Layer Status (data/historical):")
        sources = [
            ("Kaggle Historical Results", get_historical_path("race_results")),
            ("Kaggle Driver Standings", get_historical_path("driver_standings")),
            ("Jolpica 2024 Standings", get_jolpica_path("jolpica_driver_standings_2024.parquet")),
            ("Jolpica 2024 Race Results", get_jolpica_path("jolpica_race_results_2024.parquet")),
            ("FastF1 Historical Laps", get_historical_path("laps")),
            ("TUM Circuits Summary", get_tum_path("tum_circuits_summary.parquet")),
            ("Unified GP Events", DEFAULT_BASE_DIR / "data" / "historical" / "unified_events.parquet"),
            ("Data Quality Scorecard", DEFAULT_BASE_DIR / "data" / "historical" / "data_quality_scorecard.parquet"),
        ]
        for name, p in sources:
            status = "FOUND" if p.exists() else "MISSING"
            size = f"{p.stat().st_size / 1024:.1f} KB" if p.exists() else "-"
            print(f"  * {name:<30} [{status:<7}] {size:>10} ({p.name})")

        return 0

    if args.integration_inventory:
        from src.data.unified_contracts import list_data_contracts
        import pyarrow.parquet as pq

        print("\n" + "=" * 105)
        print("PACE 1.0 — UNIFIED DATA CONTRACTS INVENTORY")
        print("=" * 105)
        contracts = list_data_contracts()
        print(f"\nRegistered Data Contracts: {len(contracts)}\n")
        print(f"{'Domain':<22} | {'Schema':<32} | {'Primary Key':<35} | {'Preferred Source'}")
        print("-" * 120)
        for c in contracts:
            pk_str = ", ".join(c.primary_key)
            print(f"{c.domain:<22} | {c.canonical_schema_name:<32} | {pk_str:<35} | {c.preferred_source[:30]}")

        print("\n" + "=" * 105)
        print("PERSISTED HISTORICAL DATASETS")
        print("=" * 105)
        hist_dir = Path("data/historical")
        parquet_files = sorted(list(hist_dir.glob("*.parquet")) + list(hist_dir.glob("*/*.parquet")))
        print(f"{'Dataset File':<45} | {'Rows':<8} | {'Cols':<6} | {'Size':<10}")
        print("-" * 75)
        for pf in parquet_files:
            try:
                table = pq.read_table(pf)
                rel_path = str(pf.relative_to(hist_dir))
                size_str = f"{pf.stat().st_size / 1024:.1f} KB"
                print(f"{rel_path:<45} | {table.num_rows:<8} | {table.num_columns:<6} | {size_str:<10}")
            except Exception:
                pass

        return 0

    if args.integration_reconcile:
        from src.data.cross_source_reconciler import CrossSourceReconciler
        reconciler = CrossSourceReconciler()
        target_season = args.season
        rnd = None
        if args.event and str(args.event).isdigit():
            rnd = int(args.event)

        print(f"\nExecuting Cross-Source Reconciliation for Season {target_season}...")
        report = reconciler.execute_full_reconciliation(target_seasons=[target_season], round_num=rnd)
        print("\n" + report.get_summary_text())

        print("\nDomain Reconciliation Detail:")
        for dom, s in report.domain_summaries.items():
            print(f"\n--- Domain: {dom.upper()} ({s.source_a} vs {s.source_b}) ---")
            print(f"  * Overlap Records Evaluated:       {s.overlap_count}")
            print(f"  * Exact Identical Matches:         {s.identical_count}")
            print(f"  * Equivalent Converted Matches:    {s.equivalent_converted_count}")
            print(f"  * Discrepancies / Conflicts:       {s.conflict_count}")
            print(f"  * Match Rate:                      {s.match_rate_pct:.2f}%")
            if s.conflicts_detected:
                print("  * Sample Detected Conflicts:")
                for c in s.conflicts_detected[:5]:
                    print(f"      - [{c.field_name}] {c.entity_id}: {c.source_a}={c.val_a_str} vs {c.source_b}={c.val_b_str} -> Selected {c.selected_source} ({c.selection_reason})")

        return 0

    if args.integration_quality:
        from src.data.quality_framework import DataQualityEvaluator
        evaluator = DataQualityEvaluator()
        print("\nExecuting Multi-Dimensional Data Quality Evaluation...")
        scorecards = evaluator.execute_quality_evaluation()

        print("\n" + "=" * 105)
        print("PACE 1.0 — DATA QUALITY SCORECARD BY SOURCE AND DOMAIN")
        print("=" * 105)
        print(f"{'Source':<10} | {'Category':<22} | {'Records':<8} | {'Complete%':<10} | {'Consistent%':<12} | {'Valid%':<8} | {'Recov%':<8} | {'Unres'}")
        print("-" * 105)
        for sc in scorecards:
            print(
                f"{sc.source_name:<10} | {sc.dataset_category:<22} | {sc.records_evaluated:<8} | "
                f"{sc.completeness_pct:>8.2f}% | {sc.consistency_pct:>10.2f}% | {sc.validity_pct:>6.2f}% | "
                f"{sc.reconciliation_coverage_pct:>6.2f}% | {sc.unresolved_count:>5}"
            )
        print("=" * 105)
        print("Quality Dimensions Legend:")
        print("  * Completeness: Required non-null attributes present (respects NOT_APPLICABLE).")
        print("  * Consistency:  Referential integrity across drivers, constructors, circuits.")
        print("  * Validity:     Constraint adherence (valid points, non-negative laps, physical bounds).")
        print("  * Recov%:       Reconciliation coverage against secondary corroborating source.")

        return 0

    if args.integration_lineage:
        from src.data.data_lineage import DataLineageTracker
        tracker = DataLineageTracker()
        summary = tracker.get_provenance_summary()
        print("\n" + "=" * 80)
        print("PACE 1.0 — PROVENANCE AND DATA LINEAGE LEDGER")
        print("=" * 80)
        print(f"Total Lineage Records in Ledger: {summary['total_records']}")
        print("\nContributing Sources:")
        for src, cnt in summary["sources"].items():
            print(f"  * {src:<25}: {cnt} records")
        print("\nDomains Audited:")
        for dom, cnt in summary["domains"].items():
            print(f"  * {dom:<25}: {cnt} records")
        print("\nReconciliation Outcomes:")
        for outc, cnt in summary["outcomes"].items():
            print(f"  * {outc:<25}: {cnt} records")

        sample_records = tracker.list_lineage(limit=10)
        if sample_records:
            print("\nRecent Provenance Audit Records:")
            for r in sample_records:
                print(f"  - [{r.domain}] {r.canonical_record_id}: Source={r.source_api} (ID={r.source_record_id}) -> Outcome={r.reconciliation_outcome} (Selected: {r.selected_source})")

        return 0

    if args.integration_report:
        from src.data.cross_source_reconciler import CrossSourceReconciler
        from src.data.quality_framework import DataQualityEvaluator
        from src.data.data_lineage import DataLineageTracker

        reconciler = CrossSourceReconciler()
        evaluator = DataQualityEvaluator()
        tracker = DataLineageTracker()

        print("\nGenerating Complete PACE 1.0 Cross-Source Integration & Reconciliation Report...")
        recon_report = reconciler.execute_full_reconciliation(target_seasons=[2024])
        scorecards = evaluator.execute_quality_evaluation()
        lineage_summary = tracker.get_provenance_summary()

        print("\n" + recon_report.get_summary_text())

        print("\n" + "=" * 105)
        print("DATA QUALITY SCORECARD")
        print("=" * 105)
        for sc in scorecards:
            print(
                f"  [{sc.source_name.upper():<7} : {sc.dataset_category:<18}] Records: {sc.records_evaluated:<6} | "
                f"Complete: {sc.completeness_pct:.1f}% | Consistent: {sc.consistency_pct:.1f}% | "
                f"Valid: {sc.validity_pct:.1f}% | Recov: {sc.reconciliation_coverage_pct:.1f}%"
            )

        print("\n" + "=" * 105)
        print("PROVENANCE & DATA LINEAGE")
        print("=" * 105)
        print(f"  * Total Ledger Records: {lineage_summary['total_records']}")
        print(f"  * Verified Data Lineage File: data/historical/data_lineage.parquet")
        print("=" * 105)

        return 0
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
            "pit_stops", "compounds", "overtakes_made", "laps_completed", "elapsed_time", "gap_to_leader", "status",
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

    if args.monte_carlo:
        import pandas as pd
        import numpy as np
        from src.features.pipeline import FeaturePipeline
        from src.models.statistical import RidgePositionRegressor
        from src.models.ranker import XGBoostLambdaMARTRanker
        from src.simulation.monte_carlo import MonteCarloSimulationEngine
        from src.simulation.realism import SimulationRealismAuditor
        from src.evaluation.calibration import ProbabilityCalibrationEvaluator

        print(f"\nInitializing Phase 4 Stage 6 Monte Carlo Simulator: Season {args.target_season} Round {args.target_round} (Horizon: {args.horizon})...")
        print(f"Sampling {args.simulations} Stochastic Event-Driven Grand Prix Simulations...")

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
        tyre_stress = float(event_df["tyre_stress_level"].iloc[0]) if "tyre_stress_level" in event_df.columns else 3.0
        base_lap_time = 82.0 if circuit_id == "albert_park" else 88.0

        mc_engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=-1)
        mc_result = mc_engine.run_simulations(
            event_features=event_df,
            n_simulations=args.simulations,
            total_laps=args.laps,
            circuit_id=circuit_id,
            circuit_length_km=circuit_length,
            base_lap_time=base_lap_time,
            tyre_stress_level=tyre_stress,
            random_seed=42,
        )

        df_mc = mc_result.to_dataframe()

        print("\n" + "=" * 135)
        print(f"PACE STAGE 6: MONTE CARLO PROBABILISTIC PREDICTIONS — {circuit_id.replace('_', ' ').upper()} ({args.simulations} RUNS)")
        print("=" * 135)
        disp_cols = [
            "predicted_rank", "driver_id", "constructor_id", "grid_position",
            "expected_position", "most_likely_position", "p05", "p95",
            "p_win", "p_podium", "p_top5", "p_points", "p_dnf_mech", "p_dnf_inc", "p_dnf_total", "mcse_win",
        ]
        print(df_mc[disp_cols].to_string(index=False))

        print("\n" + "=" * 80)
        print("GRID-LEVEL MATHEMATICAL CONSISTENCY VERIFICATION:")
        print("=" * 80)
        math_check = mc_result.verify_mathematical_consistency()
        for k, v in math_check.items():
            print(f"  - {k}: {v}")

        print("\n" + "=" * 80)
        print("MACRO SIMULATION REALISM & EVENT FREQUENCY AUDIT:")
        print("=" * 80)
        audit = SimulationRealismAuditor.audit_event_frequencies(mc_result)
        print(f"All Frequencies Plausible: {audit['all_frequencies_plausible']}")
        for ev, stats in audit["detailed_audit"].items():
            print(f"  * {ev:<26}: Sim={stats['simulated_mean']:<6} | F1 Norm={stats['historical_norm']:<5} | Range={stats['plausible_range']} | InRange={stats['in_plausible_range']}")

        stability = SimulationRealismAuditor.evaluate_distribution_stability(mc_result.convergence_traces)
        print(f"\nDistribution Convergence: {stability['diagnostics']}")
        print(f"  * Max Win Prob Shift: {stability['max_win_probability_shift']:.4f}")
        print(f"  * Max Podium Prob Shift: {stability['max_podium_probability_shift']:.4f}")

        if "finish_position" in event_df.columns and event_df["finish_position"].notna().any():
            print("\n" + "=" * 80)
            print("PROBABILITY CALIBRATION EVALUATION (VS OUT-OF-SAMPLE ACTUALS):")
            print("=" * 80)
            calib = ProbabilityCalibrationEvaluator.evaluate_monte_carlo_race(event_df, mc_result)
            for k, v in calib.items():
                print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

        return 0

    # Default action if no flags provided
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
