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
        version="PACE 0.2.1-dev (Phase 2: Expanded Historical Data Infrastructure)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PACE: Predictive Analytics for Competitive Evaluation")
    print("Formula 1 Race Finishing Position Prediction System")
    print("=" * 70)
    print("Current Phase: Phase 2 (Expanded Historical Data Infrastructure)")
    print("=" * 70)

    if args.status:
        print("\nProject Component Status:")
        print("  - Data Ingestion & Storage: [src/data] (Phase 2 Implemented)")
        print("      * General Season Data (2023-2026)")
        print("      * Driver-Circuit Historical Intelligence (Complete career appearances)")
        print("      * Circuit Technical Characteristics & Regulation Eras")
        print("  - Performance Modelling:    [src/models] (Pending Phase 3)")
        print("  - Feature Engineering:      [src/features] (Pending Phase 4)")
        print("  - Race Simulation:          [src/simulation] (Pending Phase 6)")
        print("  - Model Evaluation:         [src/evaluation] (Pending Phase 7)")
        print("  - Reporting & Export:       [src/reporting] (Pending Phase 8)")
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

    # Default action if no flags provided
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
