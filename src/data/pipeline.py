"""Historical data ingestion and compilation pipeline for PACE.

Coordinates downloading, caching, normalizing, validating, and compiling
Formula 1 datasets across historical seasons and driver-circuit intelligence.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.data.schemas import (
    DRIVERS_SCHEMA,
    CONSTRUCTORS_SCHEMA,
    CIRCUITS_SCHEMA,
    EVENTS_SCHEMA,
    SESSIONS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    SPRINT_RESULTS_SCHEMA,
    LAPS_SCHEMA,
    PIT_STOPS_SCHEMA,
    PRACTICE_LAPS_SCHEMA,
)
from src.data.storage import (
    ensure_directories,
    compile_historical_dataset,
    get_historical_path,
    get_driver_circuit_path,
    load_parquet,
    dataset_exists,
    DEFAULT_BASE_DIR,
)
from src.data.fastf1_collector import FastF1Collector
from src.data.driver_circuit import DriverCircuitIntelligence
from src.data.validation import (
    validate_drivers,
    validate_constructors,
    validate_circuits,
    validate_events,
    validate_race_results,
    validate_qualifying_results,
    validate_driver_circuit_appearances,
    validate_driver_circuit_aggregated,
    validate_circuit_characteristics,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class HistoricalDataPipeline:
    """End-to-end historical data acquisition, normalization, and validation orchestrator."""

    def __init__(
        self,
        seasons: Optional[List[int]] = None,
        base_dir: Optional[Path] = None,
        request_delay: float = 0.25,
    ):
        self.seasons = seasons or [2023, 2024, 2025, 2026]
        self.base_dir = base_dir or DEFAULT_BASE_DIR
        self.collector = FastF1Collector(request_delay=request_delay)
        self.dc_intelligence = DriverCircuitIntelligence(request_delay=request_delay)
        ensure_directories(self.base_dir)

    def run_ingestion(
        self,
        force: bool = False,
        sample_only: bool = False,
        include_driver_circuit: bool = True,
        driver_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute the ingestion process across all designated seasons and circuit intelligence."""
        logger.info(f"Starting PACE Historical Data Ingestion for seasons: {self.seasons}")
        metrics: Dict[str, Any] = {
            "seasons": self.seasons,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "seasonal_summary": {},
        }

        for season in self.seasons:
            logger.info(f"=== Processing Season {season} ===")
            season_metrics: Dict[str, int] = {}

            # 1. Core Reference Entities
            df_drivers = self.collector.collect_drivers(season, force=force)
            df_const = self.collector.collect_constructors(season, force=force)
            df_circuits = self.collector.collect_circuits(season, force=force)
            df_events, df_sessions = self.collector.collect_events_and_sessions(season, force=force)

            season_metrics["drivers"] = len(df_drivers)
            season_metrics["constructors"] = len(df_const)
            season_metrics["circuits"] = len(df_circuits)
            season_metrics["events"] = len(df_events)
            season_metrics["sessions"] = len(df_sessions)

            max_rounds = 2 if sample_only else None

            # 2. Race & Qualifying Results
            df_race = self.collector.collect_race_results(season, max_rounds=max_rounds, force=force)
            df_qualy = self.collector.collect_qualifying_results(season, max_rounds=max_rounds, force=force)
            df_sprints = self.collector.collect_sprint_results(season, max_rounds=max_rounds, force=force)

            season_metrics["race_results"] = len(df_race)
            season_metrics["qualifying_results"] = len(df_qualy)
            season_metrics["sprint_results"] = len(df_sprints)

            # 3. Lap times & Pit stops
            sample_rounds = [1, 2] if sample_only else None
            df_laps, df_pits = self.collector.collect_laps_and_pits(
                season,
                rounds=sample_rounds,
                max_laps_per_round=100 if sample_only else 500,
                force=force,
            )
            season_metrics["laps"] = len(df_laps)
            season_metrics["pit_stops"] = len(df_pits)

            # 4. Practice laps
            df_practice = self.collector.collect_practice_laps(
                season,
                max_sessions=2 if sample_only else 3,
                force=force,
            )
            season_metrics["practice_laps"] = len(df_practice)

            metrics["seasonal_summary"][season] = season_metrics

        # 5. Compile into master general historical datasets
        logger.info("Compiling seasonal datasets into master historical Parquet files...")
        categories = [
            ("drivers", DRIVERS_SCHEMA),
            ("constructors", CONSTRUCTORS_SCHEMA),
            ("circuits", CIRCUITS_SCHEMA),
            ("events", EVENTS_SCHEMA),
            ("sessions", SESSIONS_SCHEMA),
            ("race_results", RACE_RESULTS_SCHEMA),
            ("qualifying", QUALIFYING_RESULTS_SCHEMA),
            ("sprints", SPRINT_RESULTS_SCHEMA),
            ("laps", LAPS_SCHEMA),
            ("pit_stops", PIT_STOPS_SCHEMA),
            ("practice_laps", PRACTICE_LAPS_SCHEMA),
        ]

        # Compile across all seasons present in data/processed
        processed_dir = self.base_dir / "data" / "processed"
        discovered_seasons = [int(p.name) for p in processed_dir.glob("*") if p.is_dir() and p.name.isdigit()]
        all_compile_seasons = sorted(list(set(self.seasons + discovered_seasons)))

        compiled_counts: Dict[str, int] = {}
        for cat, schema in categories:
            compiled_path = compile_historical_dataset(
                cat,
                all_compile_seasons,
                schema=schema,
                base_dir=self.base_dir,
            )
            if compiled_path and dataset_exists(compiled_path):
                compiled_counts[cat] = len(load_parquet(compiled_path))
            else:
                compiled_counts[cat] = 0

        metrics["compiled_totals"] = compiled_counts

        # 6. Driver-Circuit Historical Intelligence
        if include_driver_circuit:
            logger.info("Building Driver-Circuit Historical Intelligence records...")
            dc_metrics = self.run_driver_circuit_pipeline(
                driver_ids=driver_ids,
                force=force,
                sample_only=sample_only,
            )
            metrics["driver_circuit_summary"] = dc_metrics

        metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
        return metrics

    def run_driver_circuit_pipeline(
        self,
        driver_ids: Optional[List[str]] = None,
        force: bool = False,
        sample_only: bool = False,
    ) -> Dict[str, int]:
        """Collect and compile complete driver-circuit appearance and aggregated datasets."""
        # Default drivers: if sample_only, select key contemporary drivers
        if sample_only and driver_ids is None:
            driver_ids = ["max_verstappen", "hamilton", "alonso", "leclerc", "norris"]

        df_app = self.dc_intelligence.build_driver_circuit_appearances(
            driver_ids=driver_ids,
            force=force,
        )
        # Circuit technical characteristics across all circuits in appearances & registry
        df_char = self.dc_intelligence.build_circuit_characteristics(
            df_appearances=df_app,
            force=force,
        )
        df_agg = self.dc_intelligence.build_driver_circuit_aggregated(
            df_appearances=df_app,
            force=force,
        )

        return {
            "circuit_characteristics": len(df_char),
            "driver_circuit_appearances": len(df_app),
            "driver_circuit_aggregated_profiles": len(df_agg),
        }

    def run_validation(self) -> Dict[str, Any]:
        """Execute data validation gates on all compiled datasets."""
        logger.info("Executing PACE Quality Validation Gates...")
        reports: Dict[str, Any] = {}

        # 1. General Historical Reference & Result Tables
        try:
            df_drivers = load_parquet(get_historical_path("drivers", base_dir=self.base_dir))
            df_const = load_parquet(get_historical_path("constructors", base_dir=self.base_dir))
            df_circuits = load_parquet(get_historical_path("circuits", base_dir=self.base_dir))
            df_events = load_parquet(get_historical_path("events", base_dir=self.base_dir))
            df_race = load_parquet(get_historical_path("race_results", base_dir=self.base_dir))
            df_qualy = load_parquet(get_historical_path("qualifying", base_dir=self.base_dir))

            reports["drivers"] = validate_drivers(df_drivers).summary()
            reports["constructors"] = validate_constructors(df_const).summary()
            reports["circuits"] = validate_circuits(df_circuits).summary()
            reports["events"] = validate_events(df_events, df_circuits=df_circuits).summary()
            reports["race_results"] = validate_race_results(df_race, df_drivers=df_drivers, df_constructors=df_const, df_events=df_events).summary()
            reports["qualifying"] = validate_qualifying_results(df_qualy).summary()
        except Exception as e:
            logger.error(f"Error validating general historical datasets: {e}")
            reports["general_historical_error"] = str(e)

        # 2. Driver-Circuit Intelligence Tables (if present)
        app_path = get_driver_circuit_path("driver_circuit_appearances.parquet", base_dir=self.base_dir)
        agg_path = get_driver_circuit_path("driver_circuit_aggregated.parquet", base_dir=self.base_dir)
        char_path = get_driver_circuit_path("circuit_characteristics.parquet", base_dir=self.base_dir)

        if dataset_exists(app_path):
            df_app = load_parquet(app_path)
            reports["driver_circuit_appearances"] = validate_driver_circuit_appearances(df_app).summary()

        if dataset_exists(agg_path):
            df_agg = load_parquet(agg_path)
            reports["driver_circuit_aggregated"] = validate_driver_circuit_aggregated(df_agg).summary()

        if dataset_exists(char_path):
            df_char = load_parquet(char_path)
            reports["circuit_characteristics"] = validate_circuit_characteristics(df_char).summary()

        all_valid = all(
            rep.get("is_valid", False)
            for rep in reports.values()
            if isinstance(rep, dict) and "is_valid" in rep
        )

        final_summary = {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "is_valid": all_valid,
            "components": reports,
        }

        rep_dir = self.base_dir / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rep_dir / "historical_data_validation_report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, default=str)

        logger.info(f"Validation report saved to {rep_path} (Valid: {all_valid})")
        return final_summary
