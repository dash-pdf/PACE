"""Data infrastructure package for PACE."""

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
from src.data.fastf1_collector import FastF1Collector, parse_time_to_ms, classify_status
from src.data.storage import (
    save_parquet,
    load_parquet,
    dataset_exists,
    compile_historical_dataset,
    get_processed_path,
    get_historical_path,
    get_raw_path,
)
from src.data.validation import (
    ValidationReport,
    validate_drivers,
    validate_constructors,
    validate_circuits,
    validate_events,
    validate_race_results,
    validate_qualifying_results,
)
from src.data.pipeline import HistoricalDataPipeline

__all__ = [
    "DRIVERS_SCHEMA",
    "CONSTRUCTORS_SCHEMA",
    "CIRCUITS_SCHEMA",
    "EVENTS_SCHEMA",
    "SESSIONS_SCHEMA",
    "RACE_RESULTS_SCHEMA",
    "QUALIFYING_RESULTS_SCHEMA",
    "SPRINT_RESULTS_SCHEMA",
    "LAPS_SCHEMA",
    "PIT_STOPS_SCHEMA",
    "PRACTICE_LAPS_SCHEMA",
    "FastF1Collector",
    "parse_time_to_ms",
    "classify_status",
    "save_parquet",
    "load_parquet",
    "dataset_exists",
    "compile_historical_dataset",
    "get_processed_path",
    "get_historical_path",
    "get_raw_path",
    "ValidationReport",
    "validate_drivers",
    "validate_constructors",
    "validate_circuits",
    "validate_events",
    "validate_race_results",
    "validate_qualifying_results",
    "HistoricalDataPipeline",
]
