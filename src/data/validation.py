"""Data validation and quality assurance module for PACE.

Validates primary key uniqueness, foreign key consistency, value bounds,
and DNS/DNF/DSQ classification logic across historical datasets.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationReport:
    """Holds results of dataset validation checks."""

    def __init__(self):
        self.checks: List[Dict[str, Any]] = []

    def add_check(self, name: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        """Record the outcome of a single validation rule."""
        passed_bool = bool(passed)
        self.checks.append({
            "name": name,
            "passed": passed_bool,
            "message": message,
            "details": details or {},
        })
        log_fn = logger.info if passed_bool else logger.warning
        status = "PASSED" if passed_bool else "FAILED"
        log_fn(f"[Quality Gate] {name}: {status} - {message}")

    @property
    def is_valid(self) -> bool:
        """Return True if all validation rules passed."""
        return all(c["passed"] for c in self.checks)

    def summary(self) -> Dict[str, Any]:
        """Return dictionary summary of validation outcomes."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        return {
            "total_checks": total,
            "passed_checks": passed,
            "failed_checks": total - passed,
            "is_valid": self.is_valid,
            "checks": self.checks,
        }


def validate_drivers(df_drivers: pd.DataFrame) -> ValidationReport:
    """Validate drivers dataset schema, uniqueness, and completeness."""
    report = ValidationReport()

    if df_drivers.empty:
        report.add_check("drivers_not_empty", False, "Drivers dataframe is empty")
        return report
    report.add_check("drivers_not_empty", True, f"Found {len(df_drivers)} driver records")

    # Uniqueness check on driver_id
    dups = df_drivers.duplicated(subset=["driver_id"]).sum()
    report.add_check("drivers_unique_id", dups == 0, f"Duplicate driver_id count: {dups}")

    # Null identifiers
    null_ids = df_drivers["driver_id"].isna().sum() + (df_drivers["driver_id"] == "").sum()
    report.add_check("drivers_no_null_id", null_ids == 0, f"Empty or null driver_id count: {null_ids}")

    return report


def validate_constructors(df_constructors: pd.DataFrame) -> ValidationReport:
    """Validate constructors dataset schema, uniqueness, and completeness."""
    report = ValidationReport()

    if df_constructors.empty:
        report.add_check("constructors_not_empty", False, "Constructors dataframe is empty")
        return report
    report.add_check("constructors_not_empty", True, f"Found {len(df_constructors)} constructor records")

    dups = df_constructors.duplicated(subset=["constructor_id"]).sum()
    report.add_check("constructors_unique_id", dups == 0, f"Duplicate constructor_id count: {dups}")

    null_ids = df_constructors["constructor_id"].isna().sum() + (df_constructors["constructor_id"] == "").sum()
    report.add_check("constructors_no_null_id", null_ids == 0, f"Empty constructor_id count: {null_ids}")

    return report


def validate_circuits(df_circuits: pd.DataFrame) -> ValidationReport:
    """Validate circuits dataset."""
    report = ValidationReport()

    if df_circuits.empty:
        report.add_check("circuits_not_empty", False, "Circuits dataframe is empty")
        return report
    report.add_check("circuits_not_empty", True, f"Found {len(df_circuits)} circuit records")

    dups = df_circuits.duplicated(subset=["circuit_id"]).sum()
    report.add_check("circuits_unique_id", dups == 0, f"Duplicate circuit_id count: {dups}")

    # Latitude / Longitude bounds
    valid_coords = (
        df_circuits["latitude"].between(-90, 90) & df_circuits["longitude"].between(-180, 180)
    ).all()
    report.add_check("circuits_valid_coords", bool(valid_coords), "Latitude and longitude coordinates within valid bounds")

    return report


def validate_events(df_events: pd.DataFrame, df_circuits: Optional[pd.DataFrame] = None) -> ValidationReport:
    """Validate event schedule."""
    report = ValidationReport()

    if df_events.empty:
        report.add_check("events_not_empty", False, "Events dataframe is empty")
        return report
    report.add_check("events_not_empty", True, f"Found {len(df_events)} events across seasons")

    # Uniqueness of (season, round)
    dups = df_events.duplicated(subset=["season", "round"]).sum()
    report.add_check("events_unique_round", dups == 0, f"Duplicate (season, round) count: {dups}")

    # Check circuit foreign keys if circuits provided
    if df_circuits is not None and not df_circuits.empty:
        known_circuits = set(df_circuits["circuit_id"])
        unknown = set(df_events["circuit_id"]) - known_circuits
        report.add_check(
            "events_valid_circuit_fk",
            len(unknown) == 0,
            f"Events with unknown circuit_id: {len(unknown)}",
            {"unknown_circuits": list(unknown)},
        )

    return report


def validate_race_results(
    df_results: pd.DataFrame,
    df_drivers: Optional[pd.DataFrame] = None,
    df_constructors: Optional[pd.DataFrame] = None,
    df_events: Optional[pd.DataFrame] = None,
) -> ValidationReport:
    """Validate race results dataset."""
    report = ValidationReport()

    if df_results.empty:
        report.add_check("results_not_empty", False, "Race results dataframe is empty")
        return report
    report.add_check("results_not_empty", True, f"Found {len(df_results)} race result entries")

    # Primary key uniqueness: (season, round, driver_id)
    dups = df_results.duplicated(subset=["season", "round", "driver_id"]).sum()
    report.add_check("results_unique_entry", dups == 0, f"Duplicate (season, round, driver_id) entries: {dups}")

    # Finishing position consistency: classified drivers should have positive position
    classified = df_results[df_results["is_classified"]]
    invalid_pos = (classified["finish_position"] <= 0).sum()
    report.add_check("results_valid_classified_pos", invalid_pos == 0, f"Classified drivers with invalid position: {invalid_pos}")

    # DNF and Classified mutually exclusive
    overlap = (df_results["is_classified"] & df_results["is_dnf"]).sum()
    report.add_check("results_flags_mutually_exclusive", overlap == 0, f"Drivers flagged both classified and DNF: {overlap}")

    # Points non-negative
    neg_points = (df_results["points"] < 0).sum()
    report.add_check("results_non_negative_points", neg_points == 0, f"Negative points count: {neg_points}")

    # Laps completed non-negative
    neg_laps = (df_results["laps_completed"] < 0).sum()
    report.add_check("results_non_negative_laps", neg_laps == 0, f"Negative laps completed count: {neg_laps}")

    # Foreign key check on drivers
    if df_drivers is not None and not df_drivers.empty:
        known_drivers = set(df_drivers["driver_id"])
        unknown_d = set(df_results["driver_id"]) - known_drivers
        report.add_check(
            "results_valid_driver_fk",
            len(unknown_d) == 0,
            f"Results with unknown driver_id: {len(unknown_d)}",
            {"unknown_drivers": list(unknown_d)},
        )

    # Foreign key check on constructors
    if df_constructors is not None and not df_constructors.empty:
        known_c = set(df_constructors["constructor_id"])
        unknown_c = set(df_results["constructor_id"]) - known_c
        report.add_check(
            "results_valid_constructor_fk",
            len(unknown_c) == 0,
            f"Results with unknown constructor_id: {len(unknown_c)}",
            {"unknown_constructors": list(unknown_c)},
        )

    # Foreign key check on events (season, round)
    if df_events is not None and not df_events.empty:
        known_rounds = set(zip(df_events["season"], df_events["round"]))
        res_rounds = set(zip(df_results["season"], df_results["round"]))
        unknown_r = res_rounds - known_rounds
        report.add_check(
            "results_valid_event_fk",
            len(unknown_r) == 0,
            f"Results with unknown (season, round): {len(unknown_r)}",
        )

    return report


def validate_qualifying_results(df_qualy: pd.DataFrame) -> ValidationReport:
    """Validate qualifying results."""
    report = ValidationReport()

    if df_qualy.empty:
        report.add_check("qualifying_not_empty", False, "Qualifying dataframe is empty")
        return report
    report.add_check("qualifying_not_empty", True, f"Found {len(df_qualy)} qualifying entries")

    dups = df_qualy.duplicated(subset=["season", "round", "driver_id"]).sum()
    report.add_check("qualifying_unique_entry", dups == 0, f"Duplicate qualifying entries: {dups}")

    invalid_pos = (df_qualy["qualifying_position"] <= 0).sum()
    report.add_check("qualifying_valid_pos", invalid_pos == 0, f"Invalid qualifying positions count: {invalid_pos}")

    return report



def validate_driver_circuit_appearances(df_app: pd.DataFrame) -> ValidationReport:
    """Validate granular event-level driver circuit appearance records."""
    report = ValidationReport()

    if df_app.empty:
        report.add_check("appearances_not_empty", False, "Driver circuit appearances dataframe is empty")
        return report
    report.add_check("appearances_not_empty", True, f"Found {len(df_app)} driver-circuit appearances")

    # Uniqueness check on (driver_id, season, round)
    dups = df_app.duplicated(subset=["driver_id", "season", "round"]).sum()
    report.add_check("appearances_unique_entry", dups == 0, f"Duplicate (driver_id, season, round) count: {dups}")

    # Classified positions
    classified = df_app[df_app["is_classified"]]
    invalid_pos = (classified["finish_position"] <= 0).sum()
    report.add_check("appearances_valid_classified_pos", invalid_pos == 0, f"Classified drivers with invalid position: {invalid_pos}")

    # Regulation era validity
    valid_eras = {"ground_effect_2022_2026", "turbo_hybrid_2014_2021", "v8_2006_2013", "v10_1996_2005", "early_pre_1996"}
    invalid_eras = (~df_app["regulation_era"].isin(valid_eras)).sum()
    report.add_check("appearances_valid_eras", invalid_eras == 0, f"Appearances with invalid regulation era: {invalid_eras}")

    return report


def validate_driver_circuit_aggregated(df_agg: pd.DataFrame) -> ValidationReport:
    """Validate aggregated driver-circuit profiles."""
    report = ValidationReport()

    if df_agg.empty:
        report.add_check("aggregated_not_empty", False, "Aggregated driver-circuit dataframe is empty")
        return report
    report.add_check("aggregated_not_empty", True, f"Found {len(df_agg)} driver-circuit profile records")

    # Uniqueness of (driver_id, circuit_id)
    dups = df_agg.duplicated(subset=["driver_id", "circuit_id"]).sum()
    report.add_check("aggregated_unique_profile", dups == 0, f"Duplicate (driver_id, circuit_id) profiles: {dups}")

    # Starts positive
    invalid_starts = (df_agg["total_starts"] <= 0).sum()
    report.add_check("aggregated_positive_starts", invalid_starts == 0, f"Profiles with zero or negative starts: {invalid_starts}")

    # Wins <= Starts
    invalid_wins = (df_agg["wins"] > df_agg["total_starts"]).sum()
    report.add_check("aggregated_wins_le_starts", invalid_wins == 0, f"Profiles with wins > starts: {invalid_wins}")

    # DNF rate bounds [0.0, 1.0]
    invalid_dnf = (~df_agg["dnf_rate"].between(0.0, 1.0)).sum()
    report.add_check("aggregated_dnf_rate_bounds", invalid_dnf == 0, f"Profiles with DNF rate out of [0, 1]: {invalid_dnf}")

    return report


def validate_circuit_characteristics(df_char: pd.DataFrame) -> ValidationReport:
    """Validate circuit technical characteristics."""
    report = ValidationReport()

    if df_char.empty:
        report.add_check("characteristics_not_empty", False, "Circuit characteristics dataframe is empty")
        return report
    report.add_check("characteristics_not_empty", True, f"Found {len(df_char)} circuit technical profiles")

    dups = df_char.duplicated(subset=["circuit_id"]).sum()
    report.add_check("characteristics_unique_id", dups == 0, f"Duplicate circuit_id count: {dups}")

    invalid_length = (df_char["circuit_length_km"] <= 0).sum()
    report.add_check("characteristics_positive_length", invalid_length == 0, f"Circuits with length <= 0: {invalid_length}")

    return report
