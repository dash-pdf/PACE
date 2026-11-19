"""Tests for data validation and quality gates."""

import pandas as pd
from src.data.validation import (
    validate_drivers,
    validate_constructors,
    validate_circuits,
    validate_events,
    validate_race_results,
    validate_qualifying_results,
)


def test_validate_drivers_success():
    """Verify validation passes on clean driver data."""
    df = pd.DataFrame([
        {"driver_id": "norris", "driver_number": 4},
        {"driver_id": "piastri", "driver_number": 81},
    ])
    report = validate_drivers(df)
    assert report.is_valid


def test_validate_drivers_duplicates():
    """Verify validation detects duplicate driver IDs."""
    df = pd.DataFrame([
        {"driver_id": "norris", "driver_number": 4},
        {"driver_id": "norris", "driver_number": 4},
    ])
    report = validate_drivers(df)
    assert not report.is_valid
    assert report.summary()["failed_checks"] > 0


def test_validate_race_results_logic():
    """Verify checks on classified status and bounds."""
    # Valid results
    df_valid = pd.DataFrame([{
        "season": 2023,
        "round": 1,
        "driver_id": "verstappen",
        "constructor_id": "red_bull",
        "finish_position": 1,
        "is_classified": True,
        "is_dnf": False,
        "points": 25.0,
        "laps_completed": 57,
    }])
    r_valid = validate_race_results(df_valid)
    assert r_valid.is_valid

    # Invalid: classified and DNF simultaneously True
    df_invalid = pd.DataFrame([{
        "season": 2023,
        "round": 1,
        "driver_id": "verstappen",
        "constructor_id": "red_bull",
        "finish_position": 1,
        "is_classified": True,
        "is_dnf": True,
        "points": -5.0,
        "laps_completed": -1,
    }])
    r_invalid = validate_race_results(df_invalid)
    assert not r_invalid.is_valid
    summary = r_invalid.summary()
    assert summary["failed_checks"] >= 3
