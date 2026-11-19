"""Audit tests verifying completeness of the 2026 Formula 1 driver grid."""

import pandas as pd
import pytest

from src.data.storage import (
    load_parquet,
    get_historical_path,
    get_driver_circuit_path,
    get_processed_path,
    dataset_exists,
)


@pytest.fixture
def grid_2026_data():
    """Load datasets required for 2026 grid audit."""
    proc_race_2026 = get_processed_path("race_results", 2026)
    assert dataset_exists(proc_race_2026), "Processed 2026 race results must exist"
    df_race26 = load_parquet(proc_race_2026)

    drivers_path = get_historical_path("drivers")
    assert dataset_exists(drivers_path), "Historical drivers dataset must exist"
    df_drivers = load_parquet(drivers_path)

    app_path = get_driver_circuit_path("driver_circuit_appearances.parquet")
    assert dataset_exists(app_path), "Driver-circuit appearances dataset must exist"
    df_app = load_parquet(app_path)

    agg_path = get_driver_circuit_path("driver_circuit_aggregated.parquet")
    assert dataset_exists(agg_path), "Driver-circuit aggregated dataset must exist"
    df_agg = load_parquet(agg_path)

    return {
        "race_2026": df_race26,
        "drivers": df_drivers,
        "appearances": df_app,
        "aggregated": df_agg,
    }


def test_2026_grid_authoritative_drivers(grid_2026_data):
    """Verify that all drivers who participated in a 2026 Grand Prix are recorded."""
    df_race = grid_2026_data["race_2026"]
    participating_drivers = set(df_race["driver_id"].unique())

    # Exactly 25 drivers: 22 regular + 3 substitutes (Tsunoda, Hirakawa, Câmara)
    assert len(participating_drivers) == 25

    expected_regulars = {
        "albon", "alonso", "antonelli", "arvid_lindblad", "bearman", "bortoleto",
        "bottas", "colapinto", "gasly", "hadjar", "hamilton", "hulkenberg",
        "lawson", "leclerc", "max_verstappen", "norris", "ocon", "perez",
        "piastri", "russell", "sainz", "stroll"
    }
    expected_substitutes = {"tsunoda", "ryo_hirakawa", "rafael_camara"}

    assert expected_regulars.issubset(participating_drivers)
    assert expected_substitutes.issubset(participating_drivers)


def test_2026_grid_drivers_cross_check(grid_2026_data):
    """Verify every 2026 driver exists across canonical, appearance, and aggregated datasets."""
    df_race = grid_2026_data["race_2026"]
    df_drivers = grid_2026_data["drivers"]
    df_app = grid_2026_data["appearances"]
    df_agg = grid_2026_data["aggregated"]

    participating_drivers = df_race["driver_id"].unique()

    for d in participating_drivers:
        assert d in df_drivers["driver_id"].values, f"Driver {d} missing in historical_drivers"
        assert d in df_app["driver_id"].values, f"Driver {d} missing in driver_circuit_appearances"
        assert d in df_agg["driver_id"].values, f"Driver {d} missing in driver_circuit_aggregated"

        # Verify no duplicate appearances in (driver, season, round)
        d_apps = df_app[df_app["driver_id"] == d]
        dups = d_apps.duplicated(subset=["season", "round"]).sum()
        assert dups == 0, f"Driver {d} has duplicate race appearance records"


def test_2026_seasonal_dataset_rounds_completeness(grid_2026_data):
    """Verify that all completed rounds in 2026 contain exactly 22 classified/DNF cars."""
    df_race = grid_2026_data["race_2026"]
    round_counts = df_race.groupby("round")["driver_id"].count()

    # Rounds 1 through 20 must have exactly 22 entries per round
    assert len(round_counts) == 20
    for rnd, count in round_counts.items():
        assert count == 22, f"Round {rnd} in 2026 has {count} entries, expected 22"
