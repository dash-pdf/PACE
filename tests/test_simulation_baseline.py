"""PACE: Stage 2 Baseline Simulation Engine Test Suite.

Verifies:
1. State representation invariants and strict temporal monotonicity.
2. FIA Article 54.1 90% distance rule classification.
3. Permutation validity (1..N finish order, zero duplicate positions).
4. Base pace model sensitivity to Phase 3 utility scores.
5. Deterministic reproducibility under fixed random seeds.
6. Full Grand Prix execution on real Phase 3 event features (2026 Round 5).
"""

import pytest
import numpy as np
import pandas as pd

from src.simulation.state import CarState, CarStatus, RaceState, TrackCondition, RaceMode
from src.simulation.events import EventDispatcher, SimulationEvent, EventType
from src.simulation.pace_model import BaselinePaceModel
from src.simulation.rules import FIAClassificationEngine
from src.simulation.engine import BaselineRaceSimulator


def test_car_state_monotonicity():
    """Verify that CarState strictly accumulates lap times monotonically."""
    car = CarState(
        driver_id="russell",
        constructor_id="mercedes",
        grid_position=1,
        track_position=1,
    )
    assert car.elapsed_time == 0.0
    assert car.laps_completed == 0

    car.record_lap(85.123, 1)
    assert car.elapsed_time == pytest.approx(85.123)
    assert car.laps_completed == 1
    assert car.tyre_age == 1

    car.record_lap(84.950, 2)
    assert car.elapsed_time == pytest.approx(85.123 + 84.950)
    assert car.laps_completed == 2
    assert car.tyre_age == 2
    assert len(car.lap_times_history) == 2


def test_race_state_ordered_field_and_gaps():
    """Verify RaceState correctly orders cars and calculates gaps to leader and car ahead."""
    race = RaceState(
        current_lap=1,
        total_laps=50,
        circuit_id="monza",
        circuit_length_km=5.793,
        base_lap_time=82.0,
    )

    car1 = CarState("russell", "mercedes", 1, 1, elapsed_time=82.0, laps_completed=1)
    car2 = CarState("antonelli", "mercedes", 2, 2, elapsed_time=82.5, laps_completed=1)
    car3 = CarState("norris", "mclaren", 3, 3, elapsed_time=83.1, laps_completed=1)

    race.cars = {"russell": car1, "antonelli": car2, "norris": car3}
    race.update_running_positions_and_gaps()

    assert race.leader_id == "russell"
    assert car1.track_position == 1
    assert car1.gap_to_leader == 0.0
    assert car1.gap_to_car_ahead == 0.0

    assert car2.track_position == 2
    assert car2.gap_to_leader == pytest.approx(0.5)
    assert car2.gap_to_car_ahead == pytest.approx(0.5)

    assert car3.track_position == 3
    assert car3.gap_to_leader == pytest.approx(1.1)
    assert car3.gap_to_car_ahead == pytest.approx(0.6)


def test_fia_classification_90_percent_rule():
    """Verify FIA Article 54.1: cars completing >= 90% distance are classified."""
    race = RaceState(
        current_lap=50,
        total_laps=50,
        circuit_id="silverstone",
        circuit_length_km=5.891,
        base_lap_time=90.0,
    )

    # Winner completes 50 laps with mandatory 2 compounds
    c_winner = CarState("verstappen", "red_bull", 1, 1, elapsed_time=4500.0, laps_completed=50, compounds_used={"medium", "hard"})
    # Lead lap finisher
    c_p2 = CarState("norris", "mclaren", 2, 2, elapsed_time=4503.2, laps_completed=50, compounds_used={"medium", "hard"})
    # Lapped finisher (-1 lap = 49 laps >= 45 laps threshold)
    c_p3 = CarState("albon", "williams", 10, 3, elapsed_time=4480.0, laps_completed=49, compounds_used={"medium", "hard"})
    # Retired car on lap 46 (46 >= 45 laps -> classified finisher despite retirement!)
    c_ret_late = CarState("sainz", "williams", 5, 4, elapsed_time=4200.0, laps_completed=46, status=CarStatus.DNF_MECHANICAL, compounds_used={"medium", "hard"})
    # Retired car on lap 20 (< 45 laps -> unclassified DNF)
    c_ret_early = CarState("gasly", "alpine", 15, 5, elapsed_time=1800.0, laps_completed=20, status=CarStatus.DNF_INCIDENT, compounds_used={"medium"})

    race.cars = {
        "verstappen": c_winner,
        "norris": c_p2,
        "albon": c_p3,
        "sainz": c_ret_late,
        "gasly": c_ret_early,
    }

    df = FIAClassificationEngine.classify_race(race)

    assert len(df) == 5
    assert list(df["finish_position"]) == [1, 2, 3, 4, 5]

    # Verstappen and Norris are classified on lead lap
    assert bool(df.loc[df["driver_id"] == "verstappen", "is_classified"].iloc[0]) is True
    assert df.loc[df["driver_id"] == "verstappen", "gap_to_leader"].iloc[0] == "WINNER"

    assert bool(df.loc[df["driver_id"] == "norris", "is_classified"].iloc[0]) is True
    assert df.loc[df["driver_id"] == "norris", "gap_to_leader"].iloc[0] == "+3.200s"

    # Albon is classified at -1 lap
    assert bool(df.loc[df["driver_id"] == "albon", "is_classified"].iloc[0]) is True
    assert df.loc[df["driver_id"] == "albon", "gap_to_leader"].iloc[0] == "+1 Lap"

    # Gasly completed only 20 laps -> strictly unclassified DNF ranked last
    assert bool(df.loc[df["driver_id"] == "gasly", "is_classified"].iloc[0]) is False
    assert df.iloc[-1]["driver_id"] == "gasly"


def test_baseline_pace_model_utility_scaling():
    """Verify that higher Phase 3 utility translates directly to faster lap times."""
    pace_model = BaselinePaceModel(utility_scaling_factor=0.05)
    race = RaceState(1, 50, "spa", 7.004, base_lap_time=105.0)

    car_fast = CarState("russell", "mercedes", 1, 1)
    car_slow = CarState("stroll", "aston_martin", 20, 20)

    # Russell has utility +2.0, Stroll has utility -2.0, mean is 0.0
    t_fast = pace_model.compute_lap_time(car_fast, race, driver_utility=2.0, mean_utility=0.0, deterministic=True)
    t_slow = pace_model.compute_lap_time(car_slow, race, driver_utility=-2.0, mean_utility=0.0, deterministic=True)

    # 4.0 utility diff * 0.05 = 0.20s per lap advantage
    assert t_fast < t_slow
    assert pytest.approx(t_slow - t_fast, abs=1e-4) == 0.20


def test_baseline_race_simulator_deterministic_reproducibility():
    """Verify that running simulator with identical seed produces identical results."""
    features = pd.DataFrame([
        {"driver_id": "russell", "constructor_id": "mercedes", "grid_position": 1, "predicted_utility": 2.5},
        {"driver_id": "antonelli", "constructor_id": "mercedes", "grid_position": 2, "predicted_utility": 1.5},
        {"driver_id": "norris", "constructor_id": "mclaren", "grid_position": 3, "predicted_utility": 0.8},
        {"driver_id": "piastri", "constructor_id": "mclaren", "grid_position": 4, "predicted_utility": 0.7},
    ])

    sim = BaselineRaceSimulator()
    res1 = sim.simulate_race(features, total_laps=10, base_lap_time=80.0, random_seed=42)
    res2 = sim.simulate_race(features, total_laps=10, base_lap_time=80.0, random_seed=42)

    pd.testing.assert_frame_equal(res1.final_classification, res2.final_classification)
    pd.testing.assert_frame_equal(res1.lap_chart, res2.lap_chart)


def test_baseline_race_simulator_run_on_2026_round_5():
    """Verify full-distance simulation on Season 2026 Round 5 feature set."""
    from src.features.pipeline import FeaturePipeline

    rr = pd.read_parquet("data/historical/historical_race_results.parquet")
    q = pd.read_parquet("data/historical/historical_qualifying.parquet")
    laps = pd.read_parquet("data/historical/historical_laps.parquet")
    app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
    circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

    pipeline = FeaturePipeline(rr, q, laps, app, circ)
    event_df = pipeline.compile_features_for_event(2026, 5, horizon="H3")

    assert not event_df.empty
    assert len(event_df) == 22

    # Fit quick Ridge model to supply Phase 3 utilities
    from src.models.statistical import RidgePositionRegressor
    all_events = rr[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values
    prior_events = [(int(s), int(r)) for s, r in all_events if (s < 2026) or (s == 2026 and r < 5)]

    train_dfs = [pipeline.compile_features_for_event(s, r, horizon="H3") for s, r in prior_events[:10]]
    train_dfs = [df for df in train_dfs if not df.empty]
    full_train = pd.concat(train_dfs, ignore_index=True)

    feature_cols = [
        "dpi_rating", "dpi_uncertainty", "driver_starts", "cpi_pole_gap_pct",
        "cpi_reliability", "circuit_affinity_mean", "circuit_length_km",
        "grid_position", "grid_x_overtaking",
    ]
    ridge = RidgePositionRegressor().fit(full_train[feature_cols].fillna(0.0), full_train["target_rank"].values)
    raw_pred = ridge.model.predict(ridge.scaler.transform(event_df[feature_cols].fillna(0.0)))
    event_df["predicted_utility"] = -raw_pred

    sim = BaselineRaceSimulator()
    result = sim.simulate_race(
        event_features=event_df,
        total_laps=58,
        circuit_id="albert_park",
        base_lap_time=80.0,
        random_seed=42,
    )

    df_class = result.final_classification
    assert len(df_class) == 22
    assert list(df_class["finish_position"]) == list(range(1, 23))
    assert df_class["is_classified"].all()  # In baseline without DNF, all finish
    assert result.summary_metrics["winner"] is not None
    assert len(result.lap_chart) == 22 * 58
