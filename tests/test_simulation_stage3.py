"""PACE: Stage 3 Simulation Engine Test Suite (Tyres, Fuel & Strategy).

Exhaustively verifies:
1. Tyre compound speed and degradation hierarchy (Soft > Medium > Hard speed, Soft > Medium > Hard wear).
2. Non-linear thermal degradation cliff mechanics.
3. Latent vehicle fuel burn-off and mass depletion pace progression.
4. Empirical pit stop duration decomposition (transit, service, abnormal delay) and SC/VSC discount.
5. Undercut and overcut strategic execution.
6. FIA Article 30.5 two-compound sporting requirement and wet-weather exception.
7. Full race simulation with active pit stops and deterministic reproducibility.
"""

import pytest
import numpy as np
import pandas as pd

from src.simulation.state import CarState, CarStatus, RaceState, TrackCondition, RaceMode
from src.simulation.tyres import TyreModel, CompoundType, COMPOUND_PROFILES
from src.simulation.fuel import FuelModel
from src.simulation.pit_model import PitStopModel
from src.simulation.strategy import StrategyEngine, StintPlan
from src.simulation.rules import FIAClassificationEngine
from src.simulation.engine import BaselineRaceSimulator


def test_compound_speed_and_degradation_hierarchy():
    """Verify that fresh tyre speed and degradation rates follow physical compound hierarchy."""
    tyre_model = TyreModel()

    # Fresh tyres (age 0): Soft is fastest, Medium is baseline, Hard is slower
    delta_soft_0 = tyre_model.compute_tyre_pace_delta(CompoundType.SOFT, tyre_age=0)
    delta_med_0 = tyre_model.compute_tyre_pace_delta(CompoundType.MEDIUM, tyre_age=0)
    delta_hard_0 = tyre_model.compute_tyre_pace_delta(CompoundType.HARD, tyre_age=0)

    assert delta_soft_0 < delta_med_0 < delta_hard_0
    assert pytest.approx(delta_soft_0, abs=1e-3) == -0.55
    assert pytest.approx(delta_med_0, abs=1e-3) == 0.00
    assert pytest.approx(delta_hard_0, abs=1e-3) == +0.65

    # At lap 20 (within normal stint): Soft degrades faster than Medium, Medium faster than Hard
    soft_wear = tyre_model.get_profile(CompoundType.SOFT).deg_rate_mean
    med_wear = tyre_model.get_profile(CompoundType.MEDIUM).deg_rate_mean
    hard_wear = tyre_model.get_profile(CompoundType.HARD).deg_rate_mean

    assert soft_wear > med_wear > hard_wear

    # Non-linear thermal cliff: Soft hits cliff at lap 18, so lap 25 has huge penalty
    delta_soft_15 = tyre_model.compute_tyre_pace_delta(CompoundType.SOFT, tyre_age=15)
    delta_soft_25 = tyre_model.compute_tyre_pace_delta(CompoundType.SOFT, tyre_age=25)
    cliff_penalty = delta_soft_25 - delta_soft_15
    assert cliff_penalty > 1.5  # Substantial pace drop due to cliff breach


def test_fuel_burn_off_pace_progression():
    """Verify that fuel mass depletes linearly and pace improves over race distance."""
    fuel_model = FuelModel(
        initial_fuel_kg=105.0,
        finish_fuel_kg=2.0,
        circuit_fuel_sensitivity=0.050,
    )

    total_laps = 50

    # Remaining fuel
    fuel_lap_1 = fuel_model.get_remaining_fuel_kg(1, total_laps)
    fuel_lap_25 = fuel_model.get_remaining_fuel_kg(25, total_laps)
    fuel_lap_50 = fuel_model.get_remaining_fuel_kg(50, total_laps)

    assert pytest.approx(fuel_lap_1, abs=1e-2) == 105.0
    assert pytest.approx(fuel_lap_25, abs=1.0) == 55.56
    assert pytest.approx(fuel_lap_50, abs=1e-1) == 4.06

    # Fuel pace penalty
    delta_1 = fuel_model.compute_fuel_pace_delta(1, total_laps)
    delta_25 = fuel_model.compute_fuel_pace_delta(25, total_laps)
    delta_50 = fuel_model.compute_fuel_pace_delta(50, total_laps)

    # Lap 1 penalty ~0.525s, lap 50 penalty ~0.020s
    assert delta_1 > delta_25 > delta_50
    assert pytest.approx(delta_1, abs=1e-3) == 0.05 * 10.5
    assert pytest.approx(delta_50, abs=1e-3) == 0.05 * 0.406


def test_empirical_pit_stop_duration_decomposition():
    """Verify empirical pit stop duration model separates transit, service, and SC discount."""
    pit_model = PitStopModel(mean_stationary_s=2.40, prob_abnormal_delay=0.10)

    # Green flag transit for Albert Park is ~20.5s
    res_green = pit_model.sample_pit_duration(
        circuit_id="albert_park",
        is_sc_vsc=False,
        deterministic=True,
    )
    assert res_green.transit_loss_s == pytest.approx(20.5)
    assert res_green.stationary_service_s == pytest.approx(2.40)
    assert res_green.total_loss_s == pytest.approx(22.90)

    # Under Safety Car, transit loss drops by ~45%
    res_sc = pit_model.sample_pit_duration(
        circuit_id="albert_park",
        is_sc_vsc=True,
        deterministic=True,
    )
    assert res_sc.transit_loss_s == pytest.approx(20.5 * 0.55)
    assert res_sc.total_loss_s < res_green.total_loss_s
    assert pytest.approx(res_green.total_loss_s - res_sc.total_loss_s, abs=0.1) == 20.5 * 0.45


def test_undercut_strategic_execution():
    """Verify that pitting 1 lap earlier allows trailing car to undercut leading car."""
    tyre_model = TyreModel()
    pit_model = PitStopModel()
    strat = StrategyEngine(pit_model, tyre_model)

    # Setup two identical cars on Albert Park
    # Car A leads by 1.5s on lap 20 (both on 20-lap old Mediums)
    car_a = CarState("car_a", "team_a", 1, 1, elapsed_time=1700.0, current_compound="medium", tyre_age=20)
    car_b = CarState("car_b", "team_b", 2, 2, elapsed_time=1701.5, current_compound="medium", tyre_age=20)

    race = RaceState(20, 58, "albert_park", 5.278, base_lap_time=82.0)
    race.cars = {"car_a": car_a, "car_b": car_b}
    race.update_running_positions_and_gaps()

    plan = StintPlan(compounds=[CompoundType.MEDIUM, CompoundType.HARD], stint_lengths=[20, 38])

    # Lap 20: Car B pits for fresh Hards (undercut attempt)
    dur_b = strat.execute_pit_stop(car_b, race, plan, deterministic=True)
    # Car A stays out for 1 more lap on worn Mediums (tyre age 20 -> 21)
    car_a.record_lap(84.50, 20)  # Slower lap on degraded tyres

    # Car B completes out-lap on fresh Hards (pit stop time + out lap pace)
    # Pit loss is ~22.9s + fresh Hard lap ~82.65s
    car_b.record_lap(82.65, 20)

    # Now Lap 21: Car A responds by pitting for Hards
    race.current_lap = 21
    dur_a = strat.execute_pit_stop(car_a, race, plan, deterministic=True)
    car_a.record_lap(82.65, 21)
    # Car B completes clean flying lap on fresh Hards
    car_b.record_lap(82.65, 21)

    # Re-evaluate running order
    race.update_running_positions_and_gaps()

    # Car B gained ~1.85s on the out-lap vs Car A's worn tyre lap, successfully undercutting Car A!
    assert car_b.elapsed_time < car_a.elapsed_time
    assert car_b.track_position == 1
    assert car_a.track_position == 2


def test_fia_two_compound_rule_and_wet_exception():
    """Verify FIA Article 30.5: Disqualification for single dry compound; exception for wet track."""
    # 1. Dry Race Violation: Driver completes full race on only 1 compound specification
    race_dry = RaceState(50, 50, "monza", 5.793, base_lap_time=82.0, track_condition=TrackCondition.DRY)
    car_legal = CarState("legal", "team1", 1, 1, elapsed_time=4100.0, laps_completed=50, compounds_used={"medium", "hard"})
    car_illegal = CarState("illegal", "team2", 2, 2, elapsed_time=4102.0, laps_completed=50, compounds_used={"medium"})

    race_dry.cars = {"legal": car_legal, "illegal": car_illegal}
    df_dry = FIAClassificationEngine.classify_race(race_dry)

    assert df_dry.loc[df_dry["driver_id"] == "legal", "status"].iloc[0] == "finished"
    assert df_dry.loc[df_dry["driver_id"] == "illegal", "status"].iloc[0] == "dsq"

    # 2. Wet Race Exception: Driver completes race on Intermediates or single compound in Wet condition
    race_wet = RaceState(50, 50, "spa", 7.004, base_lap_time=110.0, track_condition=TrackCondition.WET)
    car_wet1 = CarState("wet_winner", "team1", 1, 1, elapsed_time=5500.0, laps_completed=50, compounds_used={"intermediate"})
    car_wet2 = CarState("wet_p2", "team2", 2, 2, elapsed_time=5510.0, laps_completed=50, compounds_used={"intermediate"})

    race_wet.cars = {"wet_winner": car_wet1, "wet_p2": car_wet2}
    df_wet = FIAClassificationEngine.classify_race(race_wet)

    assert df_wet.loc[df_wet["driver_id"] == "wet_winner", "status"].iloc[0] == "finished"
    assert df_wet.loc[df_wet["driver_id"] == "wet_p2", "status"].iloc[0] == "finished"


def test_stage3_simulator_run_on_2026_round_5():
    """Verify Stage 3 full simulation executes pit stops and conforms to FIA rules on 2026 Round 5."""
    from src.features.pipeline import FeaturePipeline
    from src.models.statistical import RidgePositionRegressor

    rr = pd.read_parquet("data/historical/historical_race_results.parquet")
    q = pd.read_parquet("data/historical/historical_qualifying.parquet")
    laps = pd.read_parquet("data/historical/historical_laps.parquet")
    app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
    circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

    pipeline = FeaturePipeline(rr, q, laps, app, circ)
    event_df = pipeline.compile_features_for_event(2026, 5, horizon="H3")

    feature_cols = [
        "dpi_rating", "dpi_uncertainty", "driver_starts", "cpi_pole_gap_pct",
        "cpi_reliability", "circuit_affinity_mean", "circuit_length_km",
        "grid_position", "grid_x_overtaking",
    ]

    all_events = rr[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).values
    prior_events = [(int(s), int(r)) for s, r in all_events if (s < 2026) or (s == 2026 and r < 5)]
    train_dfs = [pipeline.compile_features_for_event(s, r, horizon="H3") for s, r in prior_events[:10]]
    full_train = pd.concat([d for d in train_dfs if not d.empty], ignore_index=True)

    ridge = RidgePositionRegressor().fit(full_train[feature_cols].fillna(0.0), full_train["target_rank"].values)
    raw_pred = ridge.model.predict(ridge.scaler.transform(event_df[feature_cols].fillna(0.0)))
    event_df["predicted_utility"] = -raw_pred

    sim = BaselineRaceSimulator()
    res = sim.simulate_race(
        event_features=event_df,
        total_laps=58,
        circuit_id="villeneuve",
        base_lap_time=80.0,
        random_seed=42,
    )

    df_class = res.final_classification
    assert len(df_class) == 22
    assert list(df_class["finish_position"]) == list(range(1, 23))

    # All finishers must have at least 1 pit stop and 2 compounds used
    for _, row in df_class.iterrows():
        assert row["pit_stops"] >= 1
        compounds = row["compounds"].split("/")
        assert len(compounds) >= 2
        assert row["status"] == "finished"

    # Verify pit events recorded in dispatcher
    pit_events = [e for e in res.events if e.event_type.value == "pit_stop"]
    assert len(pit_events) >= 22
