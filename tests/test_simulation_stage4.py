"""PACE: Unit & Validation Tests for Phase 4 — Stage 4 (Traffic, Wake & Overtaking).

Tests:
1. Circuit DRS configurations, detection points, and FIA eligibility rules.
2. Turbulent dirty-air downforce wake and accelerated tyre wear in close following.
3. Stochastic overtaking mechanics, Bradley-Terry logistic links, and DRS train neutralization.
4. FIA Appendix H Blue-Flag yielding logic for lapped traffic.
5. 2026 FIA Sporting Regulations Article B6.3.6 tyre rule and actual wet tyre usage exemption.
6. Epistemic identifiability manifests for tyre and fuel model coefficients.
7. Full 58-lap Stage 4 Grand Prix simulation with overtakes, pit stops, and validated classification.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.simulation.state import CarState, CarStatus, RaceMode, RaceState, TrackCondition
from src.simulation.events import EventDispatcher, EventType
from src.simulation.rules import FIAClassificationEngine
from src.simulation.tyres import CompoundType, TyreModel
from src.simulation.fuel import FuelModel
from src.simulation.pit_model import PitStopModel
from src.simulation.strategy import StrategyEngine, StintPlan
from src.simulation.traffic import (
    TrafficOvertakingEngine,
    CircuitDRSConfig,
    CIRCUIT_DRS_PROFILES,
    DEFAULT_DRS_CONFIG,
)
from src.simulation.engine import BaselineRaceSimulator


def test_circuit_drs_configs_and_eligibility():
    """Verify circuit DRS configurations and strict FIA eligibility criteria."""
    engine = TrafficOvertakingEngine()

    # 1. Circuit configurations exist and are calibrated
    monaco_cfg = engine.get_circuit_config("monaco")
    monza_cfg = engine.get_circuit_config("monza")
    villeneuve_cfg = engine.get_circuit_config("villeneuve")

    assert monaco_cfg.overtaking_difficulty > monza_cfg.overtaking_difficulty
    assert monaco_cfg.overtake_threshold_s > monza_cfg.overtake_threshold_s
    assert villeneuve_cfg.num_drs_zones == 3
    assert villeneuve_cfg.primary_overtake_sector == 3

    # 2. DRS Eligibility under FIA Sporting Regulations
    # Standard legal scenario: green flag, lap >= 2, dry, gap <= 1.0s
    assert engine.is_drs_eligible(0.65, RaceMode.GREEN, lap=5, track_condition=TrackCondition.DRY) is True

    # Ineligible: gap > 1.000s
    assert engine.is_drs_eligible(1.001, RaceMode.GREEN, lap=5, track_condition=TrackCondition.DRY) is False

    # Ineligible: Lap 1 (DRS disabled at race start)
    assert engine.is_drs_eligible(0.40, RaceMode.GREEN, lap=1, track_condition=TrackCondition.DRY) is False

    # Ineligible: Safety Car / VSC
    assert engine.is_drs_eligible(0.40, RaceMode.SAFETY_CAR, lap=10, track_condition=TrackCondition.DRY) is False
    assert engine.is_drs_eligible(0.40, RaceMode.VSC, lap=10, track_condition=TrackCondition.DRY) is False

    # Ineligible: Wet track condition (DRS disabled by race control)
    assert engine.is_drs_eligible(0.40, RaceMode.GREEN, lap=10, track_condition=TrackCondition.WET) is False


def test_dirty_air_aerodynamic_wake_and_accelerated_tyre_wear():
    """Verify cornering downforce loss and accelerated tyre wear in close following."""
    engine = TrafficOvertakingEngine()
    cfg = engine.get_circuit_config("villeneuve")

    # 1. Clean air (gap > 1.8s): zero aero disturbance, 1.0x wear
    aero_clean, wear_clean = engine.compute_dirty_air_effect(
        gap_ahead=2.5, in_drs_sector=False, has_drs=False, config=cfg
    )
    assert aero_clean == 0.0
    assert wear_clean == 1.0

    # 2. Following within 0.5s in non-DRS cornering sector
    aero_dirty, wear_dirty = engine.compute_dirty_air_effect(
        gap_ahead=0.5, in_drs_sector=False, has_drs=False, config=cfg
    )
    assert aero_dirty > 0.08  # cornering downforce loss
    assert wear_dirty > 1.15  # accelerated tyre wear from lateral slip

    # 3. Following within 0.5s with DRS active on straight
    aero_drs, _ = engine.compute_dirty_air_effect(
        gap_ahead=0.5, in_drs_sector=True, has_drs=True, config=cfg
    )
    # Drag reduction counteracts cornering wake loss
    assert aero_drs < aero_dirty


def test_stochastic_overtaking_mechanics_and_drs_train():
    """Verify logistic link for overtaking, driver defense, and DRS train neutralization."""
    engine = TrafficOvertakingEngine()
    cfg = engine.get_circuit_config("villeneuve")

    attacker = CarState(driver_id="norris", constructor_id="mclaren", grid_position=3, track_position=2, tyre_age=5)
    defender = CarState(driver_id="russell", constructor_id="mercedes", grid_position=1, track_position=1, tyre_age=25)

    # 1. High delta with DRS against worn tyres
    res_easy = engine.attempt_overtake(
        attacker=attacker,
        defender=defender,
        attacker_pace=85.0,
        defender_pace=86.2,
        gap_ahead=0.45,
        has_drs=True,
        defender_has_drs=False,
        sector=3,
        config=cfg,
        defender_defensive_rating=0.0,
        deterministic=True,
    )
    assert res_easy.attempted is True
    assert res_easy.prob_success > 0.70
    assert res_easy.success is True

    # 2. DRS train neutralization: defender ALSO has DRS
    res_train = engine.attempt_overtake(
        attacker=attacker,
        defender=defender,
        attacker_pace=85.5,
        defender_pace=85.5,
        gap_ahead=0.45,
        has_drs=True,
        defender_has_drs=True,  # Defender in DRS train
        sector=3,
        config=cfg,
        defender_defensive_rating=0.15,
        deterministic=True,
    )
    # Probability drops significantly when defender has DRS
    assert res_train.prob_success < res_easy.prob_success

    # 3. Track difficulty impact: Monaco vs Monza
    monaco_cfg = engine.get_circuit_config("monaco")
    monza_cfg = engine.get_circuit_config("monza")

    p_monaco = engine.attempt_overtake(
        attacker=attacker, defender=defender, attacker_pace=85.0, defender_pace=85.5,
        gap_ahead=0.40, has_drs=False, sector=1, config=monaco_cfg, deterministic=True
    ).prob_success

    p_monza = engine.attempt_overtake(
        attacker=attacker, defender=defender, attacker_pace=85.0, defender_pace=85.5,
        gap_ahead=0.40, has_drs=False, sector=1, config=monza_cfg, deterministic=True
    ).prob_success

    assert p_monza > p_monaco


def test_blue_flag_lapping_logic():
    """Verify FIA Appendix H blue flag yielding mechanics."""
    engine = TrafficOvertakingEngine()

    lead_car = CarState(driver_id="verstappen", constructor_id="red_bull", grid_position=1, track_position=1, laps_completed=35)
    lapped_car = CarState(driver_id="bottas", constructor_id="cadillac", grid_position=22, track_position=22, laps_completed=34)

    # Lead car approaches within 1.0s of -1 lap car
    lead_pen, lapped_pen, is_lapped = engine.evaluate_blue_flag(lead_car, lapped_car, gap=0.85)

    assert is_lapped is True
    assert lapped_pen >= 0.70  # Lapped car loses time yielding off line
    assert lead_pen <= 0.25   # Leader passes with minimal disruption

    # Cars on same lap: blue flag does NOT trigger
    same_lap_car = CarState(driver_id="norris", constructor_id="mclaren", grid_position=2, track_position=2, laps_completed=35)
    _, _, is_lapped_same = engine.evaluate_blue_flag(lead_car, same_lap_car, gap=0.85)
    assert is_lapped_same is False


def test_fia_article_b6_3_6_wet_exemption_strictness():
    """Verify 2026 Article B6.3.6 requires actual wet tyre usage for exemption."""
    race = RaceState(
        current_lap=50,
        total_laps=50,
        circuit_id="silverstone",
        circuit_length_km=5.891,
        base_lap_time=90.0,
        track_condition=TrackCondition.WET,  # Track declared wet by Race Director
    )

    # Car 1: Track was declared wet, but driver ONLY ran on Medium (never used wet/intermediate tyres)
    car1 = CarState(
        driver_id="driver1",
        constructor_id="c1",
        grid_position=1,
        track_position=1,
        laps_completed=50,
        elapsed_time=4500.0,
        compounds_used={"medium"},  # Only 1 dry compound, no wet compound
        status=CarStatus.FINISHED,
    )

    # Car 2: Driver actually used Intermediate tyres during race
    car2 = CarState(
        driver_id="driver2",
        constructor_id="c2",
        grid_position=2,
        track_position=2,
        laps_completed=50,
        elapsed_time=4510.0,
        compounds_used={"medium", "intermediate"},  # Actually used wet compound
        status=CarStatus.FINISHED,
    )

    race.cars = {"driver1": car1, "driver2": car2}
    df_class = FIAClassificationEngine.classify_race(race)

    driver1_res = df_class[df_class["driver_id"] == "driver1"].iloc[0]
    driver2_res = df_class[df_class["driver_id"] == "driver2"].iloc[0]

    # Driver 1 must be DSQ because exemption requires actual wet compound usage
    assert driver1_res["status"] == "dsq"
    assert bool(driver1_res["is_classified"]) is False

    # Driver 2 is validly classified under the wet exemption
    assert driver2_res["status"] == "finished"
    assert bool(driver2_res["is_classified"]) is True


def test_tyre_and_fuel_calibration_manifests():
    """Verify explicit epistemological metadata on tyre and fuel model parameters."""
    tyre_model = TyreModel()
    fuel_model = FuelModel()

    tyre_manifest = tyre_model.get_calibration_manifest()
    assert "soft" in tyre_manifest
    assert "medium" in tyre_manifest
    assert "hard" in tyre_manifest
    assert "is_provisional" in tyre_manifest["soft"]
    assert tyre_manifest["medium"]["sample_size"] > 1000
    assert "data_source" in tyre_manifest["hard"]

    fuel_manifest = fuel_model.get_calibration_manifest()
    assert "base_sensitivity" in fuel_manifest
    assert "is_provisional" in fuel_manifest
    assert "sample_size" in fuel_manifest
    assert fuel_manifest["sample_size"] == 726


def test_stage4_simulation_full_race():
    """Execute a full 58-lap Stage 4 Grand Prix simulation at Montreal with 22 drivers."""
    features = pd.DataFrame([
        {"driver_id": "russell", "constructor_id": "mercedes", "grid_position": 1, "predicted_utility": 1.2, "driver_consistency_cv": 0.0028},
        {"driver_id": "antonelli", "constructor_id": "mercedes", "grid_position": 2, "predicted_utility": 1.1, "driver_consistency_cv": 0.0032},
        {"driver_id": "norris", "constructor_id": "mclaren", "grid_position": 3, "predicted_utility": 1.0, "driver_consistency_cv": 0.0029},
        {"driver_id": "piastri", "constructor_id": "mclaren", "grid_position": 4, "predicted_utility": 0.9, "driver_consistency_cv": 0.0030},
        {"driver_id": "hamilton", "constructor_id": "ferrari", "grid_position": 5, "predicted_utility": 0.8, "driver_consistency_cv": 0.0027},
        {"driver_id": "max_verstappen", "constructor_id": "red_bull", "grid_position": 6, "predicted_utility": 0.7, "driver_consistency_cv": 0.0025},
        {"driver_id": "hadjar", "constructor_id": "red_bull", "grid_position": 7, "predicted_utility": 0.3, "driver_consistency_cv": 0.0034},
        {"driver_id": "leclerc", "constructor_id": "ferrari", "grid_position": 8, "predicted_utility": 0.4, "driver_consistency_cv": 0.0031},
        {"driver_id": "arvid_lindblad", "constructor_id": "rb", "grid_position": 9, "predicted_utility": 0.1, "driver_consistency_cv": 0.0035},
        {"driver_id": "colapinto", "constructor_id": "alpine", "grid_position": 10, "predicted_utility": 0.0, "driver_consistency_cv": 0.0033},
        {"driver_id": "hulkenberg", "constructor_id": "audi", "grid_position": 11, "predicted_utility": -0.1, "driver_consistency_cv": 0.0032},
        {"driver_id": "lawson", "constructor_id": "rb", "grid_position": 12, "predicted_utility": -0.2, "driver_consistency_cv": 0.0034},
        {"driver_id": "bortoleto", "constructor_id": "audi", "grid_position": 13, "predicted_utility": -0.3, "driver_consistency_cv": 0.0033},
        {"driver_id": "gasly", "constructor_id": "alpine", "grid_position": 14, "predicted_utility": -0.4, "driver_consistency_cv": 0.0032},
        {"driver_id": "sainz", "constructor_id": "williams", "grid_position": 15, "predicted_utility": -0.5, "driver_consistency_cv": 0.0030},
        {"driver_id": "bearman", "constructor_id": "haas", "grid_position": 16, "predicted_utility": -0.6, "driver_consistency_cv": 0.0035},
        {"driver_id": "ocon", "constructor_id": "haas", "grid_position": 17, "predicted_utility": -0.7, "driver_consistency_cv": 0.0033},
        {"driver_id": "albon", "constructor_id": "williams", "grid_position": 18, "predicted_utility": -0.8, "driver_consistency_cv": 0.0031},
        {"driver_id": "alonso", "constructor_id": "aston_martin", "grid_position": 19, "predicted_utility": -0.9, "driver_consistency_cv": 0.0026},
        {"driver_id": "perez", "constructor_id": "cadillac", "grid_position": 20, "predicted_utility": -1.0, "driver_consistency_cv": 0.0034},
        {"driver_id": "stroll", "constructor_id": "aston_martin", "grid_position": 21, "predicted_utility": -1.1, "driver_consistency_cv": 0.0036},
        {"driver_id": "bottas", "constructor_id": "cadillac", "grid_position": 22, "predicted_utility": -1.2, "driver_consistency_cv": 0.0032},
    ])

    sim = BaselineRaceSimulator(utility_scaling_factor=0.05)
    res = sim.simulate_race(
        event_features=features,
        total_laps=58,
        circuit_id="villeneuve",
        circuit_length_km=4.361,
        base_lap_time=88.0,
        tyre_stress_level=3.0,
        random_seed=42,
    )

    df_class = res.final_classification
    assert len(df_class) == 22
    assert list(df_class["finish_position"]) == list(range(1, 23))

    # Overtakes occurred and were logged
    overtake_events = [e for e in res.events if e.event_type == EventType.OVERTAKE]
    assert len(overtake_events) > 0
    assert res.summary_metrics["total_overtakes"] > 0

    # Every classified driver satisfied Article B6.3.6 (used >= 2 compounds)
    for compounds in df_class["compounds"]:
        assert "/" in compounds  # contains at least 2 compounds separated by /

    # Verify strategy diversity: field is not all identical stops
    pit_stop_counts = set(df_class["pit_stops"])
    assert len(pit_stop_counts) >= 2  # Mix of 1-stop and 2-stop strategies
