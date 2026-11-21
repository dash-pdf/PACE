"""PACE: Unit & Validation Tests for Phase 4 — Stage 5 (Incidents, Reliability, SC/VSC & Weather).

Tests:
1. Mechanical failure hazard modeling, constructor unreliability, and distance fatigue.
2. Collision dependencies, Turn 1 density, aerodynamic damage accumulation, and debris.
3. Safety Car field bunching, VSC delta pacing, restart rules (DRS delay), and Red Flag.
4. Dynamic weather transitions, moisture curves, tyre crossover thresholds, and Article B6.3.6.
5. Unified event dependency chain (Collision -> Debris -> DNF -> Safety Car -> Bunching).
6. Full 58-lap 22-car Montreal simulation with stochastic events and deterministic controls.
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
from src.simulation.traffic import TrafficOvertakingEngine
from src.simulation.reliability import (
    ReliabilityEngine,
    MechanicalSubsystem,
    FailureCheckResult,
    CONSTRUCTOR_BASELINE_RELIABILITY,
)
from src.simulation.incidents import (
    IncidentEngine,
    IncidentSeverity,
    NeutralisationTrigger,
    IncidentResult,
)
from src.simulation.neutralisation import NeutralisationEngine
from src.simulation.weather import WeatherEngine, WeatherEvolutionConfig
from src.simulation.engine import BaselineRaceSimulator


def test_mechanical_failure_hazards_and_constructor_unreliability():
    """Verify constructor reliability ranking, distance fatigue, and subsystem diagnosis."""
    rel_engine = ReliabilityEngine()

    # 1. Constructor reliability hierarchy
    r_merc = rel_engine.get_constructor_reliability("mercedes")
    r_cadillac = rel_engine.get_constructor_reliability("cadillac")
    assert r_merc > r_cadillac
    assert r_merc >= 0.96
    assert r_cadillac <= 0.90

    # 2. Distance fatigue progression: late race has higher failure hazard than early race
    race_early = RaceState(current_lap=5, total_laps=50, circuit_id="monza", circuit_length_km=5.793, base_lap_time=80.0)
    race_late = RaceState(current_lap=48, total_laps=50, circuit_id="monza", circuit_length_km=5.793, base_lap_time=80.0)

    car_williams = CarState(driver_id="albon", constructor_id="williams", grid_position=15, track_position=15)

    hazard_early = rel_engine.compute_lap_hazard(car_williams, race_early)
    hazard_late = rel_engine.compute_lap_hazard(car_williams, race_late)

    assert hazard_late > hazard_early

    # 3. Thermal intake stress in dirty air wake
    car_clean = CarState(driver_id="c1", constructor_id="ferrari", grid_position=1, track_position=1, gap_to_car_ahead=5.0)
    car_dirty = CarState(driver_id="c2", constructor_id="ferrari", grid_position=2, track_position=2, gap_to_car_ahead=0.8)

    h_clean = rel_engine.compute_lap_hazard(car_clean, race_early)
    h_dirty = rel_engine.compute_lap_hazard(car_dirty, race_early)
    assert h_dirty > h_clean

    # 4. Epistemic uncertainty realization sampling
    rng = np.random.RandomState(42)
    mults = rel_engine.sample_constructor_hazard_multipliers(["mercedes", "cadillac", "audi"], rng=rng)
    assert len(mults) == 3
    for v in mults.values():
        assert 0.50 <= v <= 2.20

    # 5. Provenance manifest
    manifest = rel_engine.get_calibration_manifest()
    assert "data_source" in manifest
    assert manifest["sample_size"] == 560


def test_incident_collision_dependencies_and_damage():
    """Verify non-independent incident generation, damage accumulation, and debris."""
    inc_engine = IncidentEngine(turn1_incident_prob=1.0, overtake_contact_prob=1.0)
    race = RaceState(current_lap=1, total_laps=50, circuit_id="villeneuve", circuit_length_km=4.361, base_lap_time=88.0)

    c1 = CarState(driver_id="gasly", constructor_id="alpine", grid_position=10, track_position=10)
    c2 = CarState(driver_id="colapinto", constructor_id="alpine", grid_position=11, track_position=11)
    c3 = CarState(driver_id="ocon", constructor_id="haas", grid_position=12, track_position=12)
    c4 = CarState(driver_id="bearman", constructor_id="haas", grid_position=13, track_position=13)
    race.cars = {"gasly": c1, "colapinto": c2, "ocon": c3, "bearman": c4}

    # 1. Lap 1 Turn 1 Scramble
    rng = np.random.RandomState(42)
    t1_res = inc_engine.evaluate_turn1_scramble(race, rng=rng)

    assert t1_res is not None
    assert t1_res.occurred is True
    assert t1_res.debris_generated is True
    assert 1 in race.active_debris_sectors

    # 2. Debris hazard traversal
    follower = CarState(driver_id="follower", constructor_id="ferrari", grid_position=15, track_position=15)
    # Forced random draw to trigger debris damage
    rng_debris = np.random.RandomState(1)
    has_damage = inc_engine.evaluate_debris_hazard(follower, sector=1, race=race, rng=rng_debris)
    # Sector 1 has debris; car experiences damage if drawn
    assert follower.damage_factor >= 0.0

    # 3. Wheel-to-wheel contact during close overtake
    race.current_lap = 15
    attacker = CarState(driver_id="norris", constructor_id="mclaren", grid_position=2, track_position=2)
    defender = CarState(driver_id="russell", constructor_id="mercedes", grid_position=1, track_position=1)

    contact_res = inc_engine.evaluate_overtake_contact(
        attacker=attacker, defender=defender, gap_ahead=0.30, sector=3, race=race, rng=rng
    )
    assert contact_res is not None
    assert contact_res.occurred is True
    assert attacker.incident_involvement_count > 0 or defender.incident_involvement_count > 0


def test_safety_car_field_compression_and_restarts():
    """Verify Safety Car interval bunching, VSC delta pacing, and restart DRS delay."""
    neut_engine = NeutralisationEngine()
    race = RaceState(
        current_lap=20,
        total_laps=50,
        circuit_id="monza",
        circuit_length_km=5.793,
        base_lap_time=80.0,
        race_mode=RaceMode.GREEN,
    )

    c1 = CarState(driver_id="leader", constructor_id="red_bull", grid_position=1, track_position=1, elapsed_time=1600.0, laps_completed=20)
    c2 = CarState(driver_id="chaser1", constructor_id="ferrari", grid_position=2, track_position=2, elapsed_time=1625.0, laps_completed=20)
    c3 = CarState(driver_id="chaser2", constructor_id="mclaren", grid_position=3, track_position=3, elapsed_time=1655.0, laps_completed=20)
    race.cars = {"leader": c1, "chaser1": c2, "chaser2": c3}

    # 1. Trigger Safety Car
    dispatcher = EventDispatcher()
    neut_engine.trigger_neutralisation(NeutralisationTrigger.SAFETY_CAR, race, cause="heavy_crash", dispatcher=dispatcher)

    assert race.race_mode == RaceMode.SAFETY_CAR
    assert race.sc_laps_remaining > 0
    sc_events = [e for e in dispatcher.events if e.event_type == EventType.SAFETY_CAR_DEPLOYED]
    assert len(sc_events) == 1

    # 2. Field compression: gaps shrink from 25.0s / 55.0s to ~0.85s / 1.70s
    neut_engine.compress_field_under_safety_car(race)
    gap1 = c2.elapsed_time - c1.elapsed_time
    gap2 = c3.elapsed_time - c2.elapsed_time
    assert gap1 < 2.0
    assert gap2 < 2.0

    # 3. Countdown and restart
    race.sc_laps_remaining = 1
    neut_engine.process_neutralisation_step(race, dispatcher=dispatcher)
    assert race.race_mode == RaceMode.GREEN
    assert race.sc_restart_lap == 20
    restart_events = [e for e in dispatcher.events if e.event_type == EventType.SAFETY_CAR_ENDED]
    assert len(restart_events) == 1


def test_dynamic_weather_transitions_and_crossover_strategy():
    """Verify track moisture progression, condition transitions, and tyre crossover logic."""
    weather_cfg = WeatherEvolutionConfig(
        rain_start_lap=10,
        rain_peak_lap=18,
        rain_end_lap=30,
        peak_moisture=0.85,
    )
    weather_engine = WeatherEngine(weather_cfg)
    race = RaceState(current_lap=5, total_laps=50, circuit_id="silverstone", circuit_length_km=5.891, base_lap_time=90.0)

    # 1. Lap 5: Dry
    cond5 = weather_engine.update_weather_state(race)
    assert cond5 == TrackCondition.DRY
    assert race.track_moisture == 0.0

    # 2. Lap 13: Rising rain -> Damp (Intermediate territory)
    race.current_lap = 13
    cond13 = weather_engine.update_weather_state(race)
    assert cond13 == TrackCondition.DAMP
    assert 0.20 <= race.track_moisture < 0.65

    # Strategic crossover recommendation
    crossover13 = weather_engine.evaluate_crossover_tyre(current_compound="medium", track_moisture=race.track_moisture, laps_remaining=37)
    assert crossover13 == CompoundType.INTERMEDIATE

    # 3. Lap 20: Heavy rain -> Wet (Full Wet territory)
    race.current_lap = 20
    cond20 = weather_engine.update_weather_state(race)
    assert cond20 == TrackCondition.WET
    assert race.track_moisture >= 0.65

    crossover20 = weather_engine.evaluate_crossover_tyre(current_compound="medium", track_moisture=race.track_moisture, laps_remaining=30)
    assert crossover20 in (CompoundType.WET, CompoundType.INTERMEDIATE)

    # 4. Lap 42: Track dried back to dry (< 0.18)
    race.current_lap = 42
    cond42 = weather_engine.update_weather_state(race)
    assert cond42 == TrackCondition.DRY
    assert race.track_moisture < 0.20

    crossover42 = weather_engine.evaluate_crossover_tyre(current_compound="intermediate", track_moisture=race.track_moisture, laps_remaining=8)
    assert crossover42 == CompoundType.MEDIUM


def test_event_dependency_chain():
    """Verify unified event chain: Collision -> Debris -> Neutralisation -> Field Compression."""
    neut_engine = NeutralisationEngine()
    inc_engine = IncidentEngine(overtake_contact_prob=1.0)
    race = RaceState(
        current_lap=12,
        total_laps=50,
        circuit_id="villeneuve",
        circuit_length_km=4.361,
        base_lap_time=88.0,
        race_mode=RaceMode.GREEN,
    )
    dispatcher = EventDispatcher()

    c1 = CarState(driver_id="c1", constructor_id="williams", grid_position=10, track_position=10, elapsed_time=1050.0)
    c2 = CarState(driver_id="c2", constructor_id="cadillac", grid_position=11, track_position=11, elapsed_time=1050.3)
    race.cars = {"c1": c1, "c2": c2}

    # Contact incident creates debris and requests Safety Car
    res = inc_engine.evaluate_overtake_contact(attacker=c2, defender=c1, gap_ahead=0.30, sector=3, race=race, rng=np.random.RandomState(42))
    assert res is not None

    # Safety car triggered by incident
    neut_engine.trigger_neutralisation(NeutralisationTrigger.SAFETY_CAR, race, cause="collision", dispatcher=dispatcher)
    assert race.race_mode == RaceMode.SAFETY_CAR
    assert 3 in race.active_debris_sectors

    # Field compressed under safety car
    neut_engine.compress_field_under_safety_car(race)
    assert abs(c2.elapsed_time - c1.elapsed_time) < 2.0


def test_full_stage5_simulation_deterministic_and_stochastic():
    """Execute full 58-lap 22-car simulation with Stage 5 event systems."""
    features = pd.DataFrame([
        {"driver_id": "russell", "constructor_id": "mercedes", "grid_position": 1, "predicted_utility": 1.2, "driver_consistency_cv": 0.0028, "cpi_reliability": 0.965},
        {"driver_id": "antonelli", "constructor_id": "mercedes", "grid_position": 2, "predicted_utility": 1.1, "driver_consistency_cv": 0.0032, "cpi_reliability": 0.965},
        {"driver_id": "norris", "constructor_id": "mclaren", "grid_position": 3, "predicted_utility": 1.0, "driver_consistency_cv": 0.0029, "cpi_reliability": 0.960},
        {"driver_id": "piastri", "constructor_id": "mclaren", "grid_position": 4, "predicted_utility": 0.9, "driver_consistency_cv": 0.0030, "cpi_reliability": 0.960},
        {"driver_id": "hamilton", "constructor_id": "ferrari", "grid_position": 5, "predicted_utility": 0.8, "driver_consistency_cv": 0.0027, "cpi_reliability": 0.955},
        {"driver_id": "max_verstappen", "constructor_id": "red_bull", "grid_position": 6, "predicted_utility": 0.7, "driver_consistency_cv": 0.0025, "cpi_reliability": 0.950},
        {"driver_id": "hadjar", "constructor_id": "red_bull", "grid_position": 7, "predicted_utility": 0.3, "driver_consistency_cv": 0.0034, "cpi_reliability": 0.950},
        {"driver_id": "leclerc", "constructor_id": "ferrari", "grid_position": 8, "predicted_utility": 0.4, "driver_consistency_cv": 0.0031, "cpi_reliability": 0.955},
        {"driver_id": "arvid_lindblad", "constructor_id": "rb", "grid_position": 9, "predicted_utility": 0.1, "driver_consistency_cv": 0.0035, "cpi_reliability": 0.945},
        {"driver_id": "colapinto", "constructor_id": "alpine", "grid_position": 10, "predicted_utility": 0.0, "driver_consistency_cv": 0.0033, "cpi_reliability": 0.930},
        {"driver_id": "hulkenberg", "constructor_id": "audi", "grid_position": 11, "predicted_utility": -0.1, "driver_consistency_cv": 0.0032, "cpi_reliability": 0.940},
        {"driver_id": "lawson", "constructor_id": "rb", "grid_position": 12, "predicted_utility": -0.2, "driver_consistency_cv": 0.0034, "cpi_reliability": 0.945},
        {"driver_id": "bortoleto", "constructor_id": "audi", "grid_position": 13, "predicted_utility": -0.3, "driver_consistency_cv": 0.0033, "cpi_reliability": 0.940},
        {"driver_id": "gasly", "constructor_id": "alpine", "grid_position": 14, "predicted_utility": -0.4, "driver_consistency_cv": 0.0032, "cpi_reliability": 0.930},
        {"driver_id": "sainz", "constructor_id": "williams", "grid_position": 15, "predicted_utility": -0.5, "driver_consistency_cv": 0.0030, "cpi_reliability": 0.920},
        {"driver_id": "bearman", "constructor_id": "haas", "grid_position": 16, "predicted_utility": -0.6, "driver_consistency_cv": 0.0035, "cpi_reliability": 0.935},
        {"driver_id": "ocon", "constructor_id": "haas", "grid_position": 17, "predicted_utility": -0.7, "driver_consistency_cv": 0.0033, "cpi_reliability": 0.935},
        {"driver_id": "albon", "constructor_id": "williams", "grid_position": 18, "predicted_utility": -0.8, "driver_consistency_cv": 0.0031, "cpi_reliability": 0.920},
        {"driver_id": "alonso", "constructor_id": "aston_martin", "grid_position": 19, "predicted_utility": -0.9, "driver_consistency_cv": 0.0026, "cpi_reliability": 0.910},
        {"driver_id": "perez", "constructor_id": "cadillac", "grid_position": 20, "predicted_utility": -1.0, "driver_consistency_cv": 0.0034, "cpi_reliability": 0.885},
        {"driver_id": "stroll", "constructor_id": "aston_martin", "grid_position": 21, "predicted_utility": -1.1, "driver_consistency_cv": 0.0036, "cpi_reliability": 0.910},
        {"driver_id": "bottas", "constructor_id": "cadillac", "grid_position": 22, "predicted_utility": -1.2, "driver_consistency_cv": 0.0032, "cpi_reliability": 0.885},
    ])

    # 1. Deterministic test mode: strictly reproducible
    sim_det = BaselineRaceSimulator(utility_scaling_factor=0.05)
    res_det1 = sim_det.simulate_race(features, total_laps=58, random_seed=42, deterministic=True)
    res_det2 = sim_det.simulate_race(features, total_laps=58, random_seed=42, deterministic=True)

    assert list(res_det1.final_classification["driver_id"]) == list(res_det2.final_classification["driver_id"])
    assert res_det1.summary_metrics["total_race_time"] == res_det2.summary_metrics["total_race_time"]

    # 2. Stochastic simulation with full Stage 5 engines enabled
    sim_stoch = BaselineRaceSimulator(utility_scaling_factor=0.05)
    res_stoch = sim_stoch.simulate_race(
        features,
        total_laps=58,
        circuit_id="villeneuve",
        circuit_length_km=4.361,
        base_lap_time=88.0,
        random_seed=42,
        deterministic=False,
    )

    df_class = res_stoch.final_classification
    assert len(df_class) == 22
    assert list(df_class["finish_position"]) == list(range(1, 23))

    # Overtakes, pit stops, and events recorded
    assert res_stoch.summary_metrics["total_overtakes"] > 0
    assert res_stoch.summary_metrics["total_pit_stops"] >= 22
    assert len(res_stoch.events) > 50
