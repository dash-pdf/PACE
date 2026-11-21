"""PACE: Simulation Orchestration Engine (Stage 3: Tyres, Fuel & Strategy).

Implements the event-driven lap-by-lap simulation loop:
1. Initializes grid state from Phase 3 event features (grid positions, Phase 3 utilities, driver consistency).
2. Sets up pre-race stint plans (FIA Article 30.5 two-compound compliance).
3. Evaluates dynamic pit stop triggers (stint target, thermal cliff, SC/VSC window, traffic).
4. Executes pit stops with empirical transit, service, and abnormal delay distributions.
5. Simulates collective lap progress factoring in tyre wear, compound deltas, and fuel mass depletion.
6. Produces validated FIA classifications, full lap charts, and event journals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.simulation.state import CarState, CarStatus, RaceState, TrackCondition, RaceMode
from src.simulation.events import EventDispatcher, EventType, SimulationEvent
from src.simulation.pace_model import StochasticPaceModel
from src.simulation.tyres import TyreModel, CompoundType
from src.simulation.fuel import FuelModel
from src.simulation.pit_model import PitStopModel
from src.simulation.strategy import StrategyEngine, StintPlan
from src.simulation.traffic import TrafficOvertakingEngine, CircuitDRSConfig
from src.simulation.rules import FIAClassificationEngine


@dataclass
class SimulationResult:
    """Complete output of a single Grand Prix simulation run."""

    final_classification: pd.DataFrame
    lap_chart: pd.DataFrame
    events: List[SimulationEvent]
    summary_metrics: Dict[str, object] = field(default_factory=dict)


class BaselineRaceSimulator:
    """Stage 4: Event-driven F1 race simulator with tyres, fuel, pit strategy, DRS, and overtaking."""

    def __init__(
        self,
        utility_scaling_factor: float = 0.05,
        tyre_model: Optional[TyreModel] = None,
        fuel_model: Optional[FuelModel] = None,
        pit_model: Optional[PitStopModel] = None,
        traffic_engine: Optional[TrafficOvertakingEngine] = None,
    ):
        self.tyre_model = tyre_model or TyreModel()
        self.fuel_model = fuel_model or FuelModel()
        self.pit_model = pit_model or PitStopModel()
        self.traffic_engine = traffic_engine or TrafficOvertakingEngine()
        self.pace_model = StochasticPaceModel(
            utility_scaling_factor=utility_scaling_factor,
            tyre_model=self.tyre_model,
            fuel_model=self.fuel_model,
        )
        self.strategy_engine = StrategyEngine(
            pit_model=self.pit_model,
            tyre_model=self.tyre_model,
        )
        self.dispatcher = EventDispatcher()

    def simulate_race(
        self,
        event_features: pd.DataFrame,
        total_laps: int = 58,
        circuit_id: str = "albert_park",
        circuit_length_km: float = 5.278,
        base_lap_time: float = 80.0,
        tyre_stress_level: float = 3.0,
        random_seed: int = 42,
        deterministic: bool = False,
    ) -> SimulationResult:
        """Execute a full-distance Grand Prix simulation."""
        rng = np.random.RandomState(random_seed)
        self.dispatcher.clear()

        # 1. Initialize global race state and circuit DRS configuration
        drs_config = self.traffic_engine.get_circuit_config(circuit_id)
        sector_fractions = drs_config.sector_fractions

        race = RaceState(
            current_lap=0,
            total_laps=total_laps,
            circuit_id=circuit_id,
            circuit_length_km=circuit_length_km,
            base_lap_time=base_lap_time,
            track_condition=TrackCondition.DRY,
            race_mode=RaceMode.GREEN,
        )

        # 2. Extract Phase 3 utilities
        if "predicted_utility" in event_features.columns:
            raw_utilities = event_features["predicted_utility"].values
        elif "ridge_pred_pos" in event_features.columns:
            raw_utilities = -event_features["ridge_pred_pos"].values
        elif "grid_position" in event_features.columns:
            raw_utilities = -(event_features["grid_position"].values - 1.0) * 0.4
        else:
            raw_utilities = np.zeros(len(event_features))

        mean_utility = float(np.mean(raw_utilities)) if len(raw_utilities) > 0 else 0.0

        driver_utility_map = dict(zip(event_features["driver_id"], raw_utilities))
        driver_cv_map = (
            dict(zip(event_features["driver_id"], event_features["driver_consistency_cv"]))
            if "driver_consistency_cv" in event_features.columns
            else {}
        )

        # Sample circuit fuel sensitivity with uncertainty
        fuel_sens = self.fuel_model.sample_circuit_sensitivity(rng=None if deterministic else rng)

        # 3. Initialize car states and pre-race stint plans
        sorted_grid = event_features.sort_values("grid_position")
        stint_plans: Dict[str, StintPlan] = {}
        driver_stint_deg_rates: Dict[str, float] = {}

        for rank_idx, (_, row) in enumerate(sorted_grid.iterrows(), 1):
            d_id = str(row["driver_id"])
            c_id = str(row.get("constructor_id", "unknown"))
            g_pos = int(row.get("grid_position", rank_idx))

            # Sample driver initial stint degradation rate
            cpi_rel = float(row.get("cpi_reliability", 0.95))
            cpi_mult = 1.0 + max(0.0, 0.95 - cpi_rel)  # Lower reliability / worse chassis has slightly higher tyre wear
            deg_rate = self.tyre_model.sample_stint_degradation_rate(
                compound="medium",
                constructor_deg_multiplier=cpi_mult,
                rng=None if deterministic else rng,
            )
            driver_stint_deg_rates[d_id] = deg_rate

            pace_adv = float(driver_utility_map.get(d_id, mean_utility) - mean_utility)
            plan = self.strategy_engine.generate_default_stint_plan(
                total_laps=total_laps,
                grid_position=g_pos,
                circuit_tyre_stress=tyre_stress_level,
                driver_deg_rate=deg_rate,
                pace_advantage=pace_adv,
            )
            stint_plans[d_id] = plan
            starting_compound = plan.compounds[0].value

            car = CarState(
                driver_id=d_id,
                constructor_id=c_id,
                grid_position=g_pos,
                track_position=rank_idx,
                elapsed_time=0.0,
                laps_completed=0,
                current_compound=starting_compound,
                tyre_age=0,
                compounds_used={starting_compound},
                status=CarStatus.ACTIVE,
            )
            race.cars[d_id] = car

        race.update_running_positions_and_gaps()

        lap_chart_records = []

        # 4. Lap-by-lap simulation loop with micro-sector gap tracking and overtaking
        for lap in range(1, total_laps + 1):
            race.current_lap = lap

            # Check strategic pit stops before cars start the lap
            for car in race.cars.values():
                if not car.status.is_active:
                    continue
                d_id = car.driver_id
                plan = stint_plans[d_id]
                stint_deg = driver_stint_deg_rates.get(d_id, 0.05)

                if self.strategy_engine.should_pit_now(car, race, plan, stint_deg, rng=rng):
                    self.strategy_engine.execute_pit_stop(
                        car=car,
                        race=race,
                        stint_plan=plan,
                        dispatcher=self.dispatcher,
                        rng=rng,
                        deterministic=deterministic,
                    )
                    new_deg_rate = self.tyre_model.sample_stint_degradation_rate(
                        compound=car.current_compound,
                        rng=None if deterministic else rng,
                    )
                    driver_stint_deg_rates[d_id] = new_deg_rate

            # Compute nominal lap pace for each active car
            car_sector_times: Dict[str, List[float]] = {}
            for car in race.cars.values():
                if not car.status.is_active:
                    continue
                d_id = car.driver_id
                u_i = driver_utility_map.get(d_id, mean_utility)
                cv_i = driver_cv_map.get(d_id, 0.003)
                stint_deg = driver_stint_deg_rates.get(d_id, 0.05)

                nom_lap_time = self.pace_model.compute_lap_time(
                    car=car,
                    race=race,
                    driver_utility=u_i,
                    mean_utility=mean_utility,
                    consistency_cv=cv_i,
                    stint_deg_rate=stint_deg,
                    fuel_sensitivity=fuel_sens,
                    rng=rng,
                    deterministic=deterministic,
                )
                if lap == 1:
                    standing_start_delay = 3.50 + (car.grid_position - 1) * 0.12
                    nom_lap_time += standing_start_delay

                car_sector_times[d_id] = [float(f * nom_lap_time) for f in sector_fractions]

            # Simulate through the 3 micro-sectors (S1, S2, S3)
            for s_idx, (f_frac, s_num) in enumerate(zip(sector_fractions, (1, 2, 3))):
                ordered_active = [c for c in race.get_ordered_field() if c.status.is_active]

                # Pass 1: Gaps, DRS eligibility, dirty air aero/tyre effect, and blue flags
                for pos_idx, car in enumerate(ordered_active):
                    d_id = car.driver_id
                    sec_time = car_sector_times[d_id][s_idx]

                    if pos_idx == 0:
                        car.drs_active = False
                        gap_ahead = 999.0
                    else:
                        car_ahead = ordered_active[pos_idx - 1]
                        gap_ahead = max(0.0, car.elapsed_time - car_ahead.elapsed_time)

                        # Blue flag check for lapped cars
                        lead_pen, lapped_pen, is_lapped = self.traffic_engine.evaluate_blue_flag(
                            lead_car=car_ahead,
                            lapped_car=car,
                            gap=gap_ahead,
                        )
                        if is_lapped:
                            sec_time += lapped_pen
                            car.times_lapped += 1
                            self.dispatcher.emit(
                                SimulationEvent(
                                    event_type=EventType.LAPPED,
                                    lap=lap,
                                    driver_id=car.driver_id,
                                    constructor_id=car.constructor_id,
                                    track_position=car.track_position,
                                    elapsed_time=car.elapsed_time + sec_time,
                                    details={"lapped_by": car_ahead.driver_id, "sector": s_num, "yield_loss_s": lapped_pen},
                                )
                            )
                        else:
                            # DRS eligibility check
                            has_drs = self.traffic_engine.is_drs_eligible(
                                gap_at_detection=gap_ahead,
                                race_mode=race.race_mode,
                                lap=lap,
                                track_condition=race.track_condition,
                            )
                            car.drs_active = has_drs

                            # Dirty air aerodynamic wake & accelerated tyre wear
                            in_drs_sector = s_num in drs_config.drs_sectors
                            aero_delta, wear_mult = self.traffic_engine.compute_dirty_air_effect(
                                gap_ahead=gap_ahead,
                                in_drs_sector=in_drs_sector,
                                has_drs=has_drs,
                                config=drs_config,
                            )
                            sec_time += aero_delta

                            if wear_mult > 1.0:
                                extra_deg = (wear_mult - 1.0) * 0.003
                                driver_stint_deg_rates[d_id] = min(0.25, driver_stint_deg_rates[d_id] + extra_deg)

                    car_sector_times[d_id][s_idx] = sec_time

                # Pass 2: Stochastic overtaking in designated primary overtaking sector
                is_overtake_sector = (s_num == drs_config.primary_overtake_sector)
                if is_overtake_sector and lap >= 2 and race.race_mode == RaceMode.GREEN:
                    for i in range(len(ordered_active) - 1, 0, -1):
                        attacker = ordered_active[i]
                        defender = ordered_active[i - 1]

                        if defender.laps_completed != attacker.laps_completed:
                            continue

                        gap_to_def = max(0.0, attacker.elapsed_time - defender.elapsed_time)
                        if gap_to_def <= 0.75:
                            att_pace = car_sector_times[attacker.driver_id][s_idx] / f_frac
                            def_pace = car_sector_times[defender.driver_id][s_idx] / f_frac
                            def_rating = self.traffic_engine.evaluate_defensive_rating(
                                defender_utility=driver_utility_map.get(defender.driver_id, mean_utility),
                                mean_utility=mean_utility,
                                driver_cv=driver_cv_map.get(defender.driver_id, 0.003),
                            )

                            ot_res = self.traffic_engine.attempt_overtake(
                                attacker=attacker,
                                defender=defender,
                                attacker_pace=att_pace,
                                defender_pace=def_pace,
                                gap_ahead=gap_to_def,
                                has_drs=attacker.drs_active,
                                defender_has_drs=defender.drs_active,
                                sector=s_num,
                                config=drs_config,
                                defender_defensive_rating=def_rating,
                                rng=rng,
                                deterministic=deterministic,
                            )

                            if ot_res.success:
                                attacker.overtakes_made += 1
                                defender.times_overtaken += 1

                                car_sector_times[attacker.driver_id][s_idx] -= 0.22
                                car_sector_times[defender.driver_id][s_idx] += 0.22

                                ordered_active[i], ordered_active[i - 1] = ordered_active[i - 1], ordered_active[i]

                                self.dispatcher.emit(
                                    SimulationEvent(
                                        event_type=EventType.OVERTAKE,
                                        lap=lap,
                                        driver_id=attacker.driver_id,
                                        constructor_id=attacker.constructor_id,
                                        track_position=defender.track_position,
                                        elapsed_time=attacker.elapsed_time + car_sector_times[attacker.driver_id][s_idx],
                                        details={
                                            "attacker": attacker.driver_id,
                                            "defender": defender.driver_id,
                                            "sector": s_num,
                                            "drs_active": attacker.drs_active,
                                            "net_pace_delta": round(ot_res.net_pace_delta, 3),
                                            "tyre_age_delta": ot_res.tyre_age_delta,
                                            "prob_success": round(ot_res.prob_success, 3),
                                        },
                                    )
                                )
                            elif ot_res.attempted:
                                car_sector_times[attacker.driver_id][s_idx] += 0.06

            # Complete lap for each active car
            for car in race.cars.values():
                if not car.status.is_active:
                    continue
                d_id = car.driver_id
                lap_time = sum(car_sector_times[d_id])
                car.record_lap(lap_time, lap)

                self.dispatcher.emit(
                    SimulationEvent(
                        event_type=EventType.LAP_COMPLETED,
                        lap=lap,
                        driver_id=car.driver_id,
                        constructor_id=car.constructor_id,
                        track_position=car.track_position,
                        elapsed_time=car.elapsed_time,
                        details={
                            "lap_time": round(lap_time, 3),
                            "compound": car.current_compound,
                            "drs": car.drs_active,
                            "overtakes": car.overtakes_made,
                        },
                    )
                )

            # Recalculate track running order and gaps
            race.update_running_positions_and_gaps()

            # Record state for lap chart
            for car in race.cars.values():
                lap_chart_records.append({
                    "lap": lap,
                    "driver_id": car.driver_id,
                    "constructor_id": car.constructor_id,
                    "position": car.track_position,
                    "lap_time": car.last_lap_time,
                    "elapsed_time": round(car.elapsed_time, 3),
                    "gap_to_leader": round(car.gap_to_leader, 3),
                    "compound": car.current_compound,
                    "tyre_age": car.tyre_age,
                    "pit_stops": car.pit_stops_count,
                    "overtakes": car.overtakes_made,
                    "drs_active": car.drs_active,
                    "status": car.status.value,
                })

        # 5. Classify final race results under FIA Sporting Regulations
        final_classification_df = FIAClassificationEngine.classify_race(race)

        lap_chart_df = pd.DataFrame(lap_chart_records)

        # Count total pit stops, overtakes, and blue flag events across field
        total_pits = sum(car.pit_stops_count for car in race.cars.values())
        total_overtakes = sum(car.overtakes_made for car in race.cars.values())
        total_lappings = sum(car.times_lapped for car in race.cars.values())

        summary = {
            "circuit_id": circuit_id,
            "total_laps": total_laps,
            "total_drivers": len(race.cars),
            "classified_finishers": int(final_classification_df["is_classified"].sum()),
            "total_pit_stops": total_pits,
            "total_overtakes": total_overtakes,
            "total_lappings": total_lappings,
            "winner": final_classification_df.iloc[0]["driver_id"] if not final_classification_df.empty else None,
            "total_race_time": float(final_classification_df.iloc[0]["elapsed_time"]) if not final_classification_df.empty else 0.0,
        }

        return SimulationResult(
            final_classification=final_classification_df,
            lap_chart=lap_chart_df,
            events=self.dispatcher.events,
            summary_metrics=summary,
        )

        return SimulationResult(
            final_classification=final_classification_df,
            lap_chart=lap_chart_df,
            events=self.dispatcher.events,
            summary_metrics=summary,
        )
