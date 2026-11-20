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
from src.simulation.rules import FIAClassificationEngine


@dataclass
class SimulationResult:
    """Complete output of a single Grand Prix simulation run."""

    final_classification: pd.DataFrame
    lap_chart: pd.DataFrame
    events: List[SimulationEvent]
    summary_metrics: Dict[str, object] = field(default_factory=dict)


class BaselineRaceSimulator:
    """Stage 3: Event-driven F1 race simulator with tyre degradation, fuel burn, and pit strategy."""

    def __init__(
        self,
        utility_scaling_factor: float = 0.05,
        tyre_model: Optional[TyreModel] = None,
        fuel_model: Optional[FuelModel] = None,
        pit_model: Optional[PitStopModel] = None,
    ):
        self.tyre_model = tyre_model or TyreModel()
        self.fuel_model = fuel_model or FuelModel()
        self.pit_model = pit_model or PitStopModel()
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

        # 1. Initialize global race state
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

            plan = self.strategy_engine.generate_default_stint_plan(
                total_laps=total_laps,
                grid_position=g_pos,
                circuit_tyre_stress=tyre_stress_level,
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

            # Sample driver initial stint degradation rate
            cpi_rel = float(row.get("cpi_reliability", 0.95))
            cpi_mult = 1.0 + max(0.0, 0.95 - cpi_rel)  # Lower reliability / worse chassis has slightly higher tyre wear
            deg_rate = self.tyre_model.sample_stint_degradation_rate(
                compound=starting_compound,
                constructor_deg_multiplier=cpi_mult,
                rng=None if deterministic else rng,
            )
            driver_stint_deg_rates[d_id] = deg_rate

        race.update_running_positions_and_gaps()

        lap_chart_records = []

        # 4. Lap-by-lap simulation loop
        for lap in range(1, total_laps + 1):
            race.current_lap = lap

            # Simulate each active car
            for car in race.cars.values():
                if not car.status.is_active:
                    continue

                d_id = car.driver_id
                u_i = driver_utility_map.get(d_id, mean_utility)
                cv_i = driver_cv_map.get(d_id, 0.003)
                plan = stint_plans[d_id]
                stint_deg = driver_stint_deg_rates.get(d_id, 0.05)

                # Strategic Pit Stop Check before completing lap
                if self.strategy_engine.should_pit_now(car, race, plan, stint_deg, rng=rng):
                    # Execute pit stop
                    self.strategy_engine.execute_pit_stop(
                        car=car,
                        race=race,
                        stint_plan=plan,
                        dispatcher=self.dispatcher,
                        rng=rng,
                        deterministic=deterministic,
                    )
                    # Re-sample degradation slope for the new tyre compound set
                    new_deg_rate = self.tyre_model.sample_stint_degradation_rate(
                        compound=car.current_compound,
                        rng=None if deterministic else rng,
                    )
                    driver_stint_deg_rates[d_id] = new_deg_rate
                    stint_deg = new_deg_rate

                # Compute lap time
                lap_time = self.pace_model.compute_lap_time(
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

                # Lap 1 standing start delay
                if lap == 1:
                    standing_start_delay = 3.50 + (car.grid_position - 1) * 0.12
                    lap_time += standing_start_delay

                car.record_lap(lap_time, lap)

                self.dispatcher.emit(
                    SimulationEvent(
                        event_type=EventType.LAP_COMPLETED,
                        lap=lap,
                        driver_id=car.driver_id,
                        constructor_id=car.constructor_id,
                        track_position=car.track_position,
                        elapsed_time=car.elapsed_time,
                        details={"lap_time": round(lap_time, 3), "compound": car.current_compound},
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
                    "status": car.status.value,
                })

        # 5. Classify final race results under FIA Sporting Regulations
        final_classification_df = FIAClassificationEngine.classify_race(race)

        lap_chart_df = pd.DataFrame(lap_chart_records)

        # Count total pit stops across field
        total_pits = sum(car.pit_stops_count for car in race.cars.values())

        summary = {
            "circuit_id": circuit_id,
            "total_laps": total_laps,
            "total_drivers": len(race.cars),
            "classified_finishers": int(final_classification_df["is_classified"].sum()),
            "total_pit_stops": total_pits,
            "winner": final_classification_df.iloc[0]["driver_id"] if not final_classification_df.empty else None,
            "total_race_time": float(final_classification_df.iloc[0]["elapsed_time"]) if not final_classification_df.empty else 0.0,
        }

        return SimulationResult(
            final_classification=final_classification_df,
            lap_chart=lap_chart_df,
            events=self.dispatcher.events,
            summary_metrics=summary,
        )
