"""PACE: Simulation Orchestration Engine (Stage 2 Baseline).

Implements the event-driven lap-by-lap simulation loop:
1. Initializes grid state from Phase 3 event features (grid positions, Phase 3 utilities, driver consistency).
2. Simulates collective lap progress from Lap 1 to Total Laps.
3. Dynamically updates track positions, elapsed times, and gap intervals.
4. Produces validated FIA classifications, full lap charts, and event logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.simulation.state import CarState, CarStatus, RaceState, TrackCondition, RaceMode
from src.simulation.events import EventDispatcher, EventType, SimulationEvent
from src.simulation.pace_model import BaselinePaceModel
from src.simulation.rules import FIAClassificationEngine


@dataclass
class SimulationResult:
    """Complete output of a single Grand Prix simulation run."""

    final_classification: pd.DataFrame
    lap_chart: pd.DataFrame
    events: List[SimulationEvent]
    summary_metrics: Dict[str, object] = field(default_factory=dict)


class BaselineRaceSimulator:
    """Stage 2: Core event-driven F1 race simulator anchored on Phase 3 outputs."""

    def __init__(self, utility_scaling_factor: float = 0.05):
        self.pace_model = BaselinePaceModel(utility_scaling_factor=utility_scaling_factor)
        self.dispatcher = EventDispatcher()

    def simulate_race(
        self,
        event_features: pd.DataFrame,
        total_laps: int = 58,
        circuit_id: str = "albert_park",
        circuit_length_km: float = 5.278,
        base_lap_time: float = 80.0,
        random_seed: int = 42,
        deterministic: bool = False,
    ) -> SimulationResult:
        """Execute a full-distance Grand Prix simulation.
        
        Args:
            event_features: DataFrame containing driver_id, constructor_id, grid_position,
                            and utility / Phase 3 features.
            total_laps: Scheduled race lap count (e.g. 58 laps).
            circuit_id: Canonical circuit identifier.
            circuit_length_km: Circuit lap length in km.
            base_lap_time: Reference lap time in seconds under clean track conditions.
            random_seed: PRNG seed for deterministic reproducibility.
            deterministic: If True, execute with zero stochastic noise.
            
        Returns:
            SimulationResult containing final classification, lap chart, and event journal.
        """
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
        # Check available utility columns (Ridge, LambdaMART, or DPI fallback)
        if "predicted_utility" in event_features.columns:
            raw_utilities = event_features["predicted_utility"].values
        elif "ridge_pred_pos" in event_features.columns:
            raw_utilities = -event_features["ridge_pred_pos"].values
        elif "grid_position" in event_features.columns:
            # Fallback utility inversely proportional to starting grid position
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

        # 3. Initialize car states
        # Sort initial grid by starting grid position
        sorted_grid = event_features.sort_values("grid_position")

        for rank_idx, (_, row) in enumerate(sorted_grid.iterrows(), 1):
            d_id = str(row["driver_id"])
            c_id = str(row.get("constructor_id", "unknown"))
            g_pos = int(row.get("grid_position", rank_idx))

            car = CarState(
                driver_id=d_id,
                constructor_id=c_id,
                grid_position=g_pos,
                track_position=rank_idx,
                elapsed_time=0.0,
                laps_completed=0,
                status=CarStatus.ACTIVE,
            )
            race.cars[d_id] = car

        race.update_running_positions_and_gaps()

        lap_chart_records = []

        # 4. Lap-by-lap simulation loop
        for lap in range(1, total_laps + 1):
            race.current_lap = lap

            # Simulate each active car
            for car in race.cars.values():
                if not car.status.is_active:
                    continue

                u_i = driver_utility_map.get(car.driver_id, mean_utility)
                cv_i = driver_cv_map.get(car.driver_id, 0.003)

                lap_time = self.pace_model.compute_lap_time(
                    car=car,
                    race=race,
                    driver_utility=u_i,
                    mean_utility=mean_utility,
                    consistency_cv=cv_i,
                    rng=rng,
                    deterministic=deterministic,
                )

                # Lap 1 standing start delay: standing start from rest adds ~3.5s plus grid spacing delay
                if lap == 1:
                    # Cars further back on the grid react slightly later to lights out (approx 0.12s per row)
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
                        details={"lap_time": round(lap_time, 3)},
                    )
                )

            # Recalculate track running order and gaps based on updated elapsed times
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
                    "status": car.status.value,
                })

        # 5. Classify final race results under FIA Sporting Regulations
        final_classification_df = FIAClassificationEngine.classify_race(race)

        lap_chart_df = pd.DataFrame(lap_chart_records)

        summary = {
            "circuit_id": circuit_id,
            "total_laps": total_laps,
            "total_drivers": len(race.cars),
            "classified_finishers": int(final_classification_df["is_classified"].sum()),
            "winner": final_classification_df.iloc[0]["driver_id"] if not final_classification_df.empty else None,
            "total_race_time": float(final_classification_df.iloc[0]["elapsed_time"]) if not final_classification_df.empty else 0.0,
        }

        return SimulationResult(
            final_classification=final_classification_df,
            lap_chart=lap_chart_df,
            events=self.dispatcher.events,
            summary_metrics=summary,
        )
