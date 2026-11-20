"""PACE: Race Strategy & Pit Window Decision Engine (Stage 3).

Models:
1. Optimal pre-race stint planning (1-stop vs 2-stop strategies).
2. Dynamic in-race pit triggers (degradation cliff breach, scheduled window).
3. Opportunistic pit stops under Safety Car / VSC ("cheap pit stops").
4. Undercut and overcut mechanics.
5. Traffic-aware pit window evaluation (avoiding dirty air traffic re-entry).
6. FIA Article 30.5 compliance (mandatory use of 2 distinct dry compounds in dry races).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from src.simulation.state import CarState, RaceState, RaceMode, TrackCondition
from src.simulation.tyres import CompoundType, TyreModel
from src.simulation.pit_model import PitStopModel, PitStopDurationResult
from src.simulation.events import EventDispatcher, EventType, SimulationEvent


@dataclass
class StintPlan:
    """Pre-race planned sequence of tyre compounds and stint lap lengths."""

    compounds: List[CompoundType]
    stint_lengths: List[int]  # Target laps per stint

    @property
    def total_stops(self) -> int:
        return max(0, len(self.compounds) - 1)

    def get_target_pit_laps(self) -> List[int]:
        """Convert stint lengths into cumulative race pit laps."""
        pit_laps = []
        acc = 0
        for l in self.stint_lengths[:-1]:
            acc += l
            pit_laps.append(acc)
        return pit_laps


class StrategyEngine:
    """Evaluates in-race strategic pit stop decisions and manages tyre lifecycles."""

    def __init__(
        self,
        pit_model: Optional[PitStopModel] = None,
        tyre_model: Optional[TyreModel] = None,
    ):
        self.pit_model = pit_model or PitStopModel()
        self.tyre_model = tyre_model or TyreModel()

    def generate_default_stint_plan(
        self,
        total_laps: int,
        grid_position: int,
        circuit_tyre_stress: float = 3.0,
    ) -> StintPlan:
        """Generate a realistic baseline strategy complying with FIA Article 30.5.
        
        High stress circuits (> 3.5) trigger a 2-stop strategy; otherwise optimal 1-stop.
        """
        if circuit_tyre_stress > 3.8:
            # 2-Stop: Soft -> Hard -> Medium or Medium -> Hard -> Medium
            stint1 = max(12, int(total_laps * 0.28))
            stint2 = max(15, int(total_laps * 0.38))
            stint3 = total_laps - (stint1 + stint2)
            return StintPlan(
                compounds=[CompoundType.MEDIUM, CompoundType.HARD, CompoundType.SOFT],
                stint_lengths=[stint1, stint2, stint3],
            )
        else:
            # Standard 1-Stop: Medium -> Hard (or alternate Hard -> Medium for back half of grid)
            if grid_position <= 10:
                # Top 10 prefer Medium -> Hard
                stint1 = max(18, int(total_laps * 0.42))
                stint2 = total_laps - stint1
                return StintPlan(
                    compounds=[CompoundType.MEDIUM, CompoundType.HARD],
                    stint_lengths=[stint1, stint2],
                )
            else:
                # Alternative strategy for midfield/backmarkers: Hard -> Medium
                stint1 = max(24, int(total_laps * 0.55))
                stint2 = total_laps - stint1
                return StintPlan(
                    compounds=[CompoundType.HARD, CompoundType.MEDIUM],
                    stint_lengths=[stint1, stint2],
                )

    def should_pit_now(
        self,
        car: CarState,
        race: RaceState,
        stint_plan: StintPlan,
        stint_deg_rate: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> bool:
        """Evaluate whether a driver should enter the pit lane on the current lap.
        
        Considers:
        1. Planned stint expiry.
        2. Thermal degradation cliff breach.
        3. Opportunistic SC / VSC cheap pit stop.
        4. Traffic congestion on potential pit exit.
        5. Minimum remaining laps in race.
        """
        laps_remaining = race.total_laps - race.current_lap
        if laps_remaining <= 2:
            # Do not pit with 1 or 2 laps remaining unless puncture
            return False

        current_compound = CompoundType(car.current_compound)
        profile = self.tyre_model.get_profile(current_compound)

        # 1. Thermal Cliff Breach: Lap time degradation > 2.2s
        tyre_delta = self.tyre_model.compute_tyre_pace_delta(
            compound=current_compound,
            tyre_age=car.tyre_age,
            track_condition=race.track_condition,
            stint_deg_rate=stint_deg_rate,
        )
        if tyre_delta >= 2.20 and car.tyre_age >= 10:
            return True

        # 2. Opportunistic Safety Car / VSC Pit Stop:
        # Under SC/VSC, pit loss is halved. If tyres have >= 12 laps of wear and race is neutralised:
        is_neutralised = race.race_mode in (RaceMode.SAFETY_CAR, RaceMode.VSC)
        if is_neutralised and car.tyre_age >= max(10, int(profile.cliff_lap * 0.45)):
            # Cheap pit stop saves ~10s of track delta!
            return True

        # 3. Scheduled Pit Lap from Stint Plan
        target_pit_laps = stint_plan.get_target_pit_laps()
        if car.pit_stops_count < len(target_pit_laps):
            planned_lap = target_pit_laps[car.pit_stops_count]
            # Pit window: +/- 1 lap around planned lap
            if race.current_lap >= planned_lap:
                # 4. Traffic check: Avoid exiting directly into heavy traffic if tyre is still healthy
                # If pitting would drop the car behind a dense pack and tyres have not hit cliff, can extend by 1 lap
                if car.tyre_age < profile.cliff_lap and race.current_lap == planned_lap:
                    # Check if gap behind allows clean pit exit
                    pit_loss_approx = self.pit_model.get_circuit_transit_loss(race.circuit_id) + 2.5
                    reentry_time = car.elapsed_time + pit_loss_approx
                    # Check if re-entry is too close to another car (< 0.6s)
                    traffic_density = sum(
                        1 for other in race.cars.values()
                        if other != car and abs(other.elapsed_time - reentry_time) < 0.6
                    )
                    if traffic_density >= 2:
                        # Heavy traffic ahead; extend stint by 1 lap to seek clean air
                        return False
                return True

        return False

    def execute_pit_stop(
        self,
        car: CarState,
        race: RaceState,
        stint_plan: StintPlan,
        dispatcher: Optional[EventDispatcher] = None,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> PitStopDurationResult:
        """Execute the pit stop service, update car state, and log simulation event."""
        is_sc_vsc = race.race_mode in (RaceMode.SAFETY_CAR, RaceMode.VSC)
        duration_res = self.pit_model.sample_pit_duration(
            circuit_id=race.circuit_id,
            is_sc_vsc=is_sc_vsc,
            rng=rng,
            deterministic=deterministic,
        )

        old_compound = car.current_compound
        old_tyre_age = car.tyre_age

        # Determine next compound from stint plan (or fallback to opposite dry compound)
        next_stop_idx = car.pit_stops_count + 1
        if next_stop_idx < len(stint_plan.compounds):
            next_compound = stint_plan.compounds[next_stop_idx].value
        else:
            # Fallback for extra unplanned stops: switch between Medium and Hard
            next_compound = "hard" if old_compound == "medium" else "medium"

        # Apply pit stop updates to CarState
        car.elapsed_time += duration_res.total_loss_s
        car.pit_stops_count += 1
        car.current_compound = next_compound
        car.compounds_used.add(next_compound)
        car.tyre_age = 0  # Fresh tyres

        if dispatcher is not None:
            dispatcher.emit(
                SimulationEvent(
                    event_type=EventType.PIT_STOP,
                    lap=race.current_lap,
                    driver_id=car.driver_id,
                    constructor_id=car.constructor_id,
                    track_position=car.track_position,
                    elapsed_time=car.elapsed_time,
                    details={
                        "pit_loss_s": duration_res.total_loss_s,
                        "stationary_s": duration_res.stationary_service_s,
                        "abnormal_delay_s": duration_res.abnormal_delay_s,
                        "is_abnormal": duration_res.is_abnormal_delay,
                        "is_under_sc_vsc": is_sc_vsc,
                        "old_compound": old_compound,
                        "new_compound": next_compound,
                        "tyre_age_at_stop": old_tyre_age,
                    },
                )
            )

        return duration_res
