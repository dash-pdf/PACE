"""PACE: Simulation State Representation Module.

Defines the multi-agent state space for the Formula 1 Stochastic Race Simulation Engine:
1. CarStatus: Standard classification states (Active, Finished, DNF_Mechanical, DNF_Incident, DNS, DSQ).
2. CarState: Dynamic, mutable state for each car on the grid (track position, elapsed race time,
   gap to car ahead, tyre age, compound, damage factor, penalties).
3. TrackCondition & RaceMode: Global course conditions (Green, VSC, Safety Car, Red Flag).
4. RaceState: Global state of the Grand Prix at any discrete lap L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import numpy as np


class CarStatus(str, Enum):
    ACTIVE = "active"
    FINISHED = "finished"
    DNF_MECHANICAL = "dnf_mechanical"
    DNF_INCIDENT = "dnf_incident"
    DNS = "dns"
    DSQ = "dsq"

    @property
    def is_dnf(self) -> bool:
        return self in (CarStatus.DNF_MECHANICAL, CarStatus.DNF_INCIDENT)

    @property
    def is_active(self) -> bool:
        return self == CarStatus.ACTIVE


class TrackCondition(str, Enum):
    DRY = "dry"
    DAMP = "damp"
    WET = "wet"


class RaceMode(str, Enum):
    GREEN = "green"
    YELLOW_LOCAL = "yellow_local"
    VSC = "vsc"
    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"


@dataclass
class CarState:
    """Dynamic state of an individual car/driver at lap L."""

    driver_id: str
    constructor_id: str
    grid_position: int
    track_position: int
    elapsed_time: float = 0.0  # Cumulative race elapsed time in seconds
    gap_to_leader: float = 0.0  # Seconds behind P1
    gap_to_car_ahead: float = 0.0  # Seconds behind car at track_position - 1
    laps_completed: int = 0
    current_compound: str = "medium"
    tyre_age: int = 0  # Laps run on current tyre set
    compounds_used: Set[str] = field(default_factory=lambda: {"medium"})
    pit_stops_count: int = 0
    in_pit_lane: bool = False
    damage_factor: float = 0.0  # Aerodynamic / mechanical damage penalty in s/lap
    penalties_accrued: float = 0.0  # Post-race or pit-served time penalty in seconds
    status: CarStatus = CarStatus.ACTIVE
    retirement_lap: Optional[int] = None
    last_lap_time: float = 0.0
    lap_times_history: List[float] = field(default_factory=list)
    positions_history: List[int] = field(default_factory=list)

    def record_lap(self, lap_time: float, current_lap: int) -> None:
        """Record completed lap time and update elapsed time."""
        self.last_lap_time = lap_time
        self.elapsed_time += lap_time
        self.laps_completed += 1
        self.tyre_age += 1
        self.lap_times_history.append(lap_time)
        self.positions_history.append(self.track_position)

    def retire(self, lap: int, reason: CarStatus) -> None:
        """Retire vehicle from race."""
        self.status = reason
        self.retirement_lap = lap


@dataclass
class RaceState:
    """Global state of the Grand Prix event across all 22 cars."""

    current_lap: int
    total_laps: int
    circuit_id: str
    circuit_length_km: float
    base_lap_time: float  # Circuit reference lap time in seconds
    track_condition: TrackCondition = TrackCondition.DRY
    track_temperature_c: float = 30.0
    race_mode: RaceMode = RaceMode.GREEN
    sc_laps_remaining: int = 0
    cars: Dict[str, CarState] = field(default_factory=dict)
    leader_id: Optional[str] = None
    incident_log: List[dict] = field(default_factory=list)

    def get_ordered_field(self) -> List[CarState]:
        """Return cars sorted in current track running order.
        
        Sorting rule:
        1. Higher laps completed is ahead.
        2. If equal laps completed, lower elapsed race time is ahead.
        3. Active cars take precedence over unclassified retirements.
        """
        def sort_key(car: CarState):
            # Active or Finished cars sorted by completed laps (desc) then elapsed time (asc)
            is_active_or_finished = car.status in (CarStatus.ACTIVE, CarStatus.FINISHED)
            return (
                0 if is_active_or_finished else 1,
                -car.laps_completed,
                car.elapsed_time + car.penalties_accrued,
            )

        return sorted(self.cars.values(), key=sort_key)

    def update_running_positions_and_gaps(self) -> None:
        """Recalculate track positions and gaps to leader and car ahead."""
        ordered = self.get_ordered_field()
        if not ordered:
            return

        leader = ordered[0]
        self.leader_id = leader.driver_id

        for pos_idx, car in enumerate(ordered, 1):
            car.track_position = pos_idx
            car.gap_to_leader = max(0.0, car.elapsed_time - leader.elapsed_time)

            if pos_idx == 1:
                car.gap_to_car_ahead = 0.0
            else:
                car_ahead = ordered[pos_idx - 2]
                car.gap_to_car_ahead = max(0.0, car.elapsed_time - car_ahead.elapsed_time)
