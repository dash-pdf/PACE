"""PACE: Safety Car, VSC & Red Flag Neutralisation Engine (Stage 5).

Models:
1. Neutralisation state transitions: GREEN -> VSC -> SAFETY_CAR -> RED_FLAG -> GREEN.
2. Collective field bunching and interval compression under physical Safety Car.
3. Proportional delta time pacing under Virtual Safety Car.
4. Restart dynamics: pace control, unlapping, DRS disabling for 2 laps post-restart.
5. Red Flag race suspension, free tyre changes, and grid re-alignment.
6. Opportunistic cheap pit stop evaluation under course neutralisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from src.simulation.state import CarState, CarStatus, RaceMode, RaceState
from src.simulation.events import EventDispatcher, EventType, SimulationEvent
from src.simulation.incidents import NeutralisationTrigger


class NeutralisationEngine:
    """Manages course neutralisation lifecycles, field compression, and restart mechanics."""

    def __init__(
        self,
        vsc_base_duration_laps: int = 2,
        sc_base_duration_laps: int = 4,
    ):
        self.vsc_duration = vsc_base_duration_laps
        self.sc_duration = sc_base_duration_laps

    def trigger_neutralisation(
        self,
        trigger: NeutralisationTrigger,
        race: RaceState,
        cause: str = "incident",
        dispatcher: Optional[EventDispatcher] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> bool:
        """Initiate course neutralisation based on incident or weather severity."""
        if trigger in (NeutralisationTrigger.NONE, NeutralisationTrigger.LOCAL_YELLOW):
            return False

        # Red Flag takes highest priority
        if trigger == NeutralisationTrigger.RED_FLAG:
            race.race_mode = RaceMode.RED_FLAG
            race.red_flag_laps_remaining = 2
            race.sc_laps_remaining = 0
            race.neutralisation_cause = cause
            if dispatcher:
                dispatcher.emit(
                    SimulationEvent(
                        event_type=EventType.RED_FLAG_DEPLOYED,
                        lap=race.current_lap,
                        elapsed_time=0.0,
                        details={"cause": cause, "status": "race_suspended"},
                    )
                )
            return True

        # Safety Car takes precedence over VSC
        if trigger == NeutralisationTrigger.SAFETY_CAR or (race.race_mode == RaceMode.VSC and trigger != NeutralisationTrigger.NONE):
            race.race_mode = RaceMode.SAFETY_CAR
            # Sample 3 to 5 laps for SC period
            laps = int(rng.randint(3, 6)) if rng is not None else self.sc_duration
            race.sc_laps_remaining = laps
            race.neutralisation_cause = cause
            if dispatcher:
                dispatcher.emit(
                    SimulationEvent(
                        event_type=EventType.SAFETY_CAR_DEPLOYED,
                        lap=race.current_lap,
                        elapsed_time=0.0,
                        details={"cause": cause, "scheduled_duration_laps": laps},
                    )
                )
            return True

        # Virtual Safety Car
        if trigger == NeutralisationTrigger.VSC and race.race_mode == RaceMode.GREEN:
            race.race_mode = RaceMode.VSC
            laps = int(rng.randint(1, 3)) if rng is not None else self.vsc_duration
            race.sc_laps_remaining = laps
            race.neutralisation_cause = cause
            if dispatcher:
                dispatcher.emit(
                    SimulationEvent(
                        event_type=EventType.VSC_DEPLOYED,
                        lap=race.current_lap,
                        elapsed_time=0.0,
                        details={"cause": cause, "scheduled_duration_laps": laps},
                    )
                )
            return True

        return False

    def compress_field_under_safety_car(
        self,
        race: RaceState,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        """Compress the entire active field behind the physical Safety Car.

        The physical Safety Car paces the leader at ~150% base lap time.
        Trailing cars catch up and compress into a tight train with gaps ~0.65s - 1.25s.
        """
        ordered_active = [c for c in race.get_ordered_field() if c.status.is_active]
        if len(ordered_active) <= 1:
            return

        leader = ordered_active[0]

        # Trailing cars compress into queue
        current_time = leader.elapsed_time
        for pos_idx, car in enumerate(ordered_active[1:], 2):
            gap = float(rng.uniform(0.65, 1.25)) if rng is not None else 0.85
            current_time += gap
            car.elapsed_time = current_time

        race.update_running_positions_and_gaps()

    def process_neutralisation_step(
        self,
        race: RaceState,
        dispatcher: Optional[EventDispatcher] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        """Advance neutralisation counter, manage field bunching, and handle green flag restart."""
        # 1. Handle Red Flag suspension
        if race.race_mode == RaceMode.RED_FLAG:
            race.red_flag_laps_remaining -= 1
            if race.red_flag_laps_remaining <= 0:
                # Red Flag concludes: Resume under Safety Car
                race.race_mode = RaceMode.SAFETY_CAR
                race.sc_laps_remaining = 2
                race.active_debris_sectors.clear()
                if dispatcher:
                    dispatcher.emit(
                        SimulationEvent(
                            event_type=EventType.RED_FLAG_ENDED,
                            lap=race.current_lap,
                            details={"status": "resumed_behind_safety_car"},
                        )
                    )
            return

        # 2. Handle Safety Car / VSC active lap countdown
        if race.race_mode == RaceMode.SAFETY_CAR:
            # Compress the field collectively on every lap behind SC
            self.compress_field_under_safety_car(race, rng=rng)
            race.sc_laps_remaining -= 1

            if race.sc_laps_remaining <= 0:
                # Safety car in this lap!
                race.race_mode = RaceMode.GREEN
                race.sc_restart_lap = race.current_lap
                race.active_debris_sectors.clear()  # Marshals cleared track
                if dispatcher:
                    dispatcher.emit(
                        SimulationEvent(
                            event_type=EventType.SAFETY_CAR_ENDED,
                            lap=race.current_lap,
                            details={"status": "green_flag_restart", "restart_lap": race.current_lap},
                        )
                    )

        elif race.race_mode == RaceMode.VSC:
            race.sc_laps_remaining -= 1
            if race.sc_laps_remaining <= 0:
                race.race_mode = RaceMode.GREEN
                race.active_debris_sectors.clear()
                if dispatcher:
                    dispatcher.emit(
                        SimulationEvent(
                            event_type=EventType.VSC_ENDED,
                            lap=race.current_lap,
                            details={"status": "vsc_ended"},
                        )
                    )
