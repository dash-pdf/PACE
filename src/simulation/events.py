"""PACE: Simulation Event System Module.

Defines the event-driven notifications and log entries during a Grand Prix simulation:
1. EventType: Lap completions, overtakes, pit stops, incidents, mechanical failures,
   Safety Car deployments, and flag status transitions.
2. SimulationEvent: Structured event record with timestamp and context.
3. EventDispatcher: Collects and dispatches events to listeners and the race log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    LAP_COMPLETED = "lap_completed"
    OVERTAKE = "overtake"
    LAPPED = "lapped"
    PIT_STOP = "pit_stop"
    INCIDENT = "incident"
    MECHANICAL_FAILURE = "mechanical_failure"
    SAFETY_CAR_DEPLOYED = "safety_car_deployed"
    SAFETY_CAR_ENDED = "safety_car_ended"
    VSC_DEPLOYED = "vsc_deployed"
    VSC_ENDED = "vsc_ended"
    RED_FLAG_DEPLOYED = "red_flag_deployed"
    RED_FLAG_ENDED = "red_flag_ended"
    WEATHER_CHANGE = "weather_change"
    DEBRIS_WARNING = "debris_warning"
    FLAG_CHANGE = "flag_change"
    PENALTY_ISSUED = "penalty_issued"
    RACE_FINISHED = "race_finished"


@dataclass
class SimulationEvent:
    """Structured record of an event occurring in the race."""

    event_type: EventType
    lap: int
    driver_id: Optional[str] = None
    constructor_id: Optional[str] = None
    track_position: Optional[int] = None
    elapsed_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "lap": self.lap,
            "driver_id": self.driver_id,
            "constructor_id": self.constructor_id,
            "track_position": self.track_position,
            "elapsed_time": round(self.elapsed_time, 3),
            "details": self.details,
        }


class EventDispatcher:
    """Manages event logging and listener notifications during a race simulation."""

    def __init__(self):
        self.events: List[SimulationEvent] = []
        self._listeners: Dict[EventType, List[Callable[[SimulationEvent], None]]] = {
            t: [] for t in EventType
        }

    def subscribe(self, event_type: EventType, callback: Callable[[SimulationEvent], None]) -> None:
        """Subscribe a listener callback to an event type."""
        self._listeners[event_type].append(callback)

    def emit(self, event: SimulationEvent) -> None:
        """Record an event and notify registered listeners."""
        self.events.append(event)
        for callback in self._listeners.get(event.event_type, []):
            callback(event)

    def get_events_for_lap(self, lap: int) -> List[SimulationEvent]:
        """Return all events recorded during a specific lap."""
        return [e for e in self.events if e.lap == lap]

    def get_events_for_driver(self, driver_id: str) -> List[SimulationEvent]:
        """Return all events associated with a specific driver."""
        return [e for e in self.events if e.driver_id == driver_id]

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()
