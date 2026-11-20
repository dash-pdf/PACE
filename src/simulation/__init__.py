"""PACE: Formula 1 Stochastic Race Simulation Package.

Submodules:
- state: CarState, RaceState, CarStatus, RaceMode, TrackCondition
- events: EventDispatcher, SimulationEvent, EventType
- pace_model: BaselinePaceModel
- rules: FIAClassificationEngine
- engine: BaselineRaceSimulator, SimulationResult
"""

from src.simulation.state import CarState, RaceState, CarStatus, RaceMode, TrackCondition
from src.simulation.events import EventDispatcher, SimulationEvent, EventType
from src.simulation.pace_model import BaselinePaceModel
from src.simulation.rules import FIAClassificationEngine
from src.simulation.engine import BaselineRaceSimulator, SimulationResult

__all__ = [
    "CarState",
    "RaceState",
    "CarStatus",
    "RaceMode",
    "TrackCondition",
    "EventDispatcher",
    "SimulationEvent",
    "EventType",
    "BaselinePaceModel",
    "FIAClassificationEngine",
    "BaselineRaceSimulator",
    "SimulationResult",
]
