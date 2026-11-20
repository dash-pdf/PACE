"""PACE: Formula 1 Stochastic Race Simulation Package.

Submodules:
- state: CarState, RaceState, CarStatus, RaceMode, TrackCondition
- events: EventDispatcher, SimulationEvent, EventType
- tyres: TyreModel, CompoundType, CompoundProfile, COMPOUND_PROFILES
- fuel: FuelModel
- pit_model: PitStopModel, PitStopDurationResult
- strategy: StrategyEngine, StintPlan
- pace_model: StochasticPaceModel, BaselinePaceModel
- rules: FIAClassificationEngine
- engine: BaselineRaceSimulator, SimulationResult
"""

from src.simulation.state import CarState, RaceState, CarStatus, RaceMode, TrackCondition
from src.simulation.events import EventDispatcher, SimulationEvent, EventType
from src.simulation.tyres import TyreModel, CompoundType, CompoundProfile, COMPOUND_PROFILES
from src.simulation.fuel import FuelModel
from src.simulation.pit_model import PitStopModel, PitStopDurationResult
from src.simulation.strategy import StrategyEngine, StintPlan
from src.simulation.pace_model import StochasticPaceModel, BaselinePaceModel
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
    "TyreModel",
    "CompoundType",
    "CompoundProfile",
    "COMPOUND_PROFILES",
    "FuelModel",
    "PitStopModel",
    "PitStopDurationResult",
    "StrategyEngine",
    "StintPlan",
    "StochasticPaceModel",
    "BaselinePaceModel",
    "FIAClassificationEngine",
    "BaselineRaceSimulator",
    "SimulationResult",
]
