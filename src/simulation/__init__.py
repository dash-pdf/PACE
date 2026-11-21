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
from src.simulation.traffic import (
    TrafficOvertakingEngine,
    CircuitDRSConfig,
    CIRCUIT_DRS_PROFILES,
    OvertakeAttemptResult,
)
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
from src.simulation.rules import FIAClassificationEngine
from src.simulation.engine import BaselineRaceSimulator, SimulationResult
from src.simulation.monte_carlo import MonteCarloSimulationEngine, MonteCarloResult
from src.simulation.realism import SimulationRealismAuditor, HISTORICAL_F1_BENCHMARKS

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
    "TrafficOvertakingEngine",
    "CircuitDRSConfig",
    "CIRCUIT_DRS_PROFILES",
    "OvertakeAttemptResult",
    "ReliabilityEngine",
    "MechanicalSubsystem",
    "FailureCheckResult",
    "CONSTRUCTOR_BASELINE_RELIABILITY",
    "IncidentEngine",
    "IncidentSeverity",
    "NeutralisationTrigger",
    "IncidentResult",
    "NeutralisationEngine",
    "WeatherEngine",
    "WeatherEvolutionConfig",
    "StochasticPaceModel",
    "BaselinePaceModel",
    "FIAClassificationEngine",
    "BaselineRaceSimulator",
    "SimulationResult",
    "MonteCarloSimulationEngine",
    "MonteCarloResult",
    "SimulationRealismAuditor",
    "HISTORICAL_F1_BENCHMARKS",
]
