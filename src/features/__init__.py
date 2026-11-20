"""PACE Feature Engineering Subpackage."""

from src.features.dpi import (
    Glicko2Engine,
    Glicko2Rating,
    calculate_teammate_qualifying_deltas,
    calculate_teammate_race_deltas,
    calculate_rookie_learning_factor,
    calculate_pace_consistency_cv,
    BradleyTerryEngine,
)
from src.features.cpi import (
    InferredTelemetryMetrics,
    LatentFuelStateEstimate,
    infer_stint_latent_fuel_state,
    calculate_circuit_fuel_sensitivity,
    estimate_tyre_degradation_huber,
    calculate_constructor_qualifying_pole_gap,
    calculate_constructor_reliability,
    calculate_constructor_rolling_trend,
)
from src.features.driver_circuit import (
    CircuitAffinityEstimate,
    EmpiricalBayesCircuitAffinity,
    extract_circuit_topology_features,
)
from src.features.pipeline import FeaturePipeline

__all__ = [
    "Glicko2Engine",
    "Glicko2Rating",
    "calculate_teammate_qualifying_deltas",
    "calculate_teammate_race_deltas",
    "calculate_rookie_learning_factor",
    "calculate_pace_consistency_cv",
    "BradleyTerryEngine",
    "InferredTelemetryMetrics",
    "LatentFuelStateEstimate",
    "infer_stint_latent_fuel_state",
    "calculate_circuit_fuel_sensitivity",
    "estimate_tyre_degradation_huber",
    "calculate_constructor_qualifying_pole_gap",
    "calculate_constructor_reliability",
    "calculate_constructor_rolling_trend",
    "CircuitAffinityEstimate",
    "EmpiricalBayesCircuitAffinity",
    "extract_circuit_topology_features",
    "FeaturePipeline",
]
