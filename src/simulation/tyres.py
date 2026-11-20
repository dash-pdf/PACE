"""PACE: Tyre Compound & Degradation Physics Module (Stage 3).

Models:
1. Dry compounds (Soft, Medium, Hard) and Wet compounds (Intermediate, Wet).
2. Base compound speed offsets (Soft fastest, Hard most durable).
3. Compound-specific linear wear and non-linear thermal degradation cliffs.
4. Epistemic uncertainty in tyre degradation rates.
5. Track condition crossover effects (Damp / Wet surface penalties).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np

from src.simulation.state import TrackCondition


class CompoundType(str, Enum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"

    @property
    def is_dry(self) -> bool:
        return self in (CompoundType.SOFT, CompoundType.MEDIUM, CompoundType.HARD)

    @property
    def is_wet(self) -> bool:
        return self in (CompoundType.INTERMEDIATE, CompoundType.WET)


@dataclass(frozen=True)
class CompoundProfile:
    """Physical characteristics of a tyre compound specification."""

    name: str
    base_delta_s: float  # Lap time offset relative to Medium (seconds)
    deg_rate_mean: float  # Mean linear degradation rate (s/lap)
    deg_rate_std: float  # Uncertainty / variation in degradation rate (s/lap)
    cliff_lap: int  # Lap age at which thermal degradation non-linear cliff begins
    cliff_penalty_factor: float  # Quadratic penalty multiplier past cliff lap


# Standard Pirelli compound profiles calibrated from F1 historical stint data
COMPOUND_PROFILES: Dict[CompoundType, CompoundProfile] = {
    CompoundType.SOFT: CompoundProfile(
        name="soft",
        base_delta_s=-0.55,
        deg_rate_mean=0.095,
        deg_rate_std=0.015,
        cliff_lap=18,
        cliff_penalty_factor=0.018,
    ),
    CompoundType.MEDIUM: CompoundProfile(
        name="medium",
        base_delta_s=0.00,
        deg_rate_mean=0.060,
        deg_rate_std=0.010,
        cliff_lap=30,
        cliff_penalty_factor=0.014,
    ),
    CompoundType.HARD: CompoundProfile(
        name="hard",
        base_delta_s=+0.65,
        deg_rate_mean=0.035,
        deg_rate_std=0.008,
        cliff_lap=45,
        cliff_penalty_factor=0.012,
    ),
    CompoundType.INTERMEDIATE: CompoundProfile(
        name="intermediate",
        base_delta_s=+4.00,  # Slow on dry track, optimal on damp track
        deg_rate_mean=0.080,
        deg_rate_std=0.020,
        cliff_lap=25,
        cliff_penalty_factor=0.025,
    ),
    CompoundType.WET: CompoundProfile(
        name="wet",
        base_delta_s=+10.00,  # Very slow on dry track, optimal in standing water
        deg_rate_mean=0.090,
        deg_rate_std=0.025,
        cliff_lap=30,
        cliff_penalty_factor=0.030,
    ),
}


class TyreModel:
    """Calculates compound pace delta and non-linear degradation over stint age."""

    def __init__(self, profiles: Optional[Dict[CompoundType, CompoundProfile]] = None):
        self.profiles = profiles or COMPOUND_PROFILES

    def get_profile(self, compound: str | CompoundType) -> CompoundProfile:
        c_type = CompoundType(compound.lower()) if isinstance(compound, str) else compound
        return self.profiles.get(c_type, self.profiles[CompoundType.MEDIUM])

    def sample_stint_degradation_rate(
        self,
        compound: str | CompoundType,
        constructor_deg_multiplier: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Sample a stochastic degradation slope for a driver/car stint reflecting epistemic uncertainty."""
        prof = self.get_profile(compound)
        if rng is None:
            return prof.deg_rate_mean * constructor_deg_multiplier

        sampled = rng.normal(prof.deg_rate_mean, prof.deg_rate_std)
        # Bounded between 0.01 s/lap and 0.25 s/lap
        return float(np.clip(sampled * constructor_deg_multiplier, 0.01, 0.25))

    def compute_tyre_pace_delta(
        self,
        compound: str | CompoundType,
        tyre_age: int,
        track_condition: TrackCondition = TrackCondition.DRY,
        stint_deg_rate: Optional[float] = None,
        traffic_wake_factor: float = 1.0,
    ) -> float:
        """Calculate the total lap time delta from compound speed, wear, and weather mismatch.
        
        Args:
            compound: Active tyre compound.
            tyre_age: Laps completed on current set (0-indexed or 1-indexed).
            track_condition: Dry, Damp, or Wet surface condition.
            stint_deg_rate: Specific sampled degradation slope (s/lap) for this car/stint.
            traffic_wake_factor: Multiplier for tyre wear accelerated by following in dirty air (e.g. 1.25).
            
        Returns:
            delta_s: Additional seconds on lap time (positive is slower).
        """
        prof = self.get_profile(compound)
        deg_rate = stint_deg_rate if stint_deg_rate is not None else prof.deg_rate_mean

        # 1. Base compound speed delta relative to baseline Medium
        base_delta = prof.base_delta_s

        # 2. Linear tyre degradation over age
        effective_deg_rate = deg_rate * traffic_wake_factor
        linear_wear = effective_deg_rate * float(max(0, tyre_age))

        # 3. Non-linear thermal cliff (quadratic acceleration once tyre life exceeded)
        cliff_laps_exceeded = max(0, tyre_age - prof.cliff_lap)
        if cliff_laps_exceeded > 0:
            cliff_penalty = prof.cliff_penalty_factor * (cliff_laps_exceeded ** 2)
        else:
            cliff_penalty = 0.0

        # 4. Track condition crossover mismatch penalties
        condition_penalty = 0.0
        c_type = CompoundType(compound.lower()) if isinstance(compound, str) else compound

        if track_condition == TrackCondition.DRY:
            if c_type.is_wet:
                # Running wet tyres on dry track overheats rapidly
                condition_penalty = 5.0 + 0.3 * tyre_age
        elif track_condition == TrackCondition.DAMP:
            if c_type.is_dry:
                # Slicks on damp track lose grip
                condition_penalty = 3.50
            elif c_type == CompoundType.WET:
                condition_penalty = 1.80  # Wet tyre too grooved for damp track
            else:
                condition_penalty = -3.50  # Intermediate is optimal in damp
        elif track_condition == TrackCondition.WET:
            if c_type.is_dry:
                # Slicks on standing water: massive aquaplaning
                condition_penalty = 18.00
            elif c_type == CompoundType.INTERMEDIATE:
                condition_penalty = 4.00
            else:
                condition_penalty = -8.00  # Full wet is optimal in heavy rain

        return base_delta + linear_wear + cliff_penalty + condition_penalty
