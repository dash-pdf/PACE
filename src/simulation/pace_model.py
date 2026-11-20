"""PACE: Lap Pace & Execution Model (Stage 2 Baseline).

Anchors baseline speed directly on Phase 3 model outputs:
1. Translates Phase 3 Ridge / Bayesian LMM utilities into physical lap time offsets:
   Delta t_{driver/car} = -0.05 * (Utility_i - mean(Utility)).
2. Scales stochastic execution noise via empirical driver consistency CV:
   sigma_{pace} = consistency_cv_i * t_{base}.
3. Implements course flag pace limits (VSC delta: +35%, Full SC delta: +45%).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from src.simulation.state import CarState, RaceState, RaceMode


class BaselinePaceModel:
    """Computes single-lap pace anchored directly on Phase 3 performance indices."""

    def __init__(self, utility_scaling_factor: float = 0.05):
        """Args:
            utility_scaling_factor: Seconds gained per unit of Phase 3 utility score.
                                    Default 0.05 s/lap per utility unit.
        """
        self.utility_scaling = utility_scaling_factor

    def compute_base_pace_delta(self, driver_utility: float, mean_utility: float) -> float:
        """Calculate intrinsic vehicle/driver lap time delta (negative is faster)."""
        return -self.utility_scaling * (driver_utility - mean_utility)

    def compute_lap_time(
        self,
        car: CarState,
        race: RaceState,
        driver_utility: float,
        mean_utility: float,
        consistency_cv: float = 0.003,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> float:
        """Compute the lap time in seconds for car on the current lap.
        
        Args:
            car: CarState instance.
            race: RaceState global environment.
            driver_utility: Phase 3 utility score for this driver.
            mean_utility: Grid mean utility score.
            consistency_cv: Driver pace variation CV (from historical laps).
            rng: Configured PRNG stream.
            deterministic: If True, omit stochastic driver execution noise.
            
        Returns:
            lap_time: Completed lap time in seconds.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        t_base = race.base_lap_time

        # Flag neutralizations override normal racing pace
        if race.race_mode == RaceMode.SAFETY_CAR:
            # Full Safety Car lap time (approx 145% of base pace)
            return t_base * 1.45 + (car.track_position * 0.01)
        elif race.race_mode == RaceMode.VSC:
            # Virtual Safety Car delta (approx 135% of base pace)
            return t_base * 1.35 + (rng.normal(0.0, 0.05) if not deterministic else 0.0)

        # 1. Driver/Car intrinsic delta anchored on Phase 3 utility
        delta_intrinsic = self.compute_base_pace_delta(driver_utility, mean_utility)

        # 2. Stochastic driver execution noise
        if deterministic:
            noise = 0.0
        else:
            sigma_pace = max(0.05, min(consistency_cv * t_base, 0.60))
            noise = rng.normal(0.0, sigma_pace)

        # 3. Damage penalty (if any)
        damage_penalty = car.damage_factor

        # Net lap time
        lap_time = t_base + delta_intrinsic + damage_penalty + noise

        # Physical lower bound check (cannot lap faster than 80% of circuit base pace)
        return max(t_base * 0.80, lap_time)
