"""PACE: Lap Pace & Physical Decomposition Model (Stage 3).

Computes single-lap pace combining:
1. Phase 3 Ridge / Bayesian LMM intrinsic speed offset (Delta t_{driver/car}).
2. Latent fuel mass depletion over race distance (Delta t_{fuel}(L)).
3. Tyre compound speed offset, linear wear, and thermal degradation cliff (Delta t_{tyre}(C, a)).
4. Dirty air wake downforce penalty for trailing cars (Delta t_{wake}).
5. Stochastic driver execution noise scaled by empirical consistency CV (epsilon_{i,L}).
6. Course flag neutralizations (VSC, Safety Car).
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from src.simulation.state import CarState, RaceState, RaceMode, TrackCondition
from src.simulation.tyres import TyreModel, CompoundType
from src.simulation.fuel import FuelModel


class StochasticPaceModel:
    """Computes realistic lap times combining Phase 3 baseline pace, tyres, fuel, and wake."""

    def __init__(
        self,
        utility_scaling_factor: float = 0.05,
        tyre_model: Optional[TyreModel] = None,
        fuel_model: Optional[FuelModel] = None,
    ):
        """Args:
            utility_scaling_factor: Seconds gained per unit of Phase 3 utility.
            tyre_model: TyreModel instance managing compound and degradation physics.
            fuel_model: FuelModel instance managing latent vehicle mass depletion.
        """
        self.utility_scaling = utility_scaling_factor
        self.tyre_model = tyre_model or TyreModel()
        self.fuel_model = fuel_model or FuelModel()

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
        stint_deg_rate: Optional[float] = None,
        fuel_sensitivity: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> float:
        """Compute the lap time in seconds for car on the current lap."""
        if rng is None:
            rng = np.random.RandomState(42)

        t_base = race.base_lap_time

        # 1. Course Flag Neutralizations override normal racing pace
        if race.race_mode == RaceMode.SAFETY_CAR:
            # Full Safety Car lap time (approx 145% of base pace)
            return t_base * 1.45 + (car.track_position * 0.01)
        elif race.race_mode == RaceMode.VSC:
            # Virtual Safety Car delta (approx 135% of base pace)
            return t_base * 1.35 + (rng.normal(0.0, 0.04) if not deterministic else 0.0)

        # 2. Phase 3 Intrinsic Speed Delta
        delta_intrinsic = self.compute_base_pace_delta(driver_utility, mean_utility)

        # 3. Latent Fuel Mass Depletion (heavier at start, lighter at finish)
        delta_fuel = self.fuel_model.compute_fuel_pace_delta(
            lap=race.current_lap,
            total_laps=race.total_laps,
            sensitivity=fuel_sensitivity,
        )

        # 4. Traffic Wake / Dirty Air Factor
        # Following within 1.5s increases tyre degradation by 25% and incurs downforce loss
        wake_factor = 1.0
        delta_wake = 0.0
        if 0.0 < car.gap_to_car_ahead <= 1.5:
            # Trailing closely in wake: +0.25s aero loss, +25% tyre degradation
            delta_wake = 0.25 * (1.0 - (car.gap_to_car_ahead / 1.5))
            wake_factor = 1.25

        # 5. Tyre Compound & Degradation Delta
        delta_tyre = self.tyre_model.compute_tyre_pace_delta(
            compound=car.current_compound,
            tyre_age=car.tyre_age,
            track_condition=race.track_condition,
            stint_deg_rate=stint_deg_rate,
            traffic_wake_factor=wake_factor,
        )

        # 6. Stochastic Driver Execution Noise
        if deterministic:
            noise = 0.0
        else:
            sigma_pace = max(0.05, min(consistency_cv * t_base, 0.60))
            noise = float(rng.normal(0.0, sigma_pace))

        # 7. Physical Damage Penalty
        damage_penalty = car.damage_factor

        # Combined net lap time
        lap_time = t_base + delta_intrinsic + delta_fuel + delta_tyre + delta_wake + damage_penalty + noise

        # Physical lower bound check
        return float(max(t_base * 0.75, lap_time))


# Alias for backward compatibility
BaselinePaceModel = StochasticPaceModel
