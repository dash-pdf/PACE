"""PACE: Latent Vehicle Fuel Depletion Model (Stage 3).

Models:
1. Linear fuel burn-off from race start (105 kg) to finish (2 kg scrutineering buffer).
2. Circuit-specific fuel sensitivity factor S_{fuel} (seconds per lap per 10 kg).
3. Inferred fuel mass and pace progression with uncertainty bounds.
4. Physical distinction between inferred fuel state and directly observed telemetry.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from src.features.cpi import calculate_circuit_fuel_sensitivity


class FuelModel:
    """Computes latent vehicle mass depletion and its lap time effect."""

    def __init__(
        self,
        initial_fuel_kg: float = 105.0,
        finish_fuel_kg: float = 2.0,
        circuit_fuel_sensitivity: Optional[float] = None,
        circuit_length_km: float = 5.0,
        tyre_stress_level: float = 3.0,
        overtaking_difficulty: float = 3.0,
    ):
        self.initial_fuel_kg = initial_fuel_kg
        self.finish_fuel_kg = finish_fuel_kg
        self.usable_fuel_kg = max(0.0, initial_fuel_kg - finish_fuel_kg)

        if circuit_fuel_sensitivity is not None:
            self.base_sensitivity = circuit_fuel_sensitivity
        else:
            self.base_sensitivity = calculate_circuit_fuel_sensitivity(
                circuit_length_km=circuit_length_km,
                tyre_stress_level=tyre_stress_level,
                overtaking_difficulty=overtaking_difficulty,
            )

    def sample_circuit_sensitivity(
        self,
        rng: Optional[np.random.RandomState] = None,
        uncertainty_std: float = 0.008,
    ) -> float:
        """Sample circuit fuel sensitivity reflecting epistemic uncertainty."""
        if rng is None:
            return self.base_sensitivity
        sampled = rng.normal(self.base_sensitivity, uncertainty_std)
        return float(np.clip(sampled, 0.020, 0.090))

    def get_remaining_fuel_kg(self, lap: int, total_laps: int) -> float:
        """Estimate remaining fuel mass in kg at lap L."""
        if total_laps <= 1:
            return self.finish_fuel_kg
        fraction_remaining = max(0.0, 1.0 - (float(lap - 1) / float(total_laps)))
        return self.finish_fuel_kg + fraction_remaining * self.usable_fuel_kg

    def compute_fuel_pace_delta(
        self,
        lap: int,
        total_laps: int,
        sensitivity: Optional[float] = None,
    ) -> float:
        """Compute the lap time penalty (seconds) from current fuel weight.
        
        Args:
            lap: Current race lap (1-indexed).
            total_laps: Total race distance in laps.
            sensitivity: Circuit fuel sensitivity (s per 10kg). If None, uses base.
            
        Returns:
            delta_s: Additional seconds on lap time due to fuel weight (positive is slower).
                     Starts high (+0.4 to +0.6s on Lap 1) and approaches ~0s at race finish.
        """
        sens = sensitivity if sensitivity is not None else self.base_sensitivity
        remaining_kg = self.get_remaining_fuel_kg(lap, total_laps)
        # Lap time penalty relative to empty car (0 kg):
        return float(sens * (remaining_kg / 10.0))
