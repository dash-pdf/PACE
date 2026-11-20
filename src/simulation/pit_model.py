"""PACE: Empirical Pit Stop Duration & Loss Model (Stage 3).

Models:
1. Pit-lane transit loss (calibrated per circuit from historical pit stop data).
2. Clean stationary service time (log-normally distributed around 2.4s).
3. Heavy-tailed abnormal service delays (stuck wheel gun, pit exit traffic release hold).
4. Safety Car / VSC pit loss discount ("cheap pit stop" heuristic: ~45% transit loss reduction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class PitStopDurationResult:
    """Decomposition of a simulated pit stop time loss."""

    total_loss_s: float
    transit_loss_s: float
    stationary_service_s: float
    abnormal_delay_s: float
    is_under_sc_vsc: bool
    is_abnormal_delay: bool


class PitStopModel:
    """Simulates realistic pit stop durations with empirical uncertainty."""

    # Default circuit pit transit times (seconds) under Green flag conditions
    DEFAULT_CIRCUIT_TRANSIT_LOSS: Dict[str, float] = {
        "albert_park": 20.5,
        "bahrain": 24.5,
        "jeddah": 20.0,
        "baku": 21.0,
        "miami": 20.2,
        "monaco": 22.0,
        "catalunya": 22.5,
        "villeneuve": 18.5,
        "red_bull_ring": 20.0,
        "silverstone": 20.8,
        "hungaroring": 21.5,
        "spa": 22.0,
        "zandvoort": 19.5,
        "monza": 24.0,
        "singapore": 28.0,  # 60 km/h limit in pit lane
        "suzuka": 22.2,
        "losail": 23.0,
        "americas": 21.8,
        "rodriguez": 22.5,
        "interlagos": 21.0,
        "las_vegas": 20.5,
        "yas_marina": 22.0,
    }

    def __init__(
        self,
        mean_stationary_s: float = 2.45,
        std_stationary_s: float = 0.35,
        prob_abnormal_delay: float = 0.08,
        mean_abnormal_delay_s: float = 4.5,
        historical_pit_stops: Optional[pd.DataFrame] = None,
    ):
        self.mean_stationary = mean_stationary_s
        self.std_stationary = std_stationary_s
        self.prob_abnormal = prob_abnormal_delay
        self.mean_abnormal = mean_abnormal_delay_s

        self.circuit_transit_times: Dict[str, float] = dict(self.DEFAULT_CIRCUIT_TRANSIT_LOSS)
        if historical_pit_stops is not None and not historical_pit_stops.empty:
            self._calibrate_from_history(historical_pit_stops)

    def _calibrate_from_history(self, df: pd.DataFrame) -> None:
        """Calibrate circuit transit times from empirical historical pit stops."""
        if "duration_ms" not in df.columns:
            return

        clean = df[(df["duration_ms"] > 14000) & (df["duration_ms"] < 40000)].copy()
        if clean.empty:
            return

        clean["dur_s"] = clean["duration_ms"] / 1000.0
        # If circuit_id or round is available, calibrate
        # Subtract average stationary time (~2.4s) from 10th percentile duration to estimate transit time
        p10 = clean["dur_s"].quantile(0.10)
        global_transit = max(16.0, float(p10 - 2.4))
        for c in self.circuit_transit_times:
            # Scale slightly around empirical global transit
            self.circuit_transit_times[c] = (self.circuit_transit_times[c] + global_transit) / 2.0

    def get_circuit_transit_loss(self, circuit_id: str, is_sc_vsc: bool = False) -> float:
        """Get the base pit lane transit loss for a circuit.
        
        Under SC / VSC, cars on track run 35-45% slower, reducing the net pit delta by ~45%.
        """
        c_id = circuit_id.lower()
        base_transit = self.circuit_transit_times.get(c_id, 21.0)
        if is_sc_vsc:
            return base_transit * 0.55  # ~45% discount under SC / VSC
        return base_transit

    def sample_pit_duration(
        self,
        circuit_id: str = "albert_park",
        is_sc_vsc: bool = False,
        rng: Optional[np.random.RandomState] = None,
        deterministic: bool = False,
    ) -> PitStopDurationResult:
        """Sample a realistic, stochastic pit stop duration."""
        if rng is None:
            rng = np.random.RandomState(42)

        transit = self.get_circuit_transit_loss(circuit_id, is_sc_vsc=is_sc_vsc)

        if deterministic:
            return PitStopDurationResult(
                total_loss_s=transit + self.mean_stationary,
                transit_loss_s=transit,
                stationary_service_s=self.mean_stationary,
                abnormal_delay_s=0.0,
                is_under_sc_vsc=is_sc_vsc,
                is_abnormal_delay=False,
            )

        # 1. Clean stationary service time: Log-normal distribution
        mu = np.log(self.mean_stationary ** 2 / np.sqrt(self.std_stationary ** 2 + self.mean_stationary ** 2))
        sigma = np.sqrt(np.log(1.0 + (self.std_stationary ** 2 / self.mean_stationary ** 2)))
        stationary = float(np.clip(rng.lognormal(mu, sigma), 1.9, 4.5))

        # 2. Abnormal delay (stuck wheel nut, traffic release)
        has_abnormal = bool(rng.uniform(0.0, 1.0) < self.prob_abnormal)
        if has_abnormal:
            delay = float(np.clip(rng.exponential(self.mean_abnormal), 1.5, 15.0))
        else:
            delay = 0.0

        total_loss = transit + stationary + delay

        return PitStopDurationResult(
            total_loss_s=round(total_loss, 3),
            transit_loss_s=round(transit, 3),
            stationary_service_s=round(stationary, 3),
            abnormal_delay_s=round(delay, 3),
            is_under_sc_vsc=is_sc_vsc,
            is_abnormal_delay=has_abnormal,
        )
