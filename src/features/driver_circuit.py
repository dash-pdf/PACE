"""PACE: Driver-Circuit Historical Intelligence Feature Module.

Implements:
1. Empirical Bayes hierarchical shrinkage for track affinity estimation.
2. Explicit posterior variance propagation (SE of circuit affinity).
3. 4 competing temporal weighting hypotheses evaluators:
   - H1: Continuous Exponential Half-Life (decaying over time)
   - H2: Discrete Era Step-Down (regulation eras)
   - H3: Hierarchical Random Walk
   - H4: Era-Invariant Uniform Weighting
4. Circuit topology affinity vectors (downforce compatibility, overtaking, tyre stress).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.canonical_registry import resolve_circuit


@dataclass
class CircuitAffinityEstimate:
    """Carries driver circuit affinity point estimate and uncertainty."""
    driver_id: str
    circuit_id: str
    mean_finish: float              # Observed sample average finishing position
    shrinkage_mean: float           # Empirical Bayes shrinkage estimate
    sample_size: float              # Effective starts count
    shrinkage_weight: float         # Alpha weight on circuit sample vs global career
    uncertainty: float              # Posterior standard error


class EmpiricalBayesCircuitAffinity:
    """Computes Empirical Bayes shrinkage for driver performance at specific circuits."""

    def __init__(self, kappa_0: float = 5.0, default_prior_std: float = 4.5):
        self.kappa_0 = kappa_0  # Prior precision parameter (equivalent starts of prior belief)
        self.default_prior_std = default_prior_std

    def compute_affinity(
        self,
        appearances_df: pd.DataFrame,
        driver_id: str,
        circuit_id: str,
        target_season: int,
        target_round: int,
        hypothesis: str = "H1",  # "H1", "H2", "H3", "H4"
        half_life_years: float = 3.0,
    ) -> CircuitAffinityEstimate:
        """Compute circuit affinity strictly using historical data prior to target_season/round."""
        # 1. Filter prior history strictly prior to target event
        prior_history = appearances_df[
            (appearances_df["season"] < target_season) |
            ((appearances_df["season"] == target_season) & (appearances_df["round"] < target_round))
        ].copy()

        # 2. Driver career baseline
        driver_prior = prior_history[prior_history["driver_id"] == driver_id]
        if driver_prior.empty:
            # Rookie with no career history: default to field average ~11.0 with high uncertainty
            return CircuitAffinityEstimate(
                driver_id=driver_id,
                circuit_id=circuit_id,
                mean_finish=11.0,
                shrinkage_mean=11.0,
                sample_size=0.0,
                shrinkage_weight=0.0,
                uncertainty=self.default_prior_std,
            )

        valid_career_finishes = driver_prior["finish_position"].dropna()
        valid_career_finishes = valid_career_finishes[valid_career_finishes > 0]
        global_mean = float(valid_career_finishes.mean()) if not valid_career_finishes.empty else 11.0
        within_std = float(valid_career_finishes.std()) if len(valid_career_finishes) > 1 else self.default_prior_std

        # 3. Driver circuit-specific history
        circuit_prior = driver_prior[driver_prior["circuit_id"] == circuit_id]
        if circuit_prior.empty:
            # First appearance at this specific circuit
            return CircuitAffinityEstimate(
                driver_id=driver_id,
                circuit_id=circuit_id,
                mean_finish=global_mean,
                shrinkage_mean=global_mean,
                sample_size=0.0,
                shrinkage_weight=0.0,
                uncertainty=float(within_std / math.sqrt(self.kappa_0)),
            )

        # 4. Apply temporal weighting hypothesis
        finishes = circuit_prior["finish_position"].values
        seasons = circuit_prior["season"].values
        eras = circuit_prior.get("regulation_era", pd.Series([""] * len(circuit_prior))).values

        weights = self._calculate_temporal_weights(
            seasons=seasons,
            eras=eras,
            target_season=target_season,
            hypothesis=hypothesis,
            half_life_years=half_life_years,
        )

        effective_n = float(np.sum(weights))
        if effective_n <= 0:
            weighted_circuit_mean = global_mean
        else:
            weighted_circuit_mean = float(np.sum(weights * finishes) / effective_n)

        # 5. Empirical Bayes Shrinkage
        # alpha = N / (N + kappa_0)
        alpha = effective_n / (effective_n + self.kappa_0)
        shrinkage_mean = alpha * weighted_circuit_mean + (1.0 - alpha) * global_mean

        # Posterior standard error
        post_var = (within_std ** 2) / (effective_n + self.kappa_0)
        post_se = float(math.sqrt(max(post_var, 0.01)))

        return CircuitAffinityEstimate(
            driver_id=driver_id,
            circuit_id=circuit_id,
            mean_finish=weighted_circuit_mean,
            shrinkage_mean=shrinkage_mean,
            sample_size=effective_n,
            shrinkage_weight=alpha,
            uncertainty=post_se,
        )

    @staticmethod
    def _calculate_temporal_weights(
        seasons: np.ndarray,
        eras: np.ndarray,
        target_season: int,
        hypothesis: str,
        half_life_years: float,
    ) -> np.ndarray:
        """Calculate observation weights under four competing temporal hypotheses."""
        delta_years = np.maximum(target_season - seasons, 0).astype(float)

        if hypothesis == "H1":
            # Continuous Exponential Half-Life
            decay = math.log(2.0) / max(half_life_years, 1.0)
            return np.exp(-delta_years * decay)

        elif hypothesis == "H2":
            # Discrete Era Step-Down
            era_weights = {
                "ground_effect_modern": 1.00,
                "turbo_hybrid": 0.60,
                "v8_frozen": 0.25,
                "v10": 0.10,
                "classic": 0.05,
            }
            return np.array([era_weights.get(str(e), 0.50) for e in eras], dtype=float)

        elif hypothesis == "H3":
            # Hierarchical Random Walk: inverse variance decay
            # Var(t) = Var(0) + delta * sigma_w^2
            sigma_w_sq = 0.5
            return 1.0 / (1.0 + delta_years * sigma_w_sq)

        elif hypothesis == "H4":
            # Era-Invariant Uniform Weighting
            return np.ones(len(seasons), dtype=float)

        else:
            # Default to H1
            decay = math.log(2.0) / 3.0
            return np.exp(-delta_years * decay)


def _parse_level(val, default: float = 3.0) -> float:
    if val is None or pd.isna(val):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    mapping = {
        "very low": 1.0,
        "low": 2.0,
        "medium": 3.0,
        "high": 4.0,
        "very high": 5.0,
    }
    return mapping.get(str(val).strip().lower(), default)


def extract_circuit_topology_features(
    circuit_characteristics_df: pd.DataFrame,
    circuit_id: str,
) -> Dict[str, float]:
    """Extract normalized physical circuit characteristics for feature vector interactions."""
    resolved = resolve_circuit(circuit_id, strict=False)
    c_id = resolved.canonical_id if resolved.is_resolved else str(circuit_id).strip().lower()

    c_row = circuit_characteristics_df[circuit_characteristics_df["circuit_id"] == c_id]
    if c_row.empty:
        c_row = circuit_characteristics_df[circuit_characteristics_df["circuit_id"] == str(circuit_id).strip().lower()]
    if c_row.empty:
        return {
            "circuit_length_km": 5.0,
            "overtaking_difficulty": 3.0,
            "tyre_stress_level": 3.0,
            "downforce_level": 3.0,
            "altitude_m": 100.0,
        }

    row = c_row.iloc[0]
    return {
        "circuit_length_km": float(row.get("circuit_length_km", 5.0)),
        "overtaking_difficulty": _parse_level(row.get("overtaking_difficulty_rating"), 3.0),
        "tyre_stress_level": _parse_level(row.get("tyre_stress_level"), 3.0),
        "downforce_level": _parse_level(row.get("downforce_level"), 3.0),
        "altitude_m": float(row.get("altitude_m", 100.0)),
    }
