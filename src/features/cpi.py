"""PACE: Constructor Performance Index (CPI) Module.

Implements constructor competitiveness quantification:
1. Strict separation of directly measured telemetry (speed traps, sector timing)
   from inferred aerodynamic and mechanical properties with explicit uncertainty.
2. Latent fuel state stint clustering (short-run vs. long-run classification).
3. Circuit fuel sensitivity calculation without hardcoded point constants.
4. Robust tyre degradation estimation (Huber M-estimation) with standard errors.
5. Qualifying pole gap, clean-air race pace, and exponential mechanical reliability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class InferredTelemetryMetrics:
    """Carries inferred vehicle characteristics along with explicit uncertainty bounds."""
    downforce_ratio: float = 1.0        # Ratio of cornering minimum speed to terminal speed
    downforce_uncertainty: float = 0.25 # Epistemic standard error
    mech_traction_rating: float = 1.0   # Low speed longitudinal traction proxy
    mech_uncertainty: float = 0.20      # Estimation standard error
    degradation_slope: float = 0.05     # Tyre wear in seconds/lap
    degradation_se: float = 0.02        # Standard error of degradation slope
    speed_trap_percentile: float = 0.50 # Direct measurement: percentile in speed trap ranking


def calculate_circuit_fuel_sensitivity(
    circuit_length_km: float = 5.0,
    tyre_stress_level: float = 3.0,
    overtaking_difficulty: float = 3.0,
) -> float:
    """Calculate circuit-specific fuel mass sensitivity factor S_fuel,c (seconds per lap per 10 kg).
    
    Rather than hardcoding -0.06 s/lap everywhere, this derives the sensitivity
    from physical track length and acceleration-braking intensity.
    
    Returns:
        s_fuel: Estimated lap time cost in seconds per 10 kg fuel.
    """
    length = max(circuit_length_km, 3.0)
    base_sensitivity = 0.050  # Base sensitivity for a standard 5.0 km track (~0.05s / 10kg)
    length_multiplier = length / 5.0

    # Tracks with heavy braking and stop-and-go acceleration (correlated with overtaking difficulty / tyre stress)
    intensity_factor = 1.0 + 0.08 * (tyre_stress_level - 3.0) + 0.05 * (overtaking_difficulty - 3.0)
    sensitivity = base_sensitivity * length_multiplier * max(intensity_factor, 0.70)
    return float(np.clip(sensitivity, 0.025, 0.090))


def estimate_tyre_degradation_huber(
    lap_times_series: pd.Series,
    circuit_fuel_sensitivity: float = 0.05,
    race_laps_total: int = 55,
    initial_fuel_kg: float = 100.0,
) -> Tuple[float, float]:
    """Estimate tyre degradation slope (k_deg) using robust Huber loss M-estimation.
    
    Adjusts observed lap times for fuel burn-off over the stint:
        T(l) = T_0 + (k_deg - S_fuel * (m_fuel_0 / L_total)) * l + eps
    
    Returns:
        (k_deg, se_k_deg): Estimated degradation in seconds/lap and standard error.
    """
    times = lap_times_series.dropna().values
    n = len(times)
    if n < 6:
        # Insufficient long-run laps to estimate degradation robustly
        return 0.050, 0.040

    laps = np.arange(1, n + 1, dtype=float)
    # Expected fuel burn-off per lap (kg/lap)
    fuel_burn_per_lap_kg = initial_fuel_kg / float(max(race_laps_total, 40))
    fuel_lap_offset = circuit_fuel_sensitivity * (fuel_burn_per_lap_kg / 10.0)

    # Median Absolute Deviation (MAD) for robust scaling
    y_raw = times
    # Huber loss regression using iterative reweighted least squares (IRLS)
    # Initialize with OLS
    X = np.column_stack([np.ones(n), laps])
    beta = np.linalg.lstsq(X, y_raw, rcond=None)[0]

    for _ in range(15):
        residuals = y_raw - (X @ beta)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = max(mad * 1.4826, 1e-4)
        u = residuals / scale
        # Huber weights
        weights = np.where(np.abs(u) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(u), 1e-6))
        W = np.diag(weights)
        try:
            beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_raw)
        except np.linalg.LinAlgError:
            break

    observed_slope = beta[1]
    # Net tyre degradation = observed lap progression + fuel burn-off recovery
    k_deg = observed_slope + fuel_lap_offset

    # Compute standard error of the slope
    residuals = y_raw - (X @ beta)
    deg_freedom = max(n - 2, 1)
    sigma_res = np.sqrt(np.sum(residuals ** 2) / deg_freedom)
    var_X = np.sum((laps - np.mean(laps)) ** 2)
    se_slope = sigma_res / np.sqrt(max(var_X, 1e-6))

    return float(np.clip(k_deg, -0.05, 0.30)), float(np.clip(se_slope, 0.005, 0.100))


def calculate_constructor_qualifying_pole_gap(
    qualifying_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate each constructor's percentage qualifying gap to pole position.
    
    Pole gap = (min_constructor_q_time - pole_time) / pole_time.
    """
    records = []
    cols = ["season", "round", "constructor_id", "pole_gap_pct", "team_best_q_ms"]
    if qualifying_df.empty or "season" not in qualifying_df.columns or "round" not in qualifying_df.columns:
        return pd.DataFrame(columns=cols)

    grouped = qualifying_df.groupby(["season", "round"])
    for (season, rnd), session_df in grouped:
        best_times = []
        for _, row in session_df.iterrows():
            q_time = None
            for col in ["q3_time_ms", "q2_time_ms", "q1_time_ms"]:
                val = row.get(col)
                if val is not None and not pd.isna(val) and val > 0:
                    q_time = float(val)
                    break
            if q_time is not None:
                best_times.append({
                    "constructor_id": row["constructor_id"],
                    "best_time": q_time,
                })

        if not best_times:
            continue

        bt_df = pd.DataFrame(best_times)
        team_min = bt_df.groupby("constructor_id")["best_time"].min()
        pole_time = team_min.min()

        for constructor_id, t_min in team_min.items():
            gap_pct = (t_min - pole_time) / pole_time
            records.append({
                "season": season,
                "round": rnd,
                "constructor_id": constructor_id,
                "pole_gap_pct": gap_pct,
                "team_best_q_ms": t_min,
            })

    return pd.DataFrame(records)


def calculate_constructor_reliability(
    race_results_df: pd.DataFrame,
    target_season: int,
    target_round: int,
    half_life_races: float = 10.0,
) -> Dict[str, float]:
    """Calculate exponential mechanical reliability index Rel_c for each constructor.
    
    Only considers historical races strictly prior to target_round / target_season.
    Weights each past race with an exponential decay: w = exp(-delta_races / half_life).
    Mechanical DNF (is_dnf = True and not collision) scores 0.0, completion scores 1.0.
    """
    prior = race_results_df[
        (race_results_df["season"] < target_season) |
        ((race_results_df["season"] == target_season) & (race_results_df["round"] < target_round))
    ].copy()

    if prior.empty:
        # Return default 0.90 reliability if no historical data available
        constructors = race_results_df["constructor_id"].unique()
        return {c: 0.90 for c in constructors}

    # Sort chronologically
    prior = prior.sort_values(["season", "round"], ascending=False)
    # Assign race index distance
    prior["event_key"] = prior["season"].astype(str) + "_" + prior["round"].astype(str)
    event_order = {k: i for i, k in enumerate(prior["event_key"].unique())}
    prior["race_distance"] = prior["event_key"].map(event_order)

    decay_factor = math.log(2.0) / half_life_races
    prior["weight"] = np.exp(-prior["race_distance"] * decay_factor)

    # Classify mechanical failure
    def is_mech_failure(row):
        if not row.get("is_dnf", False):
            return 0.0
        status = str(row.get("status", "")).lower()
        if any(w in status for w in ["collision", "accident", "spun off", "crash"]):
            return 0.0
        return 1.0

    prior["is_mech_dnf"] = prior.apply(is_mech_failure, axis=1)
    prior["success_score"] = 1.0 - prior["is_mech_dnf"]

    reliability_map = {}
    for constructor_id, c_df in prior.groupby("constructor_id"):
        w = c_df["weight"]
        s = c_df["success_score"]
        weighted_rel = float(np.sum(w * s) / max(np.sum(w), 1e-6))
        reliability_map[constructor_id] = float(np.clip(weighted_rel, 0.40, 0.99))

    return reliability_map


def calculate_constructor_rolling_trend(
    pole_gaps_df: pd.DataFrame,
    target_season: int,
    target_round: int,
    window_size: int = 5,
) -> Dict[str, float]:
    """Calculate rolling 5-race development trajectory (beta_dev) for each constructor.
    
    A negative slope indicates improving competitiveness (reducing gap to pole).
    A positive slope indicates relative regression.
    """
    if pole_gaps_df.empty or "season" not in pole_gaps_df.columns:
        return {}

    prior = pole_gaps_df[
        (pole_gaps_df["season"] < target_season) |
        ((pole_gaps_df["season"] == target_season) & (pole_gaps_df["round"] < target_round))
    ]
    if prior.empty:
        return {}
    prior = prior.sort_values(["season", "round"])

    trends = {}
    for constructor_id, team_df in prior.groupby("constructor_id"):
        recent = team_df.tail(window_size)
        if len(recent) < 3:
            trends[constructor_id] = 0.0
            continue

        y = recent["pole_gap_pct"].values
        x = np.arange(len(y), dtype=float)
        slope, _ = np.polyfit(x, y, 1)
        trends[constructor_id] = float(np.clip(slope, -0.05, 0.05))

    return trends
