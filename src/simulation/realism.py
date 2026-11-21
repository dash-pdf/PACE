"""PACE: Simulation Realism & Macro Event Frequency Benchmark.

Validates the macro-realism of the stochastic race simulator against
authoritative historical Formula 1 empirical distributions (2021–2025 eras):
1. Safety Car, Virtual Safety Car, and Red Flag deployment frequencies.
2. Mechanical vs Incident DNF rates and grid attrition distributions.
3. Total on-track overtakes and pit stop strategic diversity.
4. Monte Carlo distribution stability and convergence across sample size scaling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.simulation.monte_carlo import MonteCarloResult

logger = logging.getLogger("pace.simulation.realism")

# Authoritative empirical Formula 1 benchmarks (derived from 2021-2025 FIA race records)
HISTORICAL_F1_BENCHMARKS = {
    "safety_cars_per_race": {"mean": 0.85, "min_plausible": 0.0, "max_plausible": 3.0},
    "vsc_per_race": {"mean": 0.65, "min_plausible": 0.0, "max_plausible": 2.5},
    "red_flags_per_race": {"mean": 0.15, "min_plausible": 0.0, "max_plausible": 1.0},
    "mechanical_dnfs_per_race": {"mean": 1.45, "min_plausible": 0.2, "max_plausible": 4.5},
    "incident_dnfs_per_race": {"mean": 0.95, "min_plausible": 0.0, "max_plausible": 3.5},
    "total_dnfs_per_race": {"mean": 2.40, "min_plausible": 0.5, "max_plausible": 6.5},
    "overtakes_per_race": {"mean": 42.0, "min_plausible": 8.0, "max_plausible": 150.0},
    "pit_stops_per_race": {"mean": 32.0, "min_plausible": 18.0, "max_plausible": 65.0},
}


class SimulationRealismAuditor:
    """Audits simulation event frequencies and distribution stability against historical truth."""

    @staticmethod
    def audit_event_frequencies(mc_result: MonteCarloResult) -> Dict[str, Any]:
        """Compares Monte Carlo event frequencies against empirical Formula 1 historical ranges."""
        stats = mc_result.event_frequencies
        audit_results = {}
        all_passed = True

        mapping = {
            "mean_safety_cars_per_race": "safety_cars_per_race",
            "mean_vsc_per_race": "vsc_per_race",
            "mean_red_flags_per_race": "red_flags_per_race",
            "mean_mechanical_dnfs_per_race": "mechanical_dnfs_per_race",
            "mean_incident_dnfs_per_race": "incident_dnfs_per_race",
            "mean_total_dnfs_per_race": "total_dnfs_per_race",
            "mean_overtakes_per_race": "overtakes_per_race",
            "mean_pit_stops_per_race": "pit_stops_per_race",
        }

        for stat_key, bench_key in mapping.items():
            sim_val = stats.get(stat_key, 0.0)
            bench = HISTORICAL_F1_BENCHMARKS[bench_key]
            in_range = bool(bench["min_plausible"] <= sim_val <= bench["max_plausible"])
            ratio_to_norm = float(sim_val / bench["mean"]) if bench["mean"] > 0 else 1.0

            if not in_range:
                all_passed = False

            audit_results[bench_key] = {
                "simulated_mean": round(sim_val, 3),
                "historical_norm": bench["mean"],
                "plausible_range": [bench["min_plausible"], bench["max_plausible"]],
                "in_plausible_range": in_range,
                "ratio_to_historical": round(ratio_to_norm, 2),
            }

        return {
            "all_frequencies_plausible": all_passed,
            "detailed_audit": audit_results,
            "n_simulations_evaluated": mc_result.n_simulations,
        }

    @staticmethod
    def evaluate_distribution_stability(
        convergence_traces: Dict[str, List[Dict[str, Any]]],
        threshold_delta: float = 0.05,
    ) -> Dict[str, Any]:
        """Evaluates whether win and podium probabilities stabilize as iteration counts increase.
        
        Calculates maximum probability shift between consecutive sample sizes.
        """
        max_win_shift = 0.0
        max_podium_shift = 0.0
        n_checkpoints = []

        for d_id, trace in convergence_traces.items():
            if len(trace) < 2:
                continue
            for i in range(1, len(trace)):
                prev = trace[i - 1]
                curr = trace[i]
                n_checkpoints.append(curr["n"])

                # Only evaluate stability after warm-up (> 200 iterations)
                if curr["n"] >= 200:
                    d_win = abs(curr["p_win"] - prev["p_win"])
                    d_pod = abs(curr["p_podium"] - prev["p_podium"])
                    if d_win > max_win_shift:
                        max_win_shift = d_win
                    if d_pod > max_podium_shift:
                        max_podium_shift = d_pod

        is_stable = bool(max_win_shift <= threshold_delta and max_podium_shift <= threshold_delta)

        return {
            "is_stable": is_stable,
            "max_win_probability_shift": round(max_win_shift, 4),
            "max_podium_probability_shift": round(max_podium_shift, 4),
            "stability_threshold": threshold_delta,
            "diagnostics": (
                "Monte Carlo estimates converged cleanly (max shift <= 5.0%)."
                if is_stable
                else f"Significant distribution shifts detected ({max_win_shift:.3f} > {threshold_delta}). Consider increasing n_simulations."
            ),
        }
