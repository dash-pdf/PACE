"""PACE: Monte Carlo Stochastic Race Simulation Engine.

Orchestrates multi-run parallel/vectorized event-driven race simulations to
derive mathematically consistent, calibrated probability distributions across
the entire Formula 1 grid.

Outputs for each driver:
1. Win probability P(P1).
2. Podium probability P(P1-P3).
3. Top-five probability P(P1-P5).
4. Points-finish probability P(P1-P10).
5. Mechanical DNF probability P(DNF_mech) and incident DNF probability P(DNF_inc).
6. Complete finishing position distribution P(Y_i = k) for k in 1..N.
7. Expected finishing position (classified and overall) and most likely position (mode).
8. Prediction intervals (P05, P10, P50, P90, P95, IQR), standard deviation, and entropy.
9. Grid-level mathematical consistency verification (doubly-stochastic permutation property).
10. Monte Carlo Standard Error (MCSE) convergence diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.simulation.state import CarStatus, RaceMode, TrackCondition
from src.simulation.engine import BaselineRaceSimulator, SimulationResult
from src.simulation.weather import WeatherEvolutionConfig

logger = logging.getLogger("pace.simulation.monte_carlo")


def _run_single_simulation_worker(
    run_seed: int,
    event_features_dict: List[Dict[str, Any]],
    total_laps: int,
    circuit_id: str,
    circuit_length_km: float,
    base_lap_time: float,
    tyre_stress_level: float,
    utility_scaling_factor: float,
    enable_mechanical_failures: bool,
    enable_incidents: bool,
    enable_safety_car: bool,
    weather_config_dict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Isolated, top-level picklable worker function for multiprocessing."""
    event_features = pd.DataFrame(event_features_dict)
    weather_config = (
        WeatherEvolutionConfig(**weather_config_dict)
        if weather_config_dict is not None
        else None
    )

    simulator = BaselineRaceSimulator(utility_scaling_factor=utility_scaling_factor)
    res: SimulationResult = simulator.simulate_race(
        event_features=event_features,
        total_laps=total_laps,
        circuit_id=circuit_id,
        circuit_length_km=circuit_length_km,
        base_lap_time=base_lap_time,
        tyre_stress_level=tyre_stress_level,
        random_seed=run_seed,
        deterministic=False,
        weather_config=weather_config,
        enable_mechanical_failures=enable_mechanical_failures,
        enable_incidents=enable_incidents,
        enable_safety_car=enable_safety_car,
    )

    df_class = res.final_classification
    car_records = []
    for _, row in df_class.iterrows():
        car_records.append({
            "driver_id": str(row["driver_id"]),
            "constructor_id": str(row.get("constructor_id", "unknown")),
            "grid_position": int(row.get("grid_position", 0)),
            "finish_position": int(row["finish_position"]),
            "laps_completed": int(row.get("laps_completed", 0)),
            "is_classified": bool(row.get("is_classified", True)),
            "status": str(row.get("status", "finished")),
            "pit_stops": int(row.get("pit_stops", 0)),
            "overtakes_made": int(row.get("overtakes_made", 0)),
        })

    return {
        "run_seed": run_seed,
        "classifications": car_records,
        "summary_metrics": dict(res.summary_metrics),
    }


@dataclass
class MonteCarloResult:
    """Aggregated probabilistic results from N stochastic race simulations."""

    n_simulations: int
    drivers: List[str]
    constructors: Dict[str, str]
    grid_positions: Dict[str, int]
    n_grid: int

    # Probability matrix: shape [N_drivers, N_grid] where [i, k-1] = P(Driver i finishes P_k)
    position_matrix: np.ndarray

    # Primary event probabilities per driver
    p_win: Dict[str, float]
    p_podium: Dict[str, float]
    p_top5: Dict[str, float]
    p_points: Dict[str, float]
    p_dnf_mech: Dict[str, float]
    p_dnf_inc: Dict[str, float]
    p_dnf_total: Dict[str, float]
    p_classified: Dict[str, float]
    p_unclassified: Dict[str, float]

    # Point predictions and summaries
    expected_position: Dict[str, float]
    expected_classified_position: Dict[str, float]
    most_likely_position: Dict[str, int]
    median_position: Dict[str, float]
    std_dev: Dict[str, float]
    entropy: Dict[str, float]

    # Percentile prediction intervals
    percentiles: Dict[str, Dict[str, float]]  # P05, P10, P25, P75, P90, P95, IQR

    # Monte Carlo Standard Error (MCSE) for probability estimates
    mcse: Dict[str, Dict[str, float]]

    # Simulation convergence history
    convergence_traces: Dict[str, Any]

    # Macro event frequencies across runs (Safety Cars, VSCs, DNFs, etc.)
    event_frequencies: Dict[str, float]

    def get_driver_position_distribution(self, driver_id: str) -> np.ndarray:
        """Returns the discrete PMF array [P(P1), P(P2), ..., P(PN)] for a driver."""
        if driver_id not in self.drivers:
            raise KeyError(f"Driver '{driver_id}' not found in simulation results.")
        idx = self.drivers.index(driver_id)
        return self.position_matrix[idx, :]

    def verify_mathematical_consistency(self, tolerance: float = 1e-4) -> Dict[str, Any]:
        """Validates formal mathematical invariants of the Monte Carlo grid outcome space."""
        # 1. Driver-wise probability conservation: sum_k P(Y_i = k) == 1.0
        row_sums = self.position_matrix.sum(axis=1)
        driver_prob_valid = bool(np.all(np.abs(row_sums - 1.0) < tolerance))

        # 2. Position-wise probability conservation: sum_i P(Y_i = k) == 1.0
        col_sums = self.position_matrix.sum(axis=0)
        position_prob_valid = bool(np.all(np.abs(col_sums - 1.0) < tolerance))

        # 3. Non-negativity and upper bound: 0.0 <= P <= 1.0
        bounds_valid = bool(np.all(self.position_matrix >= -tolerance) and np.all(self.position_matrix <= 1.0 + tolerance))

        # 4. Total probability mass in matrix equals N_grid
        total_mass = float(self.position_matrix.sum())
        total_mass_valid = bool(abs(total_mass - float(self.n_grid)) < tolerance)

        # 5. DNF decomposition consistency: P(DNF) == P(DNF_mech) + P(DNF_inc)
        dnf_decomp_valid = True
        for d in self.drivers:
            diff = abs(self.p_dnf_total[d] - (self.p_dnf_mech[d] + self.p_dnf_inc[d]))
            if diff > tolerance:
                dnf_decomp_valid = False
                break

        is_doubly_stochastic = driver_prob_valid and position_prob_valid and bounds_valid

        return {
            "is_valid": is_doubly_stochastic and dnf_decomp_valid and total_mass_valid,
            "driver_probabilities_sum_to_one": driver_prob_valid,
            "position_probabilities_sum_to_one": position_prob_valid,
            "probabilities_in_unit_interval": bounds_valid,
            "dnf_decomposition_consistent": dnf_decomp_valid,
            "is_doubly_stochastic": is_doubly_stochastic,
            "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
            "max_col_sum_error": float(np.max(np.abs(col_sums - 1.0))),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts Monte Carlo predictions into a comprehensive, sorted reporting DataFrame."""
        rows = []
        for i, d in enumerate(self.drivers):
            ptiles = self.percentiles[d]
            mc = self.mcse[d]
            rows.append({
                "driver_id": d,
                "constructor_id": self.constructors.get(d, "unknown"),
                "grid_position": self.grid_positions.get(d, 0),
                "expected_position": round(self.expected_position[d], 2),
                "expected_classified": round(self.expected_classified_position[d], 2),
                "most_likely_position": self.most_likely_position[d],
                "median_position": ptiles["P50"],
                "p05": ptiles["P05"],
                "p95": ptiles["P95"],
                "p10": ptiles["P10"],
                "p90": ptiles["P90"],
                "iqr": ptiles["IQR"],
                "std_dev": round(self.std_dev[d], 3),
                "entropy": round(self.entropy[d], 3),
                "p_win": round(self.p_win[d], 4),
                "p_podium": round(self.p_podium[d], 4),
                "p_top5": round(self.p_top5[d], 4),
                "p_points": round(self.p_points[d], 4),
                "p_dnf_mech": round(self.p_dnf_mech[d], 4),
                "p_dnf_inc": round(self.p_dnf_inc[d], 4),
                "p_dnf_total": round(self.p_dnf_total[d], 4),
                "p_classified": round(self.p_classified[d], 4),
                "mcse_win": round(mc["win"], 5),
                "mcse_podium": round(mc["podium"], 5),
            })

        df = pd.DataFrame(rows)
        # Sort by expected position ascending (predicted ranking order)
        df = df.sort_values("expected_position").reset_index(drop=True)
        df.insert(0, "predicted_rank", range(1, len(df) + 1))
        return df


class MonteCarloSimulationEngine:
    """High-performance Monte Carlo orchestration engine for stochastic race simulation."""

    def __init__(
        self,
        base_simulator: Optional[BaselineRaceSimulator] = None,
        utility_scaling_factor: float = 0.05,
        n_jobs: int = 1,
    ):
        self.base_simulator = base_simulator or BaselineRaceSimulator(
            utility_scaling_factor=utility_scaling_factor
        )
        self.utility_scaling_factor = utility_scaling_factor
        self.n_jobs = n_jobs

    def run_simulations(
        self,
        event_features: pd.DataFrame,
        n_simulations: int = 500,
        total_laps: int = 58,
        circuit_id: str = "albert_park",
        circuit_length_km: float = 5.278,
        base_lap_time: float = 80.0,
        tyre_stress_level: float = 3.0,
        random_seed: int = 42,
        weather_config: Optional[WeatherEvolutionConfig] = None,
        enable_mechanical_failures: bool = True,
        enable_incidents: bool = True,
        enable_safety_car: bool = True,
        track_convergence_checkpoints: Optional[List[int]] = None,
    ) -> MonteCarloResult:
        """Executes n_simulations independent stochastic Grand Prix simulations."""
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

        logger.info(
            f"Starting Monte Carlo simulation: {n_simulations} iterations for {circuit_id} "
            f"({len(event_features)} drivers, seed={random_seed}, n_jobs={self.n_jobs})"
        )

        # Generate reproducible, independent sub-seeds
        ss = np.random.SeedSequence(random_seed)
        child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_simulations)]

        # Prepare arguments
        features_dict = event_features.to_dict(orient="records")
        weather_dict = (
            {
                "initial_condition": weather_config.initial_condition.value if hasattr(weather_config.initial_condition, "value") else str(weather_config.initial_condition),
                "rain_probability": weather_config.rain_probability,
                "dynamic_evolution": weather_config.dynamic_evolution,
            }
            if weather_config is not None
            else None
        )

        # Checkpoints for convergence monitoring
        if track_convergence_checkpoints is None:
            raw_checkpoints = [25, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000]
            checkpoints = set([cp for cp in raw_checkpoints if cp <= n_simulations])
            checkpoints.add(n_simulations)
        else:
            checkpoints = set(track_convergence_checkpoints)

        run_results: List[Dict[str, Any]] = []

        # Execute runs: parallel if n_jobs > 1 and n_simulations >= 10, else serial
        can_parallelize = (self.n_jobs > 1 or self.n_jobs == -1) and n_simulations >= 10
        if can_parallelize:
            max_workers = multiprocessing.cpu_count() if self.n_jobs == -1 else self.n_jobs
            max_workers = min(max_workers, n_simulations, 8)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _run_single_simulation_worker,
                        seed,
                        features_dict,
                        total_laps,
                        circuit_id,
                        circuit_length_km,
                        base_lap_time,
                        tyre_stress_level,
                        self.utility_scaling_factor,
                        enable_mechanical_failures,
                        enable_incidents,
                        enable_safety_car,
                        weather_dict,
                    )
                    for seed in child_seeds
                ]
                for future in as_completed(futures):
                    run_results.append(future.result())
        else:
            # Serial execution
            for seed in child_seeds:
                res = _run_single_simulation_worker(
                    seed,
                    features_dict,
                    total_laps,
                    circuit_id,
                    circuit_length_km,
                    base_lap_time,
                    tyre_stress_level,
                    self.utility_scaling_factor,
                    enable_mechanical_failures,
                    enable_incidents,
                    enable_safety_car,
                    weather_dict,
                )
                run_results.append(res)

        return self._aggregate_results(
            run_results=run_results,
            event_features=event_features,
            checkpoints=sorted(list(checkpoints)),
        )

    def _aggregate_results(
        self,
        run_results: List[Dict[str, Any]],
        event_features: pd.DataFrame,
        checkpoints: List[int],
    ) -> MonteCarloResult:
        """Aggregates raw simulation runs into joint probabilistic distributions and metrics."""
        n_sims = len(run_results)
        drivers = list(event_features["driver_id"])
        n_drivers = len(drivers)
        driver_to_idx = {d: i for i, d in enumerate(drivers)}

        constructors = dict(zip(event_features["driver_id"], event_features["constructor_id"]))
        grid_positions = dict(zip(event_features["driver_id"], event_features["grid_position"].astype(int)))

        # Position frequency matrix: [n_drivers, n_drivers]
        position_counts = np.zeros((n_drivers, n_drivers), dtype=np.int64)

        # Event counts
        win_counts = np.zeros(n_drivers, dtype=np.int64)
        podium_counts = np.zeros(n_drivers, dtype=np.int64)
        top5_counts = np.zeros(n_drivers, dtype=np.int64)
        points_counts = np.zeros(n_drivers, dtype=np.int64)
        dnf_mech_counts = np.zeros(n_drivers, dtype=np.int64)
        dnf_inc_counts = np.zeros(n_drivers, dtype=np.int64)
        classified_counts = np.zeros(n_drivers, dtype=np.int64)

        # Finish position samples per driver for percentiles and moments
        driver_finish_samples: Dict[str, List[int]] = {d: [] for d in drivers}
        driver_classified_samples: Dict[str, List[int]] = {d: [] for d in drivers}

        # Global event counters across runs
        total_scs = 0.0
        total_vscs = 0.0
        total_red_flags = 0.0
        total_mechanical_dnfs = 0.0
        total_incident_dnfs = 0.0
        total_overtakes = 0.0
        total_pit_stops = 0.0

        # Convergence tracking structures
        checkpoint_set = set(checkpoints)
        convergence_traces: Dict[str, List[Dict[str, Any]]] = {d: [] for d in drivers}

        # Running tally for convergence
        running_wins = np.zeros(n_drivers, dtype=np.int64)
        running_podiums = np.zeros(n_drivers, dtype=np.int64)

        for run_idx, run in enumerate(run_results, 1):
            classifications = run["classifications"]
            summary = run["summary_metrics"]

            total_scs += summary.get("safety_car_deployments", 0)
            total_vscs += summary.get("vsc_deployments", 0)
            total_red_flags += summary.get("red_flag_deployments", 0)
            total_mechanical_dnfs += summary.get("mechanical_dnfs", 0)
            total_incident_dnfs += summary.get("incident_dnfs", 0)
            total_overtakes += summary.get("total_overtakes", 0)
            total_pit_stops += summary.get("total_pit_stops", 0)

            for car in classifications:
                d_id = car["driver_id"]
                if d_id not in driver_to_idx:
                    continue
                d_idx = driver_to_idx[d_id]
                pos = car["finish_position"]
                status = car["status"]
                is_class = car["is_classified"]

                # Accumulate counts
                pos_idx = pos - 1
                if 0 <= pos_idx < n_drivers:
                    position_counts[d_idx, pos_idx] += 1

                driver_finish_samples[d_id].append(pos)
                if is_class:
                    driver_classified_samples[d_id].append(pos)
                    classified_counts[d_idx] += 1

                if pos == 1:
                    win_counts[d_idx] += 1
                    running_wins[d_idx] += 1
                if pos <= 3:
                    podium_counts[d_idx] += 1
                    running_podiums[d_idx] += 1
                if pos <= 5:
                    top5_counts[d_idx] += 1
                if pos <= 10:
                    points_counts[d_idx] += 1

                if status == "dnf_mechanical":
                    dnf_mech_counts[d_idx] += 1
                elif status == "dnf_incident":
                    dnf_inc_counts[d_idx] += 1

            # Checkpoint trace recording
            if run_idx in checkpoint_set:
                for d_id, d_idx in driver_to_idx.items():
                    convergence_traces[d_id].append({
                        "n": run_idx,
                        "p_win": float(running_wins[d_idx] / run_idx),
                        "p_podium": float(running_podiums[d_idx] / run_idx),
                    })

        # Calculate probability matrix
        position_matrix = position_counts.astype(np.float64) / float(n_sims)

        # Calculate driver-level probabilities
        p_win = {d: float(win_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_podium = {d: float(podium_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_top5 = {d: float(top5_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_points = {d: float(points_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_dnf_mech = {d: float(dnf_mech_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_dnf_inc = {d: float(dnf_inc_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_dnf_total = {d: p_dnf_mech[d] + p_dnf_inc[d] for d in drivers}
        p_classified = {d: float(classified_counts[i] / n_sims) for i, d in enumerate(drivers)}
        p_unclassified = {d: float(1.0 - p_classified[d]) for d in drivers}

        # Expected positions, modes, standard deviations, and percentiles
        positions_vec = np.arange(1, n_drivers + 1, dtype=np.float64)
        expected_position = {}
        expected_classified_position = {}
        most_likely_position = {}
        median_position = {}
        std_dev = {}
        entropy = {}
        percentiles = {}
        mcse = {}

        for i, d in enumerate(drivers):
            samples = np.array(driver_finish_samples[d])
            class_samples = np.array(driver_classified_samples[d])

            # Expected position via PMF: E[Y] = sum_k k * P(Y=k)
            pmf = position_matrix[i, :]
            exp_pos = float(np.sum(positions_vec * pmf))
            expected_position[d] = exp_pos

            # Classified expected position
            if len(class_samples) > 0:
                expected_classified_position[d] = float(np.mean(class_samples))
            else:
                expected_classified_position[d] = exp_pos

            # Mode (most likely finish position)
            mode_pos = int(np.argmax(pmf) + 1)
            most_likely_position[d] = mode_pos

            # Standard deviation & Entropy
            var_pos = float(np.sum(((positions_vec - exp_pos) ** 2) * pmf))
            std_dev[d] = float(np.sqrt(max(0.0, var_pos)))

            # Shannon Entropy H = - sum(p * log2(p))
            non_zero_pmf = pmf[pmf > 0]
            entropy[d] = float(-np.sum(non_zero_pmf * np.log2(non_zero_pmf)))

            # Percentiles from empirical samples
            p05 = float(np.percentile(samples, 5))
            p10 = float(np.percentile(samples, 10))
            p25 = float(np.percentile(samples, 25))
            p50 = float(np.percentile(samples, 50))
            p75 = float(np.percentile(samples, 75))
            p90 = float(np.percentile(samples, 90))
            p95 = float(np.percentile(samples, 95))
            percentiles[d] = {
                "P05": p05,
                "P10": p10,
                "P25": p25,
                "P50": p50,
                "P75": p75,
                "P90": p90,
                "P95": p95,
                "IQR": float(p75 - p25),
            }
            median_position[d] = p50

            # Monte Carlo Standard Error (MCSE) for probability estimates: sqrt(p * (1 - p) / N)
            p_w = p_win[d]
            p_pod = p_podium[d]
            p_t5 = p_top5[d]
            p_pts = p_points[d]
            p_dnf = p_dnf_total[d]
            mcse[d] = {
                "win": float(np.sqrt(p_w * (1.0 - p_w) / n_sims)),
                "podium": float(np.sqrt(p_pod * (1.0 - p_pod) / n_sims)),
                "top5": float(np.sqrt(p_t5 * (1.0 - p_t5) / n_sims)),
                "points": float(np.sqrt(p_pts * (1.0 - p_pts) / n_sims)),
                "dnf": float(np.sqrt(p_dnf * (1.0 - p_dnf) / n_sims)),
            }

        event_frequencies = {
            "mean_safety_cars_per_race": float(total_scs / n_sims),
            "mean_vsc_per_race": float(total_vscs / n_sims),
            "mean_red_flags_per_race": float(total_red_flags / n_sims),
            "mean_mechanical_dnfs_per_race": float(total_mechanical_dnfs / n_sims),
            "mean_incident_dnfs_per_race": float(total_incident_dnfs / n_sims),
            "mean_total_dnfs_per_race": float((total_mechanical_dnfs + total_incident_dnfs) / n_sims),
            "mean_overtakes_per_race": float(total_overtakes / n_sims),
            "mean_pit_stops_per_race": float(total_pit_stops / n_sims),
        }

        return MonteCarloResult(
            n_simulations=n_sims,
            drivers=drivers,
            constructors=constructors,
            grid_positions=grid_positions,
            n_grid=n_drivers,
            position_matrix=position_matrix,
            p_win=p_win,
            p_podium=p_podium,
            p_top5=p_top5,
            p_points=p_points,
            p_dnf_mech=p_dnf_mech,
            p_dnf_inc=p_dnf_inc,
            p_dnf_total=p_dnf_total,
            p_classified=p_classified,
            p_unclassified=p_unclassified,
            expected_position=expected_position,
            expected_classified_position=expected_classified_position,
            most_likely_position=most_likely_position,
            median_position=median_position,
            std_dev=std_dev,
            entropy=entropy,
            percentiles=percentiles,
            mcse=mcse,
            convergence_traces=convergence_traces,
            event_frequencies=event_frequencies,
        )
