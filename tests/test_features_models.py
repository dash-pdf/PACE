"""PACE: Unit Tests for Feature Engineering, Modelling, and Evaluation Modules."""

import math
import numpy as np
import pandas as pd
import pytest

from src.features.dpi import (
    Glicko2Engine,
    calculate_rookie_learning_factor,
    calculate_pace_consistency_cv,
    BradleyTerryEngine,
)
from src.features.cpi import (
    calculate_circuit_fuel_sensitivity,
    estimate_tyre_degradation_huber,
    calculate_constructor_reliability,
)
from src.features.driver_circuit import (
    EmpiricalBayesCircuitAffinity,
    extract_circuit_topology_features,
)
from src.models.baselines import GridOrderBaseline
from src.models.ranker import RankPermutationSolver
from src.models.dnf import CompleteOutcomeSpaceEngine
from src.evaluation.metrics import (
    calculate_race_mae,
    calculate_spearman_rho,
    calculate_kendall_tau,
    calculate_top_k_hit_rate,
)


def test_glicko2_engine_updates():
    """Verify Glicko-2 rating updates, uncertainty shrinkage, and win-loss dynamics."""
    engine = Glicko2Engine(tau=0.5)
    r_ver = engine.get_rating("max_verstappen")
    r_per = engine.get_rating("perez")

    assert r_ver.rating == 1500.0
    assert r_ver.rd == 350.0

    # Max Verstappen beats Sergio Perez in 3 consecutive events
    for _ in range(3):
        engine.update_period([("max_verstappen", "perez", 1.0)])

    updated_ver = engine.get_rating("max_verstappen")
    updated_per = engine.get_rating("perez")

    # Winner's rating increases, loser's decreases
    assert updated_ver.rating > 1500.0
    assert updated_per.rating < 1500.0
    # Both rating deviations should shrink from 350.0 due to active data
    assert updated_ver.rd < 350.0
    assert updated_per.rd < 350.0
    assert updated_ver.starts == 3


def test_rookie_learning_factor_bounds():
    """Verify rookie learning curve: zero at start, steep slope early, asymptotic later."""
    assert calculate_rookie_learning_factor(0) == 0.0
    f_10 = calculate_rookie_learning_factor(10)
    f_25 = calculate_rookie_learning_factor(25)
    f_100 = calculate_rookie_learning_factor(100)
    f_300 = calculate_rookie_learning_factor(300)

    assert 0.0 < f_10 < f_25 < f_100 < 1.0
    # Around 25 starts (~1 season), f_learn should be approx 1 - exp(-1) = 0.632
    assert pytest.approx(f_25, abs=0.02) == 0.632
    # At 100+ starts, marginal learning benefit should plateau near 1.0
    assert f_100 > 0.98
    # Experience does not artificially exceed 1.0
    assert pytest.approx(f_300, abs=0.01) == 1.0


def test_circuit_fuel_sensitivity_bounds():
    """Verify circuit fuel sensitivity derives within realistic physical bounds (0.025 to 0.090 s/10kg)."""
    s_monza = calculate_circuit_fuel_sensitivity(circuit_length_km=5.793, tyre_stress_level=2.0, overtaking_difficulty=2.0)
    s_singapore = calculate_circuit_fuel_sensitivity(circuit_length_km=5.063, tyre_stress_level=4.0, overtaking_difficulty=4.5)
    s_short = calculate_circuit_fuel_sensitivity(circuit_length_km=3.337, tyre_stress_level=1.0, overtaking_difficulty=5.0)

    assert 0.025 <= s_monza <= 0.090
    assert 0.025 <= s_singapore <= 0.090
    assert 0.025 <= s_short <= 0.090
    # High tyre stress and stop-and-go circuit has higher fuel cost than low-stress layout
    assert s_singapore > s_short


def test_huber_tyre_degradation_estimation():
    """Verify robust Huber degradation estimation on synthetic lap times."""
    # Synthetic 15-lap stint with 0.08 s/lap degradation and 0.1s noise
    np.random.seed(42)
    laps = np.arange(1, 16)
    true_deg = 0.08
    lap_times = 90.0 + true_deg * laps + np.random.normal(0, 0.05, size=15)
    # Add an outlier lap (e.g. traffic or yellow flag)
    lap_times[7] += 2.5

    series = pd.Series(lap_times)
    k_deg, se_deg = estimate_tyre_degradation_huber(
        series,
        circuit_fuel_sensitivity=0.05,
        race_laps_total=50,
        initial_fuel_kg=100.0,
    )

    assert 0.05 <= k_deg <= 0.20
    assert se_deg > 0.0


def test_empirical_bayes_shrinkage_dynamics():
    """Verify Empirical Bayes pulls small-sample estimates toward global career baseline."""
    eb = EmpiricalBayesCircuitAffinity(kappa_0=5.0)

    # Synthetic career with 2 appearances at Monaco (P1, P1) but global average P8
    app_df = pd.DataFrame([
        {"driver_id": "norris", "circuit_id": "monaco", "season": 2024, "round": 8, "finish_position": 1.0, "regulation_era": "ground_effect_modern"},
        {"driver_id": "norris", "circuit_id": "monaco", "season": 2025, "round": 8, "finish_position": 1.0, "regulation_era": "ground_effect_modern"},
        {"driver_id": "norris", "circuit_id": "silverstone", "season": 2024, "round": 12, "finish_position": 8.0, "regulation_era": "ground_effect_modern"},
        {"driver_id": "norris", "circuit_id": "spa", "season": 2024, "round": 14, "finish_position": 10.0, "regulation_era": "ground_effect_modern"},
        {"driver_id": "norris", "circuit_id": "monza", "season": 2024, "round": 16, "finish_position": 9.0, "regulation_era": "ground_effect_modern"},
    ])

    aff = eb.compute_affinity(
        appearances_df=app_df,
        driver_id="norris",
        circuit_id="monaco",
        target_season=2026,
        target_round=8,
        hypothesis="H1",
    )

    # Observed Monaco mean is 1.0, global career mean is approx 5.8
    # With kappa_0=5.0 and N approx 2, shrinkage mean must be strictly between 1.0 and 5.8
    assert 1.0 < aff.shrinkage_mean < 5.8
    assert aff.uncertainty > 0.0


def test_rank_permutation_solver():
    """Verify RankPermutationSolver strictly guarantees valid 1..N permutation."""
    utilities = np.array([12.5, 45.1, 0.2, 8.4, 23.0])
    ranks_desc = RankPermutationSolver.solve(utilities, ascending=False)
    ranks_asc = RankPermutationSolver.solve(utilities, ascending=True)

    # Must contain exact set {1, 2, 3, 4, 5}
    assert set(ranks_desc) == {1, 2, 3, 4, 5}
    assert set(ranks_asc) == {1, 2, 3, 4, 5}

    # In descending mode, highest utility (45.1 at index 1) is rank 1
    assert ranks_desc[1] == 1
    # Lowest utility (0.2 at index 2) is rank 5
    assert ranks_desc[2] == 5


def test_complete_outcome_space_sum_to_one():
    """Verify that complete outcome distribution sums strictly to 1.0."""
    utilities = np.array([20.0, 15.0, 10.0, 5.0, 0.0])
    p_dnf = np.array([0.15, 0.20, 0.10, 0.25, 0.08])
    reliabilities = np.array([0.95, 0.90, 0.85, 0.80, 0.92])

    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
    )

    assert len(outcomes) == 5
    for out in outcomes:
        assert pytest.approx(out["sum_total"], abs=1e-5) == 1.0
        assert out["p_classified_total"] + out["p_mech_dnf"] + out["p_incident_dnf"] + out["p_dns"] + out["p_dsq"] == pytest.approx(1.0, abs=1e-5)
        assert len(out["position_distribution"]) == 5
        assert out["expected_position"] > 0.0


def test_evaluation_metrics_exactness():
    """Verify MAE, Spearman rho, Kendall tau, and Top-K accuracy on known vectors."""
    pred = np.array([1, 2, 3, 4, 5])
    actual = np.array([1, 2, 4, 3, 5])

    mae = calculate_race_mae(pred, actual)
    rho = calculate_spearman_rho(pred, actual)
    tau = calculate_kendall_tau(pred, actual)
    top1 = calculate_top_k_hit_rate(pred, actual, k=1)
    top3 = calculate_top_k_hit_rate(pred, actual, k=3)

    # Positions 3 and 4 swapped (|3-4| + |4-3| = 2 / 5 = 0.4)
    assert pytest.approx(mae, abs=1e-4) == 0.4
    assert 0.8 <= rho <= 1.0
    assert pytest.approx(tau, abs=1e-4) == 0.8
    assert top1 == 1.0
    # Top 3: pred has {0, 1, 2}, actual has {0, 1, 3} -> 2/3 hit rate
    assert pytest.approx(top3, abs=1e-4) == 2.0 / 3.0


def test_latent_fuel_state_inference():
    """Verify latent fuel state inference yields distinct low/high fuel mixtures with uncertainty."""
    from src.features.cpi import infer_stint_latent_fuel_state

    # Short stint with pace near pole (Qualy simulation)
    low_fuel_est = infer_stint_latent_fuel_state(
        stint_laps_count=3,
        median_lap_time_ms=85000.0,
        pole_time_ms=84500.0,
    )
    assert low_fuel_est.p_low_fuel > low_fuel_est.p_high_fuel
    assert low_fuel_est.expected_fuel_kg < 35.0
    assert low_fuel_est.fuel_uncertainty_kg > 0.0

    # Long stint with pace off pole (Race simulation)
    high_fuel_est = infer_stint_latent_fuel_state(
        stint_laps_count=16,
        median_lap_time_ms=90500.0,
        pole_time_ms=84500.0,
    )
    assert high_fuel_est.p_high_fuel > high_fuel_est.p_low_fuel
    assert high_fuel_est.expected_fuel_kg > 45.0
    assert high_fuel_est.fuel_uncertainty_kg > 0.0


def test_bayesian_lmm_posterior_predictions():
    """Verify Hierarchical LMM Bayesian posterior sampling produces valid permutation draws."""
    from src.models.statistical import HierarchicalLMMRanker

    df = pd.DataFrame([
        {"driver_id": "verstappen", "constructor_id": "red_bull", "grid_position": 1.0, "finish_position": 1.0},
        {"driver_id": "norris", "constructor_id": "mclaren", "grid_position": 2.0, "finish_position": 2.0},
        {"driver_id": "leclerc", "constructor_id": "ferrari", "grid_position": 3.0, "finish_position": 3.0},
        {"driver_id": "hamilton", "constructor_id": "ferrari", "grid_position": 4.0, "finish_position": 4.0},
        {"driver_id": "russell", "constructor_id": "mercedes", "grid_position": 5.0, "finish_position": 5.0},
    ])

    lmm = HierarchicalLMMRanker().fit(df)
    draws = lmm.sample_posterior_predictions(df, n_draws=50)

    assert draws.shape == (50, 5)
    for m in range(50):
        assert set(draws[m]) == {1, 2, 3, 4, 5}


def test_complete_outcome_entropy_and_intervals():
    """Verify distribution entropy and 90% credible intervals."""
    utilities = np.array([20.0, 15.0, 10.0, 5.0, 0.0])
    p_dnf = np.array([0.10, 0.15, 0.12, 0.20, 0.08])
    reliabilities = np.array([0.98, 0.95, 0.90, 0.85, 0.92])

    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
    )

    for out in outcomes:
        assert out["distribution_entropy"] > 0.0
        assert 1 <= out["rank_p05"] <= out["rank_p95"] <= 5
        assert "note_multi_car" in out
