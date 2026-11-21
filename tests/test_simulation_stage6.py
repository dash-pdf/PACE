"""PACE: Stage 6 Monte Carlo Simulation & Calibration Engine Test Suite.

Verifies:
1. Deterministic reproducibility under controlled random seeds.
2. Grid-level mathematical consistency (strictly doubly stochastic outcome space).
3. Probability outputs completeness (Win, Podium, Top 5, Points, DNF decomp, mode, intervals).
4. Monte Carlo Standard Error (MCSE) bounds and convergence stability.
5. Probability calibration scoring (Brier score, Log Loss, ECE, Temperature Scaling, Guardrails).
6. Macro simulation realism against empirical Formula 1 benchmarks.
7. Walk-forward backtesting integration with PACE Monte Carlo simulator.
"""

import pytest
import numpy as np
import pandas as pd

from src.simulation.monte_carlo import MonteCarloSimulationEngine, MonteCarloResult
from src.simulation.realism import SimulationRealismAuditor, HISTORICAL_F1_BENCHMARKS
from src.evaluation.calibration import (
    brier_score,
    binary_log_loss,
    multiclass_log_loss,
    compute_reliability_curve,
    TemperatureScaler,
    ProbabilityCalibrationEvaluator,
)
from src.evaluation.backtest import WalkForwardBacktester
from src.features.pipeline import FeaturePipeline


@pytest.fixture
def sample_event_features() -> pd.DataFrame:
    """Creates a synthetic 22-car grid with differentiated utilities and grid positions."""
    drivers = [
        "russell", "antonelli", "norris", "piastri", "hamilton", "max_verstappen",
        "hadjar", "leclerc", "arvid_lindblad", "colapinto", "hulkenberg", "lawson",
        "bortoleto", "gasly", "sainz", "bearman", "ocon", "albon",
        "alonso", "perez", "stroll", "bottas",
    ]
    constructors = [
        "mercedes", "mercedes", "mclaren", "mclaren", "ferrari", "red_bull",
        "red_bull", "ferrari", "rb", "alpine", "audi", "rb",
        "audi", "alpine", "williams", "haas", "haas", "williams",
        "aston_martin", "cadillac", "aston_martin", "cadillac",
    ]
    utilities = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.3, 0.4, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2]
    cvs = [0.0028, 0.0032, 0.0029, 0.0030, 0.0027, 0.0025, 0.0034, 0.0031, 0.0035, 0.0033, 0.0032, 0.0034, 0.0033, 0.0032, 0.0030, 0.0035, 0.0033, 0.0031, 0.0026, 0.0034, 0.0036, 0.0032]

    return pd.DataFrame([
        {
            "driver_id": d,
            "constructor_id": c,
            "grid_position": i + 1,
            "predicted_utility": u,
            "driver_consistency_cv": cv,
            "cpi_reliability": 0.96 if c in ("mercedes", "ferrari", "mclaren") else 0.89,
            "circuit_id": "villeneuve",
            "circuit_length_km": 4.361,
            "tyre_stress_level": 3.0,
        }
        for i, (d, c, u, cv) in enumerate(zip(drivers, constructors, utilities, cvs))
    ])


def test_monte_carlo_reproducibility_and_seeding(sample_event_features):
    """Verify identical random seeds produce bitwise-identical probability matrices."""
    engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=1)

    res1 = engine.run_simulations(
        sample_event_features,
        n_simulations=15,
        total_laps=30,
        circuit_id="villeneuve",
        random_seed=42,
    )
    res2 = engine.run_simulations(
        sample_event_features,
        n_simulations=15,
        total_laps=30,
        circuit_id="villeneuve",
        random_seed=42,
    )

    np.testing.assert_allclose(res1.position_matrix, res2.position_matrix, atol=1e-12)
    assert res1.p_win == res2.p_win
    assert res1.expected_position == res2.expected_position
    assert res1.most_likely_position == res2.most_likely_position


def test_monte_carlo_grid_level_mathematical_consistency(sample_event_features):
    """Verify strict doubly-stochastic permutation property and probability axioms."""
    engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=1)

    res = engine.run_simulations(
        sample_event_features,
        n_simulations=25,
        total_laps=58,
        circuit_id="villeneuve",
        random_seed=123,
    )

    check = res.verify_mathematical_consistency()
    assert check["is_valid"] is True
    assert check["driver_probabilities_sum_to_one"] is True
    assert check["position_probabilities_sum_to_one"] is True
    assert check["probabilities_in_unit_interval"] is True
    assert check["dnf_decomposition_consistent"] is True
    assert check["is_doubly_stochastic"] is True
    assert check["max_row_sum_error"] < 1e-12
    assert check["max_col_sum_error"] < 1e-12

    # Verify percentiles ordering: P05 <= P10 <= P25 <= P50 <= P75 <= P90 <= P95
    for d in res.drivers:
        pt = res.percentiles[d]
        assert pt["P05"] <= pt["P10"] <= pt["P25"] <= pt["P50"] <= pt["P75"] <= pt["P90"] <= pt["P95"]
        assert pt["IQR"] == pytest.approx(pt["P75"] - pt["P25"])
        assert res.entropy[d] >= 0.0


def test_monte_carlo_convergence_and_mcse_bounds(sample_event_features):
    """Verify Monte Carlo Standard Error (MCSE) scales as O(1/sqrt(N))."""
    engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=1)

    # 40 runs vs 160 runs (4x increase in N -> ~2x reduction in MCSE)
    res_small = engine.run_simulations(
        sample_event_features, n_simulations=40, total_laps=20, random_seed=42
    )
    res_large = engine.run_simulations(
        sample_event_features, n_simulations=160, total_laps=20, random_seed=42
    )

    # For top driver, compare podium MCSE
    d_top = res_small.drivers[0]
    mcse_small = res_small.mcse[d_top]["podium"]
    mcse_large = res_large.mcse[d_top]["podium"]

    if mcse_small > 0.02 and mcse_large > 0.01:
        ratio = mcse_small / mcse_large
        assert 1.4 <= ratio <= 2.8  # Theoretical sqrt(4) = 2.0


def test_probability_outputs_completeness(sample_event_features):
    """Verify all required probability and uncertainty fields exist in exported DataFrame."""
    engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=1)
    res = engine.run_simulations(
        sample_event_features, n_simulations=20, total_laps=58, random_seed=99
    )
    df = res.to_dataframe()

    required_columns = [
        "predicted_rank", "driver_id", "constructor_id", "grid_position",
        "expected_position", "expected_classified", "most_likely_position",
        "median_position", "p05", "p95", "p10", "p90", "iqr", "std_dev", "entropy",
        "p_win", "p_podium", "p_top5", "p_points",
        "p_dnf_mech", "p_dnf_inc", "p_dnf_total", "p_classified",
        "mcse_win", "mcse_podium",
    ]
    for col in required_columns:
        assert col in df.columns

    assert len(df) == 22
    assert list(df["predicted_rank"]) == list(range(1, 23))
    # Ranking should be strictly monotonically non-decreasing by expected_position
    assert df["expected_position"].is_monotonic_increasing


def test_probability_calibration_metrics():
    """Verify Brier score, log loss, reliability diagram, temperature scaling, and guardrails."""
    # 1. Brier score exactness
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.1, 0.9, 0.2])
    # Diff: (0.2)^2 + (0.1)^2 + (0.1)^2 + (0.2)^2 = 0.04 + 0.01 + 0.01 + 0.04 = 0.10 / 4 = 0.025
    assert brier_score(y_true, y_prob) == pytest.approx(0.025)

    # 2. Binary log loss
    ll = binary_log_loss(y_true, y_prob)
    assert ll > 0.0

    # 3. Reliability Curve & ECE
    rel = compute_reliability_curve(y_true, y_prob, n_bins=5)
    assert "ece" in rel
    assert "mce" in rel
    assert rel["ece"] >= 0.0

    # 4. Temperature Scaler
    np.random.seed(42)
    pos_matrix = np.random.dirichlet(np.ones(10), size=30)
    act_pos = np.random.randint(1, 11, size=30)

    scaler = TemperatureScaler().fit(act_pos, pos_matrix)
    assert scaler.is_fitted is True
    scaled_matrix = scaler.transform(pos_matrix)
    assert scaled_matrix.shape == pos_matrix.shape
    np.testing.assert_allclose(scaled_matrix.sum(axis=1), np.ones(30), atol=1e-5)

    # 5. Overfitting Guardrails
    # Small sample (<50) must reject post-hoc calibration
    applies, reason = ProbabilityCalibrationEvaluator.evaluate_calibration_guardrails(
        sample_size=30, uncalibrated_logloss=2.5, calibrated_logloss=2.3
    )
    assert applies is False
    assert "below minimum required threshold" in reason

    # Sufficient sample and significant gain must accept calibration
    applies_ok, _ = ProbabilityCalibrationEvaluator.evaluate_calibration_guardrails(
        sample_size=120, uncalibrated_logloss=2.5, calibrated_logloss=2.2
    )
    assert applies_ok is True


def test_simulation_realism_event_frequencies(sample_event_features):
    """Verify macro event frequencies fall within empirical Formula 1 historical ranges."""
    engine = MonteCarloSimulationEngine(utility_scaling_factor=0.05, n_jobs=1)
    res = engine.run_simulations(
        sample_event_features, n_simulations=20, total_laps=58, random_seed=42
    )

    audit = SimulationRealismAuditor.audit_event_frequencies(res)
    assert "detailed_audit" in audit
    det = audit["detailed_audit"]

    # Mechanical and incident DNFs must fall within plausible boundaries
    assert det["mechanical_dnfs_per_race"]["in_plausible_range"] is True
    assert det["safety_cars_per_race"]["in_plausible_range"] is True
    assert det["pit_stops_per_race"]["in_plausible_range"] is True

    # Check stability function
    stability = SimulationRealismAuditor.evaluate_distribution_stability(res.convergence_traces)
    assert "is_stable" in stability


def test_walk_forward_backtest_with_monte_carlo():
    """Verify WalkForwardBacktester successfully executes PACE Monte Carlo simulations."""
    rr = pd.read_parquet("data/historical/historical_race_results.parquet")
    q = pd.read_parquet("data/historical/historical_qualifying.parquet")
    laps = pd.read_parquet("data/historical/historical_laps.parquet")
    app = pd.read_parquet("data/historical/driver_circuit/driver_circuit_appearances.parquet")
    circ = pd.read_parquet("data/historical/driver_circuit/circuit_characteristics.parquet")

    pipeline = FeaturePipeline(rr, q, laps, app, circ)
    backtester = WalkForwardBacktester(
        pipeline=pipeline,
        min_train_events=5,
        include_monte_carlo=True,
        mc_simulations=20,
        n_jobs=1,
    )

    # Evaluate 2 events for rapid integration testing
    results = backtester.run_backtest(max_events=2)

    summary_df = results["summary_df"]
    assert "PACE_MonteCarlo_Simulation" in summary_df["Model"].values
    assert "Baseline_GridOrder" in summary_df["Model"].values
    assert "ML_XGBoost_LambdaMART" in summary_df["Model"].values

    mc_row = summary_df[summary_df["Model"] == "PACE_MonteCarlo_Simulation"].iloc[0]
    assert mc_row["MAE_Mean"] > 0.0
    assert 0.0 <= mc_row["Top1_Accuracy"] <= 1.0
    assert 0.0 <= mc_row["Brier_Win"] <= 1.0

    significance = results["significance"]
    assert "PACE_MonteCarlo_vs_GridBaseline" in significance
    assert "PACE_MonteCarlo_vs_LambdaMART" in significance
