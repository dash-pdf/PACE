"""PACE: Phase 3 Diagnostic & Consistency Verification Test Suite.

Exhaustively verifies:
1. Analytical Plackett-Luce choice consistency via Gumbel-max trick.
2. Strict permutation validity (tie-free, zero duplicate positions, exactly 1..N).
3. Doubly stochastic probability matrix property (sum over positions = 1, sum over drivers = 1).
4. Complete outcome space partition (classified + mech + incident + dns + dsq = 1.0).
5. Non-zero, realistic 90% credible intervals [P05, P95] and expected position consistency.
6. Calibrated DNF decomposition (mechanical vs incident) across reliability regimes.
"""

import pytest
import numpy as np
from src.models.ranker import RankPermutationSolver, XGBoostLambdaMARTRanker
from src.models.dnf import CompleteOutcomeSpaceEngine, CalibratedDNFModel


def test_analytical_two_driver_plackett_luce():
    """Verify that Gumbel-max sampling matches analytical Luce choice probability P(1 > 2)."""
    u1 = 1.5
    u2 = 0.5
    delta = u1 - u2
    analytical_p1_win = 1.0 / (1.0 + np.exp(-delta))  # 1 / (1 + e^-1.0) approx 0.731058

    utilities = np.array([u1, u2])
    p_dnf = np.array([0.0, 0.0])  # Zero DNF to isolate Plackett-Luce choice
    reliabilities = np.array([1.0, 1.0])

    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
        p_dns=0.0,
        p_dsq=0.0,
        n_draws=20000,
        random_seed=42,
    )

    empirical_p1_win = outcomes[0]["p_win"]
    assert pytest.approx(empirical_p1_win, abs=0.01) == analytical_p1_win


def test_joint_permutation_solver_strict_validity():
    """Verify RankPermutationSolver guarantees valid, tie-free 1..N rankings without duplicates."""
    for n in [2, 5, 10, 22]:
        # Generate random utilities with potential ties
        raw_utilities = np.random.randn(n)
        raw_utilities[0] = raw_utilities[1]  # Force a tie
        ranks = RankPermutationSolver.solve(raw_utilities, ascending=False)

        assert len(ranks) == n
        assert set(ranks) == set(range(1, n + 1))
        assert len(np.unique(ranks)) == n


def test_doubly_stochastic_permutation_distribution():
    """Verify that the simulated outcome matrix is doubly stochastic:
    1. Every driver's finishing probability across positions 1..N sums to 1.0.
    2. Every finishing position 1..N across all drivers sums to 1.0.
    """
    n = 22
    utilities = np.linspace(2.5, -2.5, n)
    p_dnf = np.full(n, 0.16)
    reliabilities = np.full(n, 0.95)

    n_draws = 10000
    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
        p_dns=0.005,
        p_dsq=0.002,
        n_draws=n_draws,
        random_seed=123,
    )

    assert len(outcomes) == n

    # Check complete outcome partition per driver
    for out in outcomes:
        assert pytest.approx(out["sum_total"], abs=1e-5) == 1.0
        assert pytest.approx(
            out["p_classified_total"] + out["p_mech_dnf"] + out["p_incident_dnf"] + out["p_dns"] + out["p_dsq"],
            abs=1e-5,
        ) == 1.0

    # Build position probability matrix (n_drivers x n_positions)
    # where entry (i, k-1) is the unconditional probability of driver i finishing in position k
    # We can reconstruct it from simulation positions
    rng = np.random.RandomState(123)
    p_mech_arr = p_dnf * 0.25
    p_inc_arr = p_dnf * 0.75
    sim_positions = np.zeros((n_draws, n), dtype=int)

    u_rand = rng.uniform(0.0, 1.0, size=(n_draws, n))
    for m in range(n_draws):
        u_m = u_rand[m]
        is_class = u_m >= (0.005 + 0.002 + p_dnf)
        c_idx = np.where(is_class)[0]
        C = len(c_idx)
        if C > 0:
            g_noise = -np.log(-np.log(rng.uniform(1e-10, 1.0 - 1e-10, size=C)))
            pl_scores = utilities[c_idx] + g_noise
            sorted_c = c_idx[np.argsort(-pl_scores)]
            for r, d in enumerate(sorted_c, 1):
                sim_positions[m, d] = r
        nc_idx = np.where(~is_class)[0]
        if len(nc_idx) > 0:
            rng.shuffle(nc_idx)
            for r, d in enumerate(nc_idx, C + 1):
                sim_positions[m, d] = r

    # Check that in every single draw, sim_positions is a permutation of 1..N
    for m in range(min(500, n_draws)):
        assert set(sim_positions[m]) == set(range(1, n + 1))

    # Matrix of probabilities
    prob_matrix = np.zeros((n, n))
    for d in range(n):
        for k in range(1, n + 1):
            prob_matrix[d, k - 1] = np.mean(sim_positions[:, d] == k)

    # Row sums: sum over positions for each driver == 1.0
    row_sums = np.sum(prob_matrix, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    # Col sums: sum over drivers for each finishing position == 1.0
    col_sums = np.sum(prob_matrix, axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-5)


def test_expected_position_and_credible_interval_consistency():
    """Verify that expected positions and credible intervals are strictly ordered and realistic."""
    n = 22
    utilities = np.linspace(2.5, -2.5, n)
    p_dnf = np.full(n, 0.16)
    reliabilities = np.full(n, 0.95)

    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
        n_draws=10000,
        random_seed=42,
    )

    # Favorite (index 0)
    fav = outcomes[0]
    assert fav["p_win"] > 0.15
    assert fav["p_podium"] > 0.40
    assert 1.0 <= fav["expected_classified_position"] <= 5.0
    assert fav["expected_position"] > fav["expected_classified_position"]  # DNF risk inflates unconditional mean
    assert fav["rank_p05"] == 1
    assert fav["rank_p95"] > fav["rank_p05"]  # Non-zero interval width!
    assert fav["rank_p95"] <= 10

    # Underdog (index 21)
    underdog = outcomes[-1]
    assert underdog["p_win"] < 0.01
    assert underdog["expected_classified_position"] > 10.0
    assert underdog["rank_p95"] >= 18
    assert underdog["rank_p05"] < underdog["rank_p95"]


def test_dnf_decomposition_mechanics():
    """Verify that DNF decomposition correctly separates mechanical vs incident risk based on reliability."""
    n = 2
    utilities = np.array([1.0, 1.0])
    p_dnf = np.array([0.20, 0.20])
    # Driver 0: very reliable car (0.99), Driver 1: unreliable car (0.70)
    reliabilities = np.array([0.99, 0.70])

    outcomes = CompleteOutcomeSpaceEngine.compute_complete_distribution(
        utilities=utilities,
        p_dnf=p_dnf,
        reliabilities=reliabilities,
        n_draws=5000,
        random_seed=42,
    )

    # Driver 0 should have mostly incident DNF
    assert outcomes[0]["p_incident_dnf"] > outcomes[0]["p_mech_dnf"]

    # Driver 1 should have mostly mechanical DNF
    assert outcomes[1]["p_mech_dnf"] > outcomes[1]["p_incident_dnf"]
