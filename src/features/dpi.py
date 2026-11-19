"""PACE: Driver Performance Index (DPI) Module.

Implements candidate driver capability models:
1. Glicko-2 networked paired-comparison engine with rating deviations (uncertainty).
2. Teammate-relative qualifying and clean-air race pace deltas with confounder controls.
3. Decoupled driver experience (uncertainty reduction & rookie learning curve).
4. Lap time consistency (coefficient of variation).
5. Bradley-Terry paired choice model with covariate adjustments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Glicko-2 Rating Engine
# -----------------------------------------------------------------------------

@dataclass
class Glicko2Rating:
    """Represents a driver's Glicko-2 rating distribution."""
    rating: float = 1500.0          # Latent skill point estimate (mu)
    rd: float = 350.0               # Rating deviation / epistemic uncertainty (phi)
    volatility: float = 0.06        # Volatility / aleatoric performance variance (sigma)
    starts: int = 0                 # Career race starts count

    @property
    def mu(self) -> float:
        """Glicko-2 internal scale rating."""
        return (self.rating - 1500.0) / 173.7178

    @property
    def phi(self) -> float:
        """Glicko-2 internal scale rating deviation."""
        return self.rd / 173.7178


class Glicko2Engine:
    """Glicko-2 Networked Paired Comparison Engine.
    
    Treats teammate comparisons within each race/qualifying session as matches.
    Propagates ratings across teams as drivers switch constructors over time.
    """

    def __init__(self, tau: float = 0.5, default_rd: float = 350.0, default_vol: float = 0.06):
        self.tau = tau  # System constant constraining volatility change over time
        self.default_rd = default_rd
        self.default_vol = default_vol
        self.ratings: Dict[str, Glicko2Rating] = {}

    def get_rating(self, driver_id: str) -> Glicko2Rating:
        """Retrieve existing rating or initialize new driver with default prior."""
        if driver_id not in self.ratings:
            self.ratings[driver_id] = Glicko2Rating(
                rating=1500.0,
                rd=self.default_rd,
                volatility=self.default_vol,
                starts=0,
            )
        return self.ratings[driver_id]

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + (3.0 * (phi ** 2)) / (math.pi ** 2))

    @staticmethod
    def _e(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-Glicko2Engine._g(phi_j) * (mu - mu_j)))

    def update_period(
        self,
        matches: List[Tuple[str, str, float]],  # (driver_a, driver_b, score_a: 1.0=win, 0.5=tie, 0.0=loss)
    ) -> None:
        """Update ratings for a discrete rating period (e.g., after an event)."""
        # Collect opponents and results per driver
        driver_matches: Dict[str, List[Tuple[str, float]]] = {}
        for d1, d2, s in matches:
            driver_matches.setdefault(d1, []).append((d2, s))
            driver_matches.setdefault(d2, []).append((d1, 1.0 - s))

        updated_ratings: Dict[str, Glicko2Rating] = {}

        for driver_id, opponents in driver_matches.items():
            curr = self.get_rating(driver_id)
            mu = curr.mu
            phi = curr.phi
            sigma = curr.volatility

            # Step 1: Compute variance v and delta
            v_inv = 0.0
            delta_sum = 0.0

            for opp_id, score in opponents:
                opp = self.get_rating(opp_id)
                g_phi = self._g(opp.phi)
                e_val = self._e(mu, opp.mu, opp.phi)
                v_inv += (g_phi ** 2) * e_val * (1.0 - e_val)
                delta_sum += g_phi * (score - e_val)

            if v_inv == 0.0:
                # No competitive outcomes this period
                continue

            v = 1.0 / v_inv
            delta = v * delta_sum

            # Step 2: Determine new volatility sigma' via Illinois algorithm
            a = math.log(sigma ** 2)
            tau = self.tau

            def f(x: float) -> float:
                e_x = math.exp(x)
                d_sq = delta ** 2
                phi_sq = phi ** 2
                denom = 2.0 * ((phi_sq + v + e_x) ** 2)
                term1 = (e_x * (d_sq - phi_sq - v - e_x)) / denom
                term2 = (x - a) / (tau ** 2)
                return term1 - term2

            A = a
            if delta ** 2 > (phi ** 2 + v):
                B = math.log(delta ** 2 - phi ** 2 - v)
            else:
                k = 1
                while f(a - k * tau) < 0:
                    k += 1
                B = a - k * tau

            f_A = f(A)
            f_B = f(B)

            # Illinois bracket method
            for _ in range(50):
                if abs(B - A) <= 1e-6:
                    break
                C = A + (A - B) * f_A / (f_B - f_A)
                f_C = f(C)
                if f_C * f_B <= 0:
                    A = B
                    f_A = f_B
                else:
                    f_A = f_A / 2.0
                B = C
                f_B = f_C

            sigma_prime = math.exp(A / 2.0)

            # Step 3: Update rating deviation and rating
            phi_star = math.sqrt(phi ** 2 + sigma_prime ** 2)
            phi_prime = 1.0 / math.sqrt((1.0 / (phi_star ** 2)) + (1.0 / v))
            mu_prime = mu + (phi_prime ** 2) * delta_sum

            # Convert back to original scale
            new_rating = 173.7178 * mu_prime + 1500.0
            new_rd = 173.7178 * phi_prime

            updated_ratings[driver_id] = Glicko2Rating(
                rating=new_rating,
                rd=new_rd,
                volatility=sigma_prime,
                starts=curr.starts + 1,
            )

        # Apply updates
        for driver_id, new_r in updated_ratings.items():
            self.ratings[driver_id] = new_r


# -----------------------------------------------------------------------------
# 2. Teammate Pace Delta Calculators
# -----------------------------------------------------------------------------

def calculate_teammate_qualifying_deltas(
    qualifying_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate qualifying pace deltas between teammates for each race weekend.
    
    Identifies pairs of teammates, finds their best mutually reached qualifying session
    (Q3, Q2, or Q1), and computes:
    - delta_q_pct: (T_driver - T_teammate) / min(T_driver, T_teammate)
    - h2h_qual_win: 1.0 if driver qualified ahead of teammate, 0.0 otherwise.
    """
    records = []
    required_cols = {"season", "round", "driver_id", "constructor_id", "qualifying_position"}
    if not required_cols.issubset(qualifying_df.columns):
        return pd.DataFrame()

    grouped = qualifying_df.groupby(["season", "round", "constructor_id"])

    for (season, rnd, constructor_id), team_df in grouped:
        if len(team_df) < 2:
            continue

        drivers = team_df.to_dict("records")
        # Compare first two teammates
        d1, d2 = drivers[0], drivers[1]

        # Determine best mutually completed session
        t1, t2 = None, None
        for q_time in ["q3_time_ms", "q2_time_ms", "q1_time_ms"]:
            val1 = d1.get(q_time)
            val2 = d2.get(q_time)
            if val1 is not None and val2 is not None and not pd.isna(val1) and not pd.isna(val2):
                if val1 > 0 and val2 > 0:
                    t1, t2 = float(val1), float(val2)
                    break

        if t1 is not None and t2 is not None:
            min_t = min(t1, t2)
            delta1 = (t1 - t2) / min_t
            delta2 = (t2 - t1) / min_t
        else:
            delta1, delta2 = 0.0, 0.0

        pos1 = d1["qualifying_position"]
        pos2 = d2["qualifying_position"]

        h2h_1 = 1.0 if pos1 < pos2 else (0.5 if pos1 == pos2 else 0.0)
        h2h_2 = 1.0 - h2h_1

        records.append({
            "season": season,
            "round": rnd,
            "constructor_id": constructor_id,
            "driver_id": d1["driver_id"],
            "teammate_id": d2["driver_id"],
            "qualifying_delta_pct": delta1,
            "h2h_qual_win": h2h_1,
            "driver_qual_pos": pos1,
            "teammate_qual_pos": pos2,
        })
        records.append({
            "season": season,
            "round": rnd,
            "constructor_id": constructor_id,
            "driver_id": d2["driver_id"],
            "teammate_id": d1["driver_id"],
            "qualifying_delta_pct": delta2,
            "h2h_qual_win": h2h_2,
            "driver_qual_pos": pos2,
            "teammate_qual_pos": pos1,
        })

    return pd.DataFrame(records)


def calculate_teammate_race_deltas(
    race_results_df: pd.DataFrame,
    laps_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculate teammate race deltas with incident, collision, and clean-air controls.
    
    If both teammates are classified:
    - h2h_race_win = 1.0 if finish_position < teammate_finish_position
    - delta_pos = teammate_finish_position - driver_finish_position
    
    If a driver suffers an unforced crash/collision, they are scored with a loss (0.0).
    If a driver suffers an asymmetric mechanical failure, the comparison is flagged as confounded.
    """
    records = []
    grouped = race_results_df.groupby(["season", "round", "constructor_id"])

    for (season, rnd, constructor_id), team_df in grouped:
        if len(team_df) < 2:
            continue

        drivers = team_df.to_dict("records")
        d1, d2 = drivers[0], drivers[1]

        c1, c2 = d1.get("is_classified", False), d2.get("is_classified", False)
        dnf1, dnf2 = d1.get("is_dnf", False), d2.get("is_dnf", False)
        status1, status2 = str(d1.get("status", "")).lower(), str(d2.get("status", "")).lower()

        is_collision1 = any(w in status1 for w in ["collision", "accident", "spun off", "crash"])
        is_collision2 = any(w in status2 for w in ["collision", "accident", "spun off", "crash"])

        is_mech1 = dnf1 and not is_collision1
        is_mech2 = dnf2 and not is_collision2

        confounded = False
        if c1 and c2:
            p1, p2 = d1["finish_position"], d2["finish_position"]
            h2h_1 = 1.0 if p1 < p2 else (0.5 if p1 == p2 else 0.0)
            h2h_2 = 1.0 - h2h_1
            pos_delta1 = p2 - p1
            pos_delta2 = p1 - p2
        elif is_mech1 and not is_mech2:
            # d1 mechanical failure -> non-comparable / confounded
            confounded = True
            h2h_1, h2h_2 = 0.5, 0.5
            pos_delta1, pos_delta2 = 0, 0
        elif is_mech2 and not is_mech1:
            confounded = True
            h2h_1, h2h_2 = 0.5, 0.5
            pos_delta1, pos_delta2 = 0, 0
        elif is_collision1 and not is_collision2:
            # d1 driver error
            h2h_1, h2h_2 = 0.0, 1.0
            pos_delta1, pos_delta2 = -10, 10
        elif is_collision2 and not is_collision1:
            h2h_1, h2h_2 = 1.0, 0.0
            pos_delta1, pos_delta2 = 10, -10
        else:
            # Both retired or undefined
            confounded = True
            h2h_1, h2h_2 = 0.5, 0.5
            pos_delta1, pos_delta2 = 0, 0

        records.append({
            "season": season,
            "round": rnd,
            "constructor_id": constructor_id,
            "driver_id": d1["driver_id"],
            "teammate_id": d2["driver_id"],
            "h2h_race_win": h2h_1,
            "race_pos_delta": pos_delta1,
            "is_confounded": confounded,
        })
        records.append({
            "season": season,
            "round": rnd,
            "constructor_id": constructor_id,
            "driver_id": d2["driver_id"],
            "teammate_id": d1["driver_id"],
            "h2h_race_win": h2h_2,
            "race_pos_delta": pos_delta2,
            "is_confounded": confounded,
        })

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 3. Decoupled Experience & Consistency Metrics
# -----------------------------------------------------------------------------

def calculate_rookie_learning_factor(starts: int, tau_learn: float = 25.0) -> float:
    """Non-linear rookie learning rate function: f_learn(starts) = 1.0 - exp(-starts / tau).
    
    Provides marginal performance adaptation benefit during the rookie season (starts <= 25),
    reaching an asymptote (f_learn approx 1.0) beyond 50 career starts.
    Does NOT provide additive advantage to veterans over sophomores.
    """
    if starts <= 0:
        return 0.0
    return 1.0 - math.exp(-float(starts) / tau_learn)


def calculate_pace_consistency_cv(laps_df: pd.DataFrame, driver_id: str) -> Tuple[float, float]:
    """Calculate lap time consistency metrics:
    - cv_pace: Coefficient of variation (std / mean) on clean green-flag laps.
    - lap_count: Valid clean laps analyzed.
    
    Excludes in/out laps and outlier safety-car laps (> 115% of median lap time).
    """
    d_laps = laps_df[laps_df["driver_id"] == driver_id]
    if len(d_laps) < 5 or "lap_time_ms" not in d_laps.columns:
        return 0.05, 0.0  # Default neutral consistency, 0 sample size

    valid_times = d_laps["lap_time_ms"].dropna()
    valid_times = valid_times[valid_times > 0]
    if len(valid_times) < 5:
        return 0.05, 0.0

    med = valid_times.median()
    # Filter out SC / VSC / in-lap outliers (> 115% of median)
    clean_laps = valid_times[(valid_times >= 0.90 * med) & (valid_times <= 1.15 * med)]
    if len(clean_laps) < 5:
        return 0.05, float(len(clean_laps))

    mean_t = clean_laps.mean()
    std_t = clean_laps.std()
    cv = float(std_t / mean_t) if mean_t > 0 else 0.05
    return cv, float(len(clean_laps))


# -----------------------------------------------------------------------------
# 4. Bradley-Terry Driver Rating Alternative
# -----------------------------------------------------------------------------

class BradleyTerryEngine:
    """Bradley-Terry Paired Choice Model with intra-team comparisons.
    
    Models logit(P(i beats j | team)) = alpha_i - alpha_j + z_ij * delta.
    Provides an interpretable statistical baseline to compare against Glicko-2.
    """

    def __init__(self, regularization: float = 1.0):
        self.regularization = regularization
        self.driver_strengths: Dict[str, float] = {}

    def fit(self, matches: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """Fit driver log-ability parameters alpha_i via iterative MM algorithm."""
        drivers = sorted(list(set([m[0] for m in matches] + [m[1] for m in matches])))
        if not drivers:
            return {}

        idx_map = {d: i for i, d in enumerate(drivers)}
        n = len(drivers)

        # Wins matrix W[i, j] and total paired matches N[i, j]
        W = np.zeros((n, n), dtype=float)
        for d1, d2, s in matches:
            i, j = idx_map[d1], idx_map[d2]
            W[i, j] += s
            W[j, i] += (1.0 - s)

        # Initial ability parameters p_i = exp(alpha_i)
        p = np.ones(n, dtype=float)
        wins_i = W.sum(axis=1) + self.regularization

        for _ in range(100):
            p_old = p.copy()
            for i in range(n):
                denom = 0.0
                for j in range(n):
                    if i != j and (W[i, j] + W[j, i]) > 0:
                        denom += (W[i, j] + W[j, i]) / (p_old[i] + p_old[j])
                denom += self.regularization / (p_old[i] + 1.0)
                if denom > 0:
                    p[i] = wins_i[i] / denom
            # Normalize sum(p) = n
            p = p * (n / p.sum())
            if np.max(np.abs(p - p_old)) < 1e-5:
                break

        # Log abilities normalized with mean 0
        log_abilities = np.log(p)
        log_abilities -= np.mean(log_abilities)

        self.driver_strengths = {d: float(log_abilities[idx_map[d]]) for d in drivers}
        return self.driver_strengths
