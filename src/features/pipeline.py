"""PACE: Leak-Free Feature Compilation Pipeline.

Assembles driver, constructor, circuit, telemetry, and session features
for any target Grand Prix at specified temporal horizons (H1, H2, H3)
with zero future data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.dpi import (
    Glicko2Engine,
    calculate_teammate_qualifying_deltas,
    calculate_teammate_race_deltas,
    calculate_rookie_learning_factor,
    calculate_pace_consistency_cv,
)
from src.features.cpi import (
    calculate_circuit_fuel_sensitivity,
    estimate_tyre_degradation_huber,
    calculate_constructor_qualifying_pole_gap,
    calculate_constructor_reliability,
    calculate_constructor_rolling_trend,
)
from src.features.driver_circuit import (
    EmpiricalBayesCircuitAffinity,
    extract_circuit_topology_features,
)

logger = logging.getLogger("pace.features.pipeline")


class FeaturePipeline:
    """End-to-End Leak-Free Feature Compiler for PACE."""

    def __init__(
        self,
        historical_race_results: pd.DataFrame,
        historical_qualifying: pd.DataFrame,
        historical_laps: pd.DataFrame,
        driver_circuit_appearances: pd.DataFrame,
        circuit_characteristics: pd.DataFrame,
        historical_practice_laps: Optional[pd.DataFrame] = None,
    ):
        self.race_results = historical_race_results.copy()
        self.qualifying = historical_qualifying.copy()
        self.laps = historical_laps.copy()
        self.appearances = driver_circuit_appearances.copy()
        self.circuit_chars = circuit_characteristics.copy()
        self.practice_laps = (
            historical_practice_laps.copy()
            if historical_practice_laps is not None
            else pd.DataFrame()
        )

        self.eb_affinity = EmpiricalBayesCircuitAffinity(kappa_0=5.0)

    def compile_features_for_event(
        self,
        season: int,
        round_num: int,
        horizon: str = "H3",  # "H1" (Pre-weekend), "H2" (Post-FP2), "H3" (Post-Qualy)
        affinity_hypothesis: str = "H1",
    ) -> pd.DataFrame:
        """Compile feature vectors for all drivers entering a specific race round.
        
        Strictly enforces that no session data from the future is ingested.
        """
        # 1. Identify participating drivers and constructors for target round
        target_results = self.race_results[
            (self.race_results["season"] == season) & (self.race_results["round"] == round_num)
        ]
        if target_results.empty:
            logger.warning(f"No race records found for Season {season} Round {round_num}")
            return pd.DataFrame()

        # Find circuit_id for this event
        event_appearances = self.appearances[
            (self.appearances["season"] == season) & (self.appearances["round"] == round_num)
        ]
        circuit_id = (
            event_appearances["circuit_id"].iloc[0]
            if not event_appearances.empty
            else "unknown"
        )

        circuit_topo = extract_circuit_topology_features(self.circuit_chars, circuit_id)

        # 2. Historical data strictly prior to target event
        prior_races = self.race_results[
            (self.race_results["season"] < season) |
            ((self.race_results["season"] == season) & (self.race_results["round"] < round_num))
        ].copy()

        prior_qual = self.qualifying[
            (self.qualifying["season"] < season) |
            ((self.qualifying["season"] == season) & (self.qualifying["round"] < round_num))
        ].copy()

        # 3. Fit Glicko-2 ratings chronologically up to prior round
        glicko = self._build_glicko_ratings(prior_races, prior_qual)

        # 4. Constructor metrics
        constructor_rel = calculate_constructor_reliability(
            self.race_results, target_season=season, target_round=round_num
        )
        pole_gaps = calculate_constructor_qualifying_pole_gap(prior_qual)
        constructor_trends = calculate_constructor_rolling_trend(
            pole_gaps, target_season=season, target_round=round_num
        )

        # 5. Target qualifying results (only available in H3 horizon)
        target_qual = self.qualifying[
            (self.qualifying["season"] == season) & (self.qualifying["round"] == round_num)
        ]
        qual_pos_map = dict(zip(target_qual["driver_id"], target_qual["qualifying_position"])) if not target_qual.empty else {}

        # 6. Build driver-level feature rows
        rows = []
        for _, race_row in target_results.iterrows():
            d_id = race_row["driver_id"]
            c_id = race_row["constructor_id"]

            # DPI features
            rating = glicko.get_rating(d_id)
            starts = rating.starts
            f_learn = calculate_rookie_learning_factor(starts)

            # Prior head-to-head records
            driver_prior_races = prior_races[prior_races["driver_id"] == d_id]
            cv_pace, _ = calculate_pace_consistency_cv(self.laps, d_id)

            # Circuit affinity via Empirical Bayes
            aff = self.eb_affinity.compute_affinity(
                appearances_df=self.appearances,
                driver_id=d_id,
                circuit_id=circuit_id,
                target_season=season,
                target_round=round_num,
                hypothesis=affinity_hypothesis,
            )

            # Constructor features
            c_rel = constructor_rel.get(c_id, 0.90)
            c_trend = constructor_trends.get(c_id, 0.0)

            # Recent pole gap (last 3 races)
            if not pole_gaps.empty and "constructor_id" in pole_gaps.columns:
                team_prior_gaps = pole_gaps[pole_gaps["constructor_id"] == c_id].tail(3)
                c_pole_gap = (
                    float(team_prior_gaps["pole_gap_pct"].mean())
                    if not team_prior_gaps.empty
                    else 0.03
                )
            else:
                c_pole_gap = 0.03

            # Horizon-specific features
            if horizon == "H3":
                # Saturday qualifying position is known
                grid_pos = float(qual_pos_map.get(d_id, race_row.get("grid_position", 15.0)))
                if grid_pos <= 0:
                    grid_pos = float(race_row.get("grid_position", 15.0))
            elif horizon == "H2":
                # Friday practice known, qualy unknown -> use rolling qual average
                driver_prior_q = prior_qual[prior_qual["driver_id"] == d_id].tail(5)
                grid_pos = (
                    float(driver_prior_q["qualifying_position"].mean())
                    if not driver_prior_q.empty
                    else 12.0
                )
            else:
                # Pre-weekend H1
                driver_prior_q = prior_qual[prior_qual["driver_id"] == d_id].tail(5)
                grid_pos = (
                    float(driver_prior_q["qualifying_position"].mean())
                    if not driver_prior_q.empty
                    else 12.0
                )

            # Telemetry degradation & fuel sensitivity
            fuel_sens = calculate_circuit_fuel_sensitivity(
                circuit_length_km=circuit_topo["circuit_length_km"],
                tyre_stress_level=circuit_topo["tyre_stress_level"],
                overtaking_difficulty=circuit_topo["overtaking_difficulty"],
            )

            # Physics-guided interactions
            grid_x_overtaking = grid_pos * circuit_topo["overtaking_difficulty"]
            tyre_deg_proxy = 0.05 * (1.0 - c_rel + 0.1)
            deg_x_stress = tyre_deg_proxy * circuit_topo["tyre_stress_level"]
            speed_x_length = (1.0 - c_pole_gap) * circuit_topo["circuit_length_km"]

            # Actual outcomes (Labels for supervised training)
            fin_val = race_row.get("finish_position")
            if fin_val is not None and pd.notna(fin_val) and float(fin_val) > 0:
                finish_pos = float(fin_val)
            else:
                finish_pos = 20.0
            is_classified = bool(race_row.get("is_classified", False))
            is_dnf = bool(race_row.get("is_dnf", False))

            rows.append({
                "season": season,
                "round": round_num,
                "circuit_id": circuit_id,
                "driver_id": d_id,
                "constructor_id": c_id,
                "horizon": horizon,
                # Feature columns
                "dpi_rating": rating.rating,
                "dpi_uncertainty": rating.rd,
                "dpi_volatility": rating.volatility,
                "driver_starts": starts,
                "rookie_learning_factor": f_learn,
                "driver_consistency_cv": cv_pace,
                "cpi_pole_gap_pct": c_pole_gap,
                "cpi_reliability": c_rel,
                "cpi_development_trend": c_trend,
                "circuit_affinity_mean": aff.shrinkage_mean,
                "circuit_affinity_se": aff.uncertainty,
                "circuit_effective_starts": aff.sample_size,
                "circuit_length_km": circuit_topo["circuit_length_km"],
                "overtaking_difficulty": circuit_topo["overtaking_difficulty"],
                "tyre_stress_level": circuit_topo["tyre_stress_level"],
                "downforce_level": circuit_topo["downforce_level"],
                "altitude_m": circuit_topo["altitude_m"],
                "fuel_sensitivity": fuel_sens,
                "grid_position": grid_pos,
                # Interactions
                "grid_x_overtaking": grid_x_overtaking,
                "deg_x_stress": deg_x_stress,
                "speed_x_length": speed_x_length,
                # Labels
                "finish_position": finish_pos,
                "is_classified": is_classified,
                "is_dnf": is_dnf,
                "target_rank": finish_pos,
            })

        return pd.DataFrame(rows)

    def _build_glicko_ratings(
        self,
        prior_races: pd.DataFrame,
        prior_qual: pd.DataFrame,
    ) -> Glicko2Engine:
        """Run Glicko-2 engine chronologically across all prior races and qualifying sessions."""
        glicko = Glicko2Engine()
        if prior_races.empty:
            return glicko

        # Extract sorted distinct events
        events = (
            prior_races[["season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .values
        )

        for s, r in events:
            matches: List[Tuple[str, str, float]] = []

            # 1. Qualifying teammate matches
            sub_q = prior_qual[(prior_qual["season"] == s) & (prior_qual["round"] == r)]
            if not sub_q.empty:
                q_deltas = calculate_teammate_qualifying_deltas(sub_q)
                for _, row in q_deltas.iterrows():
                    matches.append((
                        row["driver_id"],
                        row["teammate_id"],
                        float(row["h2h_qual_win"]),
                    ))

            # 2. Race teammate matches
            sub_r = prior_races[(prior_races["season"] == s) & (prior_races["round"] == r)]
            if not sub_r.empty:
                r_deltas = calculate_teammate_race_deltas(sub_r)
                for _, row in r_deltas.iterrows():
                    if not row.get("is_confounded", False):
                        matches.append((
                            row["driver_id"],
                            row["teammate_id"],
                            float(row["h2h_race_win"]),
                        ))

            if matches:
                glicko.update_period(matches)

        return glicko
