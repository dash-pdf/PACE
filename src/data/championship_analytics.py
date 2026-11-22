"""PACE: Championship Analytics & Temporal Standings Engine.

Provides leak-free, temporally strict as-of championship progression queries, points
trajectories, mathematical elimination thresholds, and prediction integration contracts
for Formula 1 World Drivers' and World Constructors' Championships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
import pandas as pd

from src.data.schemas import (
    DRIVER_STANDINGS_SCHEMA,
    CONSTRUCTOR_STANDINGS_SCHEMA,
    RACE_RESULTS_SCHEMA,
)
from src.data.storage import (
    DEFAULT_BASE_DIR,
    get_historical_path,
    load_parquet,
    dataset_exists,
)

logger = logging.getLogger(__name__)


@dataclass
class ChampionshipProgressionEntry:
    """Single round progression snapshot for a driver or constructor."""
    round: int
    points_scored: float
    cumulative_points: float
    position: int
    gap_to_leader: float
    gap_to_ahead: float
    positions_gained_lost: Optional[int] = None


@dataclass
class MathematicalEliminationReport:
    """Detailed evaluation of mathematical championship contention."""
    season: int
    as_of_round: int
    total_rounds: int
    remaining_rounds: int
    remaining_sprints: int
    max_remaining_points: float
    leader_id: str
    leader_points: float
    active_contenders_count: int
    eliminated_count: int
    active_contenders: List[str]
    eliminated_drivers: List[str]
    eliminated_this_round: List[str]
    standings_summary: pd.DataFrame


# ---------------------------------------------------------------------------
# Scoring Rules Engine
# ---------------------------------------------------------------------------

class F1ScoringRules:
    """Historical and modern Formula 1 scoring systems."""

    @staticmethod
    def get_max_points_per_event(season: int, has_sprint: bool = False) -> float:
        """Calculate maximum points a single driver can score in a Grand Prix weekend."""
        # 1. Race win points
        if season >= 2010:
            win_pts = 25.0
        elif season >= 2003:
            win_pts = 10.0
        elif season >= 1991:
            win_pts = 10.0
        elif season >= 1961:
            win_pts = 9.0
        else:
            win_pts = 8.0

        # 2. Fastest lap points
        # Modern bonus point was awarded 2019-2024. FIA abolished it for 2025-2026.
        # Historically awarded 1950-1959 (1 pt).
        if 2019 <= season <= 2024 or 1950 <= season <= 1959:
            fl_pts = 1.0
        else:
            fl_pts = 0.0

        # 3. Sprint race points (Introduced 2021)
        sprint_pts = 0.0
        if has_sprint:
            if season >= 2022:
                sprint_pts = 8.0  # 1st: 8 pts
            elif season == 2021:
                sprint_pts = 3.0  # 1st: 3 pts

        return win_pts + fl_pts + sprint_pts

    @staticmethod
    def get_max_constructor_points_per_event(season: int, has_sprint: bool = False) -> float:
        """Calculate maximum points a single constructor can score (1-2 finish)."""
        if season >= 2010:
            race_pts = 25.0 + 18.0  # 1st and 2nd
        elif season >= 2003:
            race_pts = 10.0 + 8.0
        elif season >= 1991:
            race_pts = 10.0 + 6.0
        else:
            race_pts = 9.0 + 6.0

        fl_pts = 1.0 if (2019 <= season <= 2024 or 1950 <= season <= 1959) else 0.0
        sprint_pts = 0.0
        if has_sprint:
            if season >= 2022:
                sprint_pts = 8.0 + 7.0  # 1st: 8, 2nd: 7
            elif season == 2021:
                sprint_pts = 3.0 + 2.0

        return race_pts + fl_pts + sprint_pts


# ---------------------------------------------------------------------------
# Championship Analytics Engine
# ---------------------------------------------------------------------------

class ChampionshipAnalytics:
    """Computes leak-free championship standings, points progressions, and elimination thresholds."""

    def __init__(
        self,
        df_driver_standings: Optional[pd.DataFrame] = None,
        df_constructor_standings: Optional[pd.DataFrame] = None,
        df_race_results: Optional[pd.DataFrame] = None,
        df_sprints: Optional[pd.DataFrame] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir or DEFAULT_BASE_DIR

        # Load standings tables
        if df_driver_standings is not None:
            self.df_d_stand = df_driver_standings
        else:
            p = get_historical_path("driver_standings", base_dir=self.base_dir)
            self.df_d_stand = load_parquet(p) if dataset_exists(p) else pd.DataFrame(columns=DRIVER_STANDINGS_SCHEMA.names)

        if df_constructor_standings is not None:
            self.df_c_stand = df_constructor_standings
        else:
            p = get_historical_path("constructor_standings", base_dir=self.base_dir)
            self.df_c_stand = load_parquet(p) if dataset_exists(p) else pd.DataFrame(columns=CONSTRUCTOR_STANDINGS_SCHEMA.names)

        if df_race_results is not None:
            self.df_results = df_race_results
        else:
            p = get_historical_path("race_results", base_dir=self.base_dir)
            self.df_results = load_parquet(p) if dataset_exists(p) else pd.DataFrame(columns=RACE_RESULTS_SCHEMA.names)

        if df_sprints is not None:
            self.df_sprints = df_sprints
        else:
            p = get_historical_path("sprints", base_dir=self.base_dir)
            self.df_sprints = load_parquet(p) if dataset_exists(p) else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Temporal As-Of Standings (Zero Future Leakage)
    # -----------------------------------------------------------------------

    def get_driver_standings_as_of(
        self,
        season: int,
        as_of_round: int,
        before_event: bool = False,
    ) -> pd.DataFrame:
        """Return driver championship standings strictly as of target round without future leakage.
        
        Args:
            season: Championship year.
            as_of_round: Target Grand Prix round.
            before_event: If True, evaluates standings immediately prior to target event (as_of_round - 1).
        """
        target_round = (as_of_round - 1) if before_event else as_of_round
        if target_round < 1:
            # Pre-season round 0: all drivers have 0 points
            return pd.DataFrame(columns=[
                "season", "round", "driver_id", "points", "position",
                "position_text", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"
            ])

        # Filter strictly by target round
        subset = self.df_d_stand[
            (self.df_d_stand["season"] == season) & (self.df_d_stand["round"] == target_round)
        ].copy()

        # Fallback: if standings table lacks this exact round, compute dynamically from race results + sprints
        if subset.empty and not self.df_results.empty:
            rr_sub = self.df_results[
                (self.df_results["season"] == season) & (self.df_results["round"] <= target_round)
            ]
            sp_sub = self.df_sprints[
                (self.df_sprints["season"] == season) & (self.df_sprints["round"] <= target_round)
            ] if not self.df_sprints.empty and "season" in self.df_sprints.columns else pd.DataFrame()

            if not rr_sub.empty:
                rr_pts = rr_sub.groupby("driver_id")["points"].sum().to_dict()
                rr_wins = (rr_sub["finish_position"] == 1).groupby(rr_sub["driver_id"]).sum().to_dict()

                sp_pts = {}
                if not sp_sub.empty:
                    sp_pts = sp_sub.groupby("driver_id")["points"].sum().to_dict()

                all_drivers = set(rr_pts.keys()).union(set(sp_pts.keys()))
                rows = []
                for d_id in all_drivers:
                    total_pts = float(rr_pts.get(d_id, 0.0) + sp_pts.get(d_id, 0.0))
                    w_count = int(rr_wins.get(d_id, 0))
                    rows.append({
                        "season": season,
                        "round": target_round,
                        "driver_id": d_id,
                        "points": total_pts,
                        "wins": w_count,
                        "position_text": "",
                        "source_api": "pace.dynamic_as_of",
                        "retrieved_at": "",
                    })
                subset = pd.DataFrame(rows)
                # Sort by points desc, then wins desc
                subset = subset.sort_values(by=["points", "wins"], ascending=[False, False]).reset_index(drop=True)
                subset["position"] = range(1, len(subset) + 1)
                subset["position_text"] = subset["position"].astype(str)

        if subset.empty:
            return pd.DataFrame(columns=[
                "season", "round", "driver_id", "points", "position",
                "position_text", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"
            ])

        subset = subset.sort_values(by="position", ascending=True).reset_index(drop=True)

        # Compute gaps
        leader_pts = float(subset.iloc[0]["points"])
        subset["gap_to_leader"] = leader_pts - subset["points"]

        # Gap to ahead
        pts_list = subset["points"].tolist()
        gaps_ahead = [0.0]
        for i in range(1, len(pts_list)):
            gaps_ahead.append(max(0.0, float(pts_list[i - 1] - pts_list[i])))
        subset["gap_to_ahead"] = gaps_ahead

        # Positions gained / lost since prior round
        subset["positions_gained_lost"] = None
        if target_round > 1:
            prev_round_stand = self.df_d_stand[
                (self.df_d_stand["season"] == season) & (self.df_d_stand["round"] == target_round - 1)
            ]
            if not prev_round_stand.empty:
                prev_pos_map = dict(zip(prev_round_stand["driver_id"], prev_round_stand["position"]))
                deltas = []
                for _, r in subset.iterrows():
                    d_id = r["driver_id"]
                    cur_p = int(r["position"])
                    prev_p = prev_pos_map.get(d_id)
                    if prev_p is not None:
                        # Positive delta means gained positions (e.g. from 5 to 3 is +2)
                        deltas.append(int(prev_p - cur_p))
                    else:
                        deltas.append(None)
                subset["positions_gained_lost"] = deltas

        return subset

    def get_constructor_standings_as_of(
        self,
        season: int,
        as_of_round: int,
        before_event: bool = False,
    ) -> pd.DataFrame:
        """Return constructor championship standings strictly as of target round."""
        target_round = (as_of_round - 1) if before_event else as_of_round
        if target_round < 1:
            return pd.DataFrame(columns=[
                "season", "round", "constructor_id", "points", "position",
                "position_text", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"
            ])

        subset = self.df_c_stand[
            (self.df_c_stand["season"] == season) & (self.df_c_stand["round"] == target_round)
        ].copy()

        # Dynamic fallback if needed
        if subset.empty and not self.df_results.empty:
            rr_sub = self.df_results[
                (self.df_results["season"] == season) & (self.df_results["round"] <= target_round)
            ]
            sp_sub = self.df_sprints[
                (self.df_sprints["season"] == season) & (self.df_sprints["round"] <= target_round)
            ] if not self.df_sprints.empty and "season" in self.df_sprints.columns else pd.DataFrame()

            if not rr_sub.empty:
                c_pts = rr_sub.groupby("constructor_id")["points"].sum().to_dict()
                c_wins = (rr_sub["finish_position"] == 1).groupby(rr_sub["constructor_id"]).sum().to_dict()

                if not sp_sub.empty and "constructor_id" in sp_sub.columns:
                    sp_c_pts = sp_sub.groupby("constructor_id")["points"].sum().to_dict()
                    for c_id, sp_p in sp_c_pts.items():
                        c_pts[c_id] = c_pts.get(c_id, 0.0) + sp_p

                rows = []
                for c_id, total_pts in c_pts.items():
                    rows.append({
                        "season": season,
                        "round": target_round,
                        "constructor_id": c_id,
                        "points": float(total_pts),
                        "wins": int(c_wins.get(c_id, 0)),
                        "position_text": "",
                        "source_api": "pace.dynamic_as_of",
                        "retrieved_at": "",
                    })
                subset = pd.DataFrame(rows)
                subset = subset.sort_values(by=["points", "wins"], ascending=[False, False]).reset_index(drop=True)
                subset["position"] = range(1, len(subset) + 1)
                subset["position_text"] = subset["position"].astype(str)

        if subset.empty:
            return pd.DataFrame(columns=[
                "season", "round", "constructor_id", "points", "position",
                "position_text", "wins", "gap_to_leader", "gap_to_ahead", "positions_gained_lost"
            ])

        subset = subset.sort_values(by="position", ascending=True).reset_index(drop=True)
        leader_pts = float(subset.iloc[0]["points"])
        subset["gap_to_leader"] = leader_pts - subset["points"]

        pts_list = subset["points"].tolist()
        gaps_ahead = [0.0]
        for i in range(1, len(pts_list)):
            gaps_ahead.append(max(0.0, float(pts_list[i - 1] - pts_list[i])))
        subset["gap_to_ahead"] = gaps_ahead

        # Positions gained/lost
        subset["positions_gained_lost"] = None
        if target_round > 1:
            prev_c = self.df_c_stand[
                (self.df_c_stand["season"] == season) & (self.df_c_stand["round"] == target_round - 1)
            ]
            if not prev_c.empty:
                prev_map = dict(zip(prev_c["constructor_id"], prev_c["position"]))
                deltas = []
                for _, r in subset.iterrows():
                    cid = r["constructor_id"]
                    cur_p = int(r["position"])
                    prev_p = prev_map.get(cid)
                    deltas.append(int(prev_p - cur_p) if prev_p is not None else None)
                subset["positions_gained_lost"] = deltas

        return subset

    # -----------------------------------------------------------------------
    # Points Progression Trajectories
    # -----------------------------------------------------------------------

    def get_points_progression(
        self,
        season: int,
        driver_id: Optional[str] = None,
        constructor_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return chronological progression of standings and points across rounds."""
        if driver_id:
            sub = self.df_d_stand[(self.df_d_stand["season"] == season) & (self.df_d_stand["driver_id"] == driver_id)].copy()
            if sub.empty:
                return pd.DataFrame()
            sub = sub.sort_values(by="round").reset_index(drop=True)
            # Compute points scored in each round
            pts = sub["points"].tolist()
            round_pts = [pts[0]] + [pts[i] - pts[i - 1] for i in range(1, len(pts))]
            sub["points_scored_this_round"] = round_pts
            return sub[["season", "round", "driver_id", "points", "points_scored_this_round", "position", "wins"]]

        if constructor_id:
            sub = self.df_c_stand[(self.df_c_stand["season"] == season) & (self.df_c_stand["constructor_id"] == constructor_id)].copy()
            if sub.empty:
                return pd.DataFrame()
            sub = sub.sort_values(by="round").reset_index(drop=True)
            pts = sub["points"].tolist()
            round_pts = [pts[0]] + [pts[i] - pts[i - 1] for i in range(1, len(pts))]
            sub["points_scored_this_round"] = round_pts
            return sub[["season", "round", "constructor_id", "points", "points_scored_this_round", "position", "wins"]]

        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Mathematical Championship Elimination Engine
    # -----------------------------------------------------------------------

    def compute_elimination_status(
        self,
        season: int,
        as_of_round: int,
        total_rounds: Optional[int] = None,
        remaining_sprints: int = 0,
    ) -> MathematicalEliminationReport:
        """Calculate mathematical eligibility and elimination for World Drivers' Championship."""
        # 1. Determine total season rounds
        if total_rounds is None:
            # Query maximum round in current season dataset
            season_rounds = self.df_d_stand[self.df_d_stand["season"] == season]["round"].max()
            if pd.notna(season_rounds) and season_rounds > 0:
                total_rounds = int(max(season_rounds, 24 if season >= 2024 else 22))
            else:
                total_rounds = 24 if season >= 2024 else 22

        remaining_rounds = max(0, total_rounds - as_of_round)

        # 2. Maximum available points remaining
        # Formula: N_rem * (25 + FL_pts) + N_sprints_rem * 8
        fl_pts = 1.0 if (2019 <= season <= 2024 or 1950 <= season <= 1959) else 0.0
        max_pts_per_race = 25.0 + fl_pts
        max_remaining_points = (remaining_rounds * max_pts_per_race) + (remaining_sprints * 8.0)

        # 3. Current Standings as of target round
        cur_standings = self.get_driver_standings_as_of(season, as_of_round)
        if cur_standings.empty:
            raise ValueError(f"No driver standings available for season {season} as of round {as_of_round}")

        leader_row = cur_standings.iloc[0]
        leader_id = str(leader_row["driver_id"])
        leader_pts = float(leader_row["points"])

        # 4. Prior round standings for elimination delta tracking
        prev_eliminated_set: Set[str] = set()
        if as_of_round > 1:
            try:
                prev_stand = self.get_driver_standings_as_of(season, as_of_round - 1)
                if not prev_stand.empty:
                    prev_leader_pts = float(prev_stand.iloc[0]["points"])
                    prev_rem_r = remaining_rounds + 1
                    # Approximate prev remaining points
                    prev_rem_pts = (prev_rem_r * max_pts_per_race) + (remaining_sprints * 8.0)
                    for _, pr in prev_stand.iterrows():
                        if float(pr["points"]) + prev_rem_pts < prev_leader_pts:
                            prev_eliminated_set.add(str(pr["driver_id"]))
            except Exception:
                pass

        # 5. Evaluate contention for each driver
        active_contenders: List[str] = []
        eliminated_drivers: List[str] = []
        eliminated_this_round: List[str] = []

        summary_rows: List[Dict[str, Any]] = []

        for _, row in cur_standings.iterrows():
            d_id = str(row["driver_id"])
            pts = float(row["points"])
            max_possible = pts + max_remaining_points
            gap = leader_pts - pts

            is_elim = max_possible < leader_pts

            if is_elim:
                eliminated_drivers.append(d_id)
                if d_id not in prev_eliminated_set:
                    eliminated_this_round.append(d_id)
            else:
                active_contenders.append(d_id)

            summary_rows.append({
                "position": int(row["position"]),
                "driver_id": d_id,
                "current_points": pts,
                "gap_to_leader": gap,
                "max_possible_points": max_possible,
                "is_eliminated": is_elim,
                "status": "ELIMINATED" if is_elim else "CONTENDER",
            })

        summary_df = pd.DataFrame(summary_rows)

        return MathematicalEliminationReport(
            season=season,
            as_of_round=as_of_round,
            total_rounds=total_rounds,
            remaining_rounds=remaining_rounds,
            remaining_sprints=remaining_sprints,
            max_remaining_points=max_remaining_points,
            leader_id=leader_id,
            leader_points=leader_pts,
            active_contenders_count=len(active_contenders),
            eliminated_count=len(eliminated_drivers),
            active_contenders=active_contenders,
            eliminated_drivers=eliminated_drivers,
            eliminated_this_round=eliminated_this_round,
            standings_summary=summary_df,
        )

    # -----------------------------------------------------------------------
    # Predictive Simulation Title Probability Evaluation Contract
    # -----------------------------------------------------------------------

    def evaluate_championship_outcomes(
        self,
        season: int,
        as_of_round: int,
        simulated_remaining_points: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Data contract connecting Monte Carlo simulated points to championship win probabilities.
        
        Args:
            season: Target championship season.
            as_of_round: Snapshot round before simulated remaining races.
            simulated_remaining_points: Dict mapping driver_id -> List of simulated points scored
                                        across remaining rounds (length M = number of MC simulations).
                                        
        Returns:
            Dict mapping driver_id -> empirical championship title win probability (0.0 to 1.0).
        """
        cur_standings = self.get_driver_standings_as_of(season, as_of_round)
        if cur_standings.empty:
            return {}

        cur_pts_map = dict(zip(cur_standings["driver_id"], cur_standings["points"]))
        drivers = list(simulated_remaining_points.keys())
        if not drivers:
            return {}

        n_sims = len(next(iter(simulated_remaining_points.values())))
        if n_sims == 0:
            return {}

        # Tally championships won across iterations
        championship_wins: Dict[str, int] = {d: 0 for d in drivers}

        for m in range(n_sims):
            final_totals = {}
            for d in drivers:
                base_pts = cur_pts_map.get(d, 0.0)
                sim_pts = simulated_remaining_points[d][m] if m < len(simulated_remaining_points[d]) else 0.0
                final_totals[d] = base_pts + sim_pts

            # Champion is driver with highest final score
            champion = max(final_totals, key=lambda k: final_totals[k])
            championship_wins[champion] += 1

        return {d: float(wins / n_sims) for d, wins in championship_wins.items()}
