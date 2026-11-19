"""PACE: Non-Parametric Baseline Models (Tier 1).

Implements interpretable, non-parametric heuristic benchmarks:
1. GridOrderBaseline: Finish = Starting Grid Position.
2. ChampionshipStandingsBaseline: Finishing order based on accumulated points entering event.
3. RollingConstructorBaseline: Finishing order based on constructor rolling average finish.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class GridOrderBaseline:
    """Predicts finishing position strictly based on confirmed starting grid position.
    
    Ties or unranked cars are sorted by original input order.
    """

    def __init__(self):
        self.name = "Baseline_GridOrder"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict finishing positions for each driver in df.
        
        Returns:
            predicted_positions: 1-indexed ranks (1 to N).
        """
        if "grid_position" not in df.columns:
            return np.arange(1, len(df) + 1, dtype=int)

        grid = df["grid_position"].fillna(20.0).values
        # argsort of argsort gives 0-indexed ranking
        order = np.argsort(grid)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(grid) + 1)
        return ranks.astype(int)


class ChampionshipStandingsBaseline:
    """Predicts finishing position based on cumulative points in the season entering the round."""

    def __init__(self):
        self.name = "Baseline_ChampionshipStandings"

    def predict_with_history(
        self,
        event_df: pd.DataFrame,
        prior_season_results: pd.DataFrame,
    ) -> np.ndarray:
        """Predict rank based on season points before this round."""
        if prior_season_results.empty:
            # Fall back to grid order or default
            return GridOrderBaseline().predict(event_df)

        points_map = prior_season_results.groupby("driver_id")["points"].sum().to_dict()
        driver_points = [
            points_map.get(d, 0.0) for d in event_df["driver_id"]
        ]

        # Higher points -> lower rank (P1 has most points)
        # Sort descending by points
        neg_points = -np.array(driver_points, dtype=float)
        order = np.argsort(neg_points)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(driver_points) + 1)
        return ranks.astype(int)


class RollingConstructorBaseline:
    """Predicts finishing position based on constructor 3-race rolling average finish."""

    def __init__(self, window: int = 3):
        self.name = "Baseline_RollingConstructor"
        self.window = window

    def predict_with_history(
        self,
        event_df: pd.DataFrame,
        prior_results: pd.DataFrame,
    ) -> np.ndarray:
        """Predict rank based on team's recent average finish position."""
        if prior_results.empty:
            return GridOrderBaseline().predict(event_df)

        # Calculate constructor rolling average finish over recent races
        recent_races = prior_results.sort_values(["season", "round"], ascending=False)
        team_avg_map = (
            recent_races.groupby("constructor_id")["finish_position"]
            .apply(lambda s: s.dropna().head(self.window * 2).mean())
            .to_dict()
        )

        team_averages = [
            team_avg_map.get(c, 12.0) for c in event_df["constructor_id"]
        ]

        # Lower average finish -> better rank
        order = np.argsort(team_averages)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(team_averages) + 1)
        return ranks.astype(int)
