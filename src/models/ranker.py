"""PACE: Learning-to-Rank GBDT & Permutation Engine (Tier 4).

Implements:
1. XGBoostPairwiseRanker: Pairwise margin ranking (rank:pairwise).
2. XGBoostLambdaMARTRanker: Listwise ranking optimizing NDCG (rank:ndcg).
3. RankPermutationSolver: Guarantees deterministic P1..PN finishing order.
4. Monte Carlo Posterior Sampler: Generates full finish probability distributions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb


class RankPermutationSolver:
    """Enforces that predicted continuous utility scores are converted into
    a strictly valid, tie-free permutation ranking [1, ..., N].
    """

    @staticmethod
    def solve(utilities: np.ndarray, ascending: bool = False) -> np.ndarray:
        """Project continuous utilities into 1-indexed ranks (1 to N).
        
        Args:
            utilities: Array of driver utility scores.
            ascending: If True, lower score is better (P1). If False, higher utility is better (P1).
            
        Returns:
            ranks: 1-indexed integers from 1 to N.
        """
        n = len(utilities)
        if n == 0:
            return np.array([], dtype=int)

        # argsort gives indices of sorted elements
        if ascending:
            order = np.argsort(utilities)
        else:
            order = np.argsort(-utilities)

        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        return ranks.astype(int)


class XGBoostPairwiseRanker:
    """Tier 4: Supervised Pairwise Ranking model using XGBoost."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.name = "ML_XGBoost_Pairwise"
        self.feature_names: List[str] = []
        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_ranks: np.ndarray,
        groups: List[int],
    ) -> "XGBoostPairwiseRanker":
        """Fit pairwise ranker on historical races.
        
        Note: For rank:pairwise, higher relevance target corresponds to higher rank (better position).
        Relevance is transformed as: rel = 25 - finish_position.
        """
        self.feature_names = X.columns.tolist()
        # Convert finish position (1=best) to relevance (25=best)
        relevance = np.maximum(25 - y_ranks, 0).astype(int)

        self.model.fit(
            X,
            relevance,
            group=groups,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict integer finishing ranks 1 to N for a single race grid."""
        utilities = self.model.predict(X[self.feature_names])
        # Higher predicted utility corresponds to better rank (P1)
        return RankPermutationSolver.solve(utilities, ascending=False)


class XGBoostLambdaMARTRanker:
    """Tier 4: Supervised Listwise LambdaMART ranking model optimizing NDCG."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.name = "ML_XGBoost_LambdaMART"
        self.feature_names: List[str] = []
        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_ranks: np.ndarray,
        groups: List[int],
    ) -> "XGBoostLambdaMARTRanker":
        self.feature_names = X.columns.tolist()
        relevance = np.maximum(25 - y_ranks, 0).astype(int)

        self.model.fit(
            X,
            relevance,
            group=groups,
            verbose=False,
        )
        return self

    def predict_utilities(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous utility scores."""
        return self.model.predict(X[self.feature_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict integer finishing ranks 1 to N."""
        utilities = self.predict_utilities(X)
        return RankPermutationSolver.solve(utilities, ascending=False)

    def sample_posterior_permutations(
        self,
        X: pd.DataFrame,
        n_draws: int = 1000,
        uncertainty_scale: float = 0.15,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Sample M permutations from the predictive posterior distribution.
        
        Combines model utility predictions with feature-level epistemic uncertainty
        and Plackett-Luce Gumbel choice stochasticity to draw valid permutation realizations.
        
        Returns:
            permutations: Matrix of shape (n_draws, n_drivers) with 1-indexed ranks.
        """
        rng = np.random.RandomState(random_seed)
        base_utilities = self.predict_utilities(X)
        n_drivers = len(base_utilities)

        # Extract driver uncertainty scaling if available
        if "dpi_uncertainty" in X.columns:
            # Scale Glicko-2 RD (~50-350) into utility perturbation
            rds = X["dpi_uncertainty"].values
            se_scale = (rds / 350.0) * uncertainty_scale
        else:
            se_scale = np.full(n_drivers, uncertainty_scale)

        draws = np.empty((n_draws, n_drivers), dtype=int)
        for m in range(n_draws):
            epistemic_noise = rng.normal(0.0, se_scale, size=n_drivers)
            u_gumbel = rng.uniform(1e-10, 1.0 - 1e-10, size=n_drivers)
            gumbel_noise = -np.log(-np.log(u_gumbel))
            draw_utilities = base_utilities + epistemic_noise + gumbel_noise
            draws[m] = RankPermutationSolver.solve(draw_utilities, ascending=False)

        return draws


class ChronologicalBootstrapRanker:
    """Approach 3: Chronological Block Bootstrap Model Ensemble.
    
    Resamples past race events with replacement, trains an ensemble of K rankers,
    and quantifies model epistemic uncertainty across ensemble forecasts.
    """

    def __init__(
        self,
        n_bootstrap: int = 10,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.08,
    ):
        self.n_bootstrap = n_bootstrap
        self.models: List[XGBoostLambdaMARTRanker] = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(
        self,
        event_dfs: List[pd.DataFrame],
        feature_columns: List[str],
    ) -> "ChronologicalBootstrapRanker":
        """Fit ensemble across bootstrap resamples of past race events."""
        n_events = len(event_dfs)
        self.models = []

        if n_events == 0:
            return self

        for b in range(self.n_bootstrap):
            # Resample event indices with replacement
            resample_idx = np.random.choice(n_events, size=n_events, replace=True)
            resampled_dfs = [event_dfs[i] for i in resample_idx]

            concat_df = pd.concat(resampled_dfs, ignore_index=True)
            groups = [len(df) for df in resampled_dfs]

            X = concat_df[feature_columns].fillna(0.0)
            y = concat_df["target_rank"].values

            model = XGBoostLambdaMARTRanker(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42 + b,
            )
            model.fit(X, y, groups=groups)
            self.models.append(model)

        return self

    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict ensemble average ranks and epistemic uncertainty across models.
        
        Returns:
            (mean_ranks, rank_std): 1-indexed deterministic rank and ensemble std dev.
        """
        if not self.models:
            n = len(X)
            return np.arange(1, n + 1), np.zeros(n)

        all_ranks = []
        for model in self.models:
            ranks = model.predict(X)
            all_ranks.append(ranks)

        rank_matrix = np.array(all_ranks)  # shape (n_bootstrap, n_drivers)
        avg_rank = np.mean(rank_matrix, axis=0)
        rank_std = np.std(rank_matrix, axis=0)

        # Deterministic consensus rank
        final_ranks = RankPermutationSolver.solve(avg_rank, ascending=True)
        return final_ranks, rank_std
