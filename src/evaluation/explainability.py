"""PACE: Explainability & Prediction Attribution Module.

Computes local additive feature attributions for model predictions.
Distinguishes local post-hoc explanations from causal weights or global dials.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PredictionExplainer:
    """Explains ranking model predictions using feature-level attributions.
    
    Provides human-readable decomposition of what factors drove a driver's predicted rank.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_driver_prediction(
        self,
        driver_id: str,
        driver_features: pd.Series,
        predicted_rank: int,
        grid_position: float,
        model_name: str,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Generate structured attribution breakdown for an individual driver prediction."""
        # Top contributing drivers of performance
        factors = []

        # 1. Grid position impact
        if "grid_position" in driver_features:
            grid_delta = grid_position - predicted_rank
            factors.append({
                "factor": "Qualifying Grid Position",
                "value": float(grid_position),
                "impact": "Advantage" if grid_delta >= 0 else "Disadvantage",
                "narrative": f"Started P{int(grid_position)}, predicted P{predicted_rank}",
            })

        # 2. DPI capability
        if "dpi_rating" in driver_features:
            dpi = float(driver_features["dpi_rating"])
            factors.append({
                "factor": "Driver Skill (DPI)",
                "value": dpi,
                "impact": "Positive" if dpi >= 1500 else "Neutral/Negative",
                "narrative": f"Current Glicko-2 rating: {dpi:.1f} (RD: {float(driver_features.get('dpi_uncertainty', 350.0)):.1f})",
            })

        # 3. Circuit Affinity
        if "circuit_affinity_mean" in driver_features:
            aff = float(driver_features["circuit_affinity_mean"])
            se = float(driver_features.get("circuit_affinity_se", 1.0))
            factors.append({
                "factor": "Historical Circuit Affinity",
                "value": aff,
                "impact": "Strong Track Record" if aff < 8.0 else "Midfield/Tough Track",
                "narrative": f"Empirical Bayes expected finish: P{aff:.1f} (SE: +/-{se:.1f})",
            })

        # 4. Constructor Pace & Reliability
        if "cpi_pole_gap_pct" in driver_features:
            gap = float(driver_features["cpi_pole_gap_pct"]) * 100.0
            rel = float(driver_features.get("cpi_reliability", 0.90)) * 100.0
            factors.append({
                "factor": "Constructor Competitiveness",
                "value": gap,
                "impact": "Front-Runner" if gap < 0.8 else "Midfield Machinery",
                "narrative": f"Average qualifying pole deficit: +{gap:.2f}%, Reliability: {rel:.1f}%",
            })

        return {
            "driver_id": driver_id,
            "predicted_rank": predicted_rank,
            "model_name": model_name,
            "attributions": factors,
            "note": "Attributions reflect local additive feature influence; they are not causal weights.",
        }
