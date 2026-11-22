"""PACE: Jolpica Cross-Source Reconciliation Engine.

Compares and reconciles Jolpica API records against PACE's existing Kaggle historical
archive and FastF1 modern datasets to ensure data integrity, detect discrepancies,
and enforce explicit dual-provenance source ownership without silent overwrites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

from src.data.storage import (
    DEFAULT_BASE_DIR,
    get_historical_path,
    get_jolpica_path,
    load_parquet,
    dataset_exists,
)

logger = logging.getLogger(__name__)


@dataclass
class JolpicaConflict:
    """Detailed record of a discrepancy between Jolpica and an existing reference source."""
    dataset_category: str
    season: int
    round: int
    entity_id: str
    field_name: str
    jolpica_value: Any
    reference_value: Any
    reference_source: str
    authoritative_source: str
    resolution_action: str


@dataclass
class JolpicaReconciliationReport:
    """Structured report documenting cross-source reconciliation metrics and differences."""
    dataset_category: str
    seasons_evaluated: List[int]
    total_jolpica_records: int = 0
    total_reference_records: int = 0
    overlap_count: int = 0
    identical_count: int = 0
    conflicts_detected: List[JolpicaConflict] = field(default_factory=list)
    jolpica_only_count: int = 0
    reference_only_count: int = 0
    authoritative_rules: Dict[str, str] = field(default_factory=dict)
    summary_message: str = ""

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts_detected)

    @property
    def match_rate_pct(self) -> float:
        if self.overlap_count == 0:
            return 100.0
        return float((self.identical_count / self.overlap_count) * 100.0)

    def summary(self) -> str:
        return (
            f"[{self.dataset_category}] Overlap={self.overlap_count}, Identical={self.identical_count} "
            f"({self.match_rate_pct:.2f}%), Conflicts={self.conflict_count}, "
            f"JolpicaOnly={self.jolpica_only_count}, RefOnly={self.reference_only_count}"
        )


class JolpicaReconciler:
    """Executes field-level cross-source reconciliation between Jolpica and existing PACE datasets."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or DEFAULT_BASE_DIR

    def reconcile_driver_standings(
        self,
        df_jolpica: pd.DataFrame,
        df_reference: pd.DataFrame,
        target_seasons: Optional[List[int]] = None,
        reference_source_name: str = "kaggle_archive",
    ) -> JolpicaReconciliationReport:
        """Reconcile driver standings between Jolpica and reference historical standings."""
        authoritative_rules = {
            "standings_official_positions": "Jolpica takes precedence as official FIA Ergast successor",
            "historical_pre_2023": "Kaggle archive retained for offline resilience",
        }

        if df_jolpica.empty and df_reference.empty:
            return JolpicaReconciliationReport(
                dataset_category="driver_standings",
                seasons_evaluated=[],
                authoritative_rules=authoritative_rules,
                summary_message="Both Jolpica and reference driver standings are empty.",
            )

        j_df = df_jolpica.copy()
        r_df = df_reference.copy()

        if target_seasons:
            if not j_df.empty and "season" in j_df.columns:
                j_df = j_df[j_df["season"].isin(target_seasons)]
            if not r_df.empty and "season" in r_df.columns:
                r_df = r_df[r_df["season"].isin(target_seasons)]

        seasons_eval = sorted(list(set(j_df["season"].dropna()).union(set(r_df["season"].dropna()))))

        j_map = {
            (int(r["season"]), int(r["round"]), str(r["driver_id"])): r
            for _, r in j_df.iterrows()
        }
        r_map = {
            (int(r["season"]), int(r["round"]), str(r["driver_id"])): r
            for _, r in r_df.iterrows()
        }

        j_keys = set(j_map.keys())
        r_keys = set(r_map.keys())

        overlap_keys = j_keys.intersection(r_keys)
        j_only_keys = j_keys - r_keys
        r_only_keys = r_keys - j_keys

        identical_count = 0
        conflicts: List[JolpicaConflict] = []

        fields_to_check = ["points", "position", "wins"]

        for key in sorted(list(overlap_keys)):
            j_row = j_map[key]
            r_row = r_map[key]

            is_match = True
            for f in fields_to_check:
                j_val = j_row.get(f)
                r_val = r_row.get(f)

                if pd.notna(j_val) and pd.notna(r_val):
                    # Numeric tolerance for points float
                    if isinstance(j_val, (int, float)) and isinstance(r_val, (int, float)):
                        diff = abs(float(j_val) - float(r_val))
                        if diff > 0.05:
                            is_match = False
                            conflicts.append(JolpicaConflict(
                                dataset_category="driver_standings",
                                season=key[0],
                                round=key[1],
                                entity_id=key[2],
                                field_name=f,
                                jolpica_value=float(j_val),
                                reference_value=float(r_val),
                                reference_source=reference_source_name,
                                authoritative_source="jolpica.ergast",
                                resolution_action="Jolpica official standing accepted",
                            ))
                    else:
                        if str(j_val).strip() != str(r_val).strip():
                            is_match = False
                            conflicts.append(JolpicaConflict(
                                dataset_category="driver_standings",
                                season=key[0],
                                round=key[1],
                                entity_id=key[2],
                                field_name=f,
                                jolpica_value=j_val,
                                reference_value=r_val,
                                reference_source=reference_source_name,
                                authoritative_source="jolpica.ergast",
                                resolution_action="Jolpica official standing accepted",
                            ))

            if is_match:
                identical_count += 1

        msg = (
            f"Driver standings reconciliation across seasons {seasons_eval}: {len(overlap_keys)} overlapping records, "
            f"{identical_count} identical ({((identical_count/len(overlap_keys))*100 if overlap_keys else 100):.2f}%), "
            f"{len(conflicts)} discrepancies detected."
        )

        return JolpicaReconciliationReport(
            dataset_category="driver_standings",
            seasons_evaluated=[int(s) for s in seasons_eval],
            total_jolpica_records=len(j_df),
            total_reference_records=len(r_df),
            overlap_count=len(overlap_keys),
            identical_count=identical_count,
            conflicts_detected=conflicts,
            jolpica_only_count=len(j_only_keys),
            reference_only_count=len(r_only_keys),
            authoritative_rules=authoritative_rules,
            summary_message=msg,
        )

    def reconcile_race_results(
        self,
        df_jolpica: pd.DataFrame,
        df_reference: pd.DataFrame,
        target_seasons: Optional[List[int]] = None,
        reference_source_name: str = "fastf1_or_kaggle",
    ) -> JolpicaReconciliationReport:
        """Reconcile Grand Prix finishing classifications and points."""
        authoritative_rules = {
            "modern_timing_telemetry": "FastF1 is primary and authoritative for lap telemetry and sector times (2023+)",
            "official_points_classification": "Jolpica is verified official FIA classification baseline",
        }

        if df_jolpica.empty and df_reference.empty:
            return JolpicaReconciliationReport(
                dataset_category="race_results",
                seasons_evaluated=[],
                authoritative_rules=authoritative_rules,
                summary_message="Both Jolpica and reference race results are empty.",
            )

        j_df = df_jolpica.copy()
        r_df = df_reference.copy()

        if target_seasons:
            if not j_df.empty and "season" in j_df.columns:
                j_df = j_df[j_df["season"].isin(target_seasons)]
            if not r_df.empty and "season" in r_df.columns:
                r_df = r_df[r_df["season"].isin(target_seasons)]

        seasons_eval = sorted(list(set(j_df["season"].dropna()).union(set(r_df["season"].dropna()))))

        j_map = {
            (int(r["season"]), int(r["round"]), str(r["driver_id"])): r
            for _, r in j_df.iterrows()
        }
        r_map = {
            (int(r["season"]), int(r["round"]), str(r["driver_id"])): r
            for _, r in r_df.iterrows()
        }

        overlap_keys = set(j_map.keys()).intersection(set(r_map.keys()))
        identical_count = 0
        conflicts: List[JolpicaConflict] = []

        fields_to_check = ["grid_position", "finish_position", "points", "is_classified", "is_dnf"]

        for key in sorted(list(overlap_keys)):
            j_row = j_map[key]
            r_row = r_map[key]

            is_match = True
            for f in fields_to_check:
                j_val = j_row.get(f)
                r_val = r_row.get(f)

                if pd.notna(j_val) and pd.notna(r_val):
                    if isinstance(j_val, (int, float)) and isinstance(r_val, (int, float)):
                        if abs(float(j_val) - float(r_val)) > 0.05:
                            is_match = False
                            conflicts.append(JolpicaConflict(
                                dataset_category="race_results",
                                season=key[0],
                                round=key[1],
                                entity_id=key[2],
                                field_name=f,
                                jolpica_value=float(j_val),
                                reference_value=float(r_val),
                                reference_source=reference_source_name,
                                authoritative_source="fastf1" if key[0] >= 2023 else "kaggle_archive",
                                resolution_action="Precedence rule maintained",
                            ))
                    else:
                        if str(j_val).strip() != str(r_val).strip():
                            is_match = False
                            conflicts.append(JolpicaConflict(
                                dataset_category="race_results",
                                season=key[0],
                                round=key[1],
                                entity_id=key[2],
                                field_name=f,
                                jolpica_value=j_val,
                                reference_value=r_val,
                                reference_source=reference_source_name,
                                authoritative_source="fastf1" if key[0] >= 2023 else "kaggle_archive",
                                resolution_action="Precedence rule maintained",
                            ))

            if is_match:
                identical_count += 1

        msg = (
            f"Race results reconciliation across seasons {seasons_eval}: {len(overlap_keys)} overlapping records, "
            f"{identical_count} identical ({((identical_count/len(overlap_keys))*100 if overlap_keys else 100):.2f}%), "
            f"{len(conflicts)} discrepancies detected."
        )

        return JolpicaReconciliationReport(
            dataset_category="race_results",
            seasons_evaluated=[int(s) for s in seasons_eval],
            total_jolpica_records=len(j_df),
            total_reference_records=len(r_df),
            overlap_count=len(overlap_keys),
            identical_count=identical_count,
            conflicts_detected=conflicts,
            jolpica_only_count=len(set(j_map.keys()) - overlap_keys),
            reference_only_count=len(set(r_map.keys()) - overlap_keys),
            authoritative_rules=authoritative_rules,
            summary_message=msg,
        )
