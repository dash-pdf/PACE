"""PACE: OpenF1 vs FastF1 Cross-Source Reconciliation Engine.

Audits, compares, and reconciles OpenF1 race intelligence datasets against FastF1 timing
and telemetry baselines for overlapping historical sessions. Establishes field-level
tolerances, reports match rates, detects discrepancies, and defines source precedence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class OpenF1Discrepancy:
    """Detailed record of a discrepancy between OpenF1 and FastF1."""
    category: str                           # 'laps', 'stints', 'positions', 'intervals'
    entity_id: str                          # driver_id or session key
    lap_number: Optional[int]
    field_name: str
    openf1_value: Any
    fastf1_value: Any
    absolute_delta: Optional[float]
    tolerance_applied: float
    authoritative_source: str
    resolution_action: str


@dataclass
class OpenF1ReconciliationReport:
    """Structured report documenting cross-source reconciliation metrics and findings."""
    session_key: int
    season: int
    round_num: Optional[int]
    category: str
    openf1_record_count: int
    fastf1_record_count: int
    overlap_count: int
    identical_count: int
    conflicts_detected: List[OpenF1Discrepancy] = field(default_factory=list)
    openf1_only_count: int = 0
    fastf1_only_count: int = 0
    tolerance_threshold: float = 0.010      # 10ms default tolerance for timing
    authoritative_precedence: str = "FastF1"
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
            f"[OpenF1 vs FastF1: {self.category}] Overlap={self.overlap_count}, "
            f"Identical={self.identical_count} ({self.match_rate_pct:.2f}%), "
            f"Conflicts={self.conflict_count}, Tol={self.tolerance_threshold*1000:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Reconciler Class
# ---------------------------------------------------------------------------

class OpenF1Reconciler:
    """Executes cross-source comparison and reconciliation between OpenF1 and FastF1."""

    # Official Source Precedence Matrix
    SOURCE_PRECEDENCE_MATRIX: Dict[str, str] = {
        "lap_duration_timing": "FastF1 is Authoritative (direct FOM timing feed transponder precision)",
        "sector_times": "FastF1 is Authoritative (micro-sector timing integration)",
        "speed_traps": "FastF1 is Authoritative (calibrated speed trap transponder lines)",
        "gaps_and_intervals": "OpenF1 is Authoritative (continuous high-frequency interval stream)",
        "race_control_messages": "OpenF1 is Authoritative (verbatim FIA race director message text & timestamps)",
        "overtaking_maneuvers": "OpenF1 is Corroborative (independent track overtake sensor detection)",
        "tyre_compounds_stints": "FastF1 is Authoritative; OpenF1 provides corroboration on lap bounds",
    }

    def __init__(self) -> None:
        pass

    def reconcile_laps(
        self,
        df_openf1_laps: pd.DataFrame,
        df_fastf1_laps: pd.DataFrame,
        session_key: int = 0,
        season: int = 2023,
        round_num: Optional[int] = None,
        timing_tolerance_ms: float = 15.0,  # 15 ms tolerance for rounding/transponder truncation
    ) -> OpenF1ReconciliationReport:
        """Compare lap times between OpenF1 and FastF1 on (driver_id, lap_number)."""
        if df_openf1_laps.empty and df_fastf1_laps.empty:
            return OpenF1ReconciliationReport(
                session_key=session_key,
                season=season,
                round_num=round_num,
                category="laps",
                openf1_record_count=0,
                fastf1_record_count=0,
                overlap_count=0,
                identical_count=0,
                summary_message="Both lap datasets are empty.",
            )

        # Normalize OpenF1 lap duration to milliseconds
        o_df = df_openf1_laps.copy()
        if "lap_duration" in o_df.columns:
            o_df["lap_time_ms"] = o_df["lap_duration"] * 1000.0
        elif "lap_time_ms" not in o_df.columns and "lap_time_str" in o_df.columns:
            # Parse if needed
            o_df["lap_time_ms"] = np.nan

        f_df = df_fastf1_laps.copy()

        # Build index maps: (driver_id, lap_number) -> row
        o_map: Dict[Tuple[str, int], Any] = {}
        for _, r in o_df.iterrows():
            d_id = str(r.get("driver_id") or "")
            lap = int(r.get("lap_number", 0))
            if d_id and lap > 0:
                o_map[(d_id, lap)] = r

        f_map: Dict[Tuple[str, int], Any] = {}
        for _, r in f_df.iterrows():
            d_id = str(r.get("driver_id") or "")
            lap = int(r.get("lap_number", 0))
            if d_id and lap > 0:
                f_map[(d_id, lap)] = r

        overlap_keys = set(o_map.keys()).intersection(set(f_map.keys()))
        identical_count = 0
        conflicts: List[OpenF1Discrepancy] = []

        for key in sorted(list(overlap_keys)):
            o_row = o_map[key]
            f_row = f_map[key]

            o_ms = o_row.get("lap_time_ms")
            f_ms = f_row.get("lap_time_ms")

            if pd.notna(o_ms) and pd.notna(f_ms):
                delta = abs(float(o_ms) - float(f_ms))
                if delta <= timing_tolerance_ms:
                    identical_count += 1
                else:
                    conflicts.append(OpenF1Discrepancy(
                        category="laps",
                        entity_id=key[0],
                        lap_number=key[1],
                        field_name="lap_time_ms",
                        openf1_value=float(o_ms),
                        fastf1_value=float(f_ms),
                        absolute_delta=round(delta, 3),
                        tolerance_applied=timing_tolerance_ms,
                        authoritative_source="FastF1",
                        resolution_action="FastF1 timing transponder baseline retained",
                    ))
            else:
                identical_count += 1

        msg = (
            f"Laps reconciliation across session {session_key}: {len(overlap_keys)} overlapping laps, "
            f"{identical_count} within {timing_tolerance_ms:.1f}ms tolerance "
            f"({((identical_count/len(overlap_keys))*100 if overlap_keys else 100):.2f}%), "
            f"{len(conflicts)} discrepancies detected."
        )

        return OpenF1ReconciliationReport(
            session_key=session_key,
            season=season,
            round_num=round_num,
            category="laps",
            openf1_record_count=len(o_df),
            fastf1_record_count=len(f_df),
            overlap_count=len(overlap_keys),
            identical_count=identical_count,
            conflicts_detected=conflicts,
            openf1_only_count=len(set(o_map.keys()) - overlap_keys),
            fastf1_only_count=len(set(f_map.keys()) - overlap_keys),
            tolerance_threshold=timing_tolerance_ms / 1000.0,
            authoritative_precedence="FastF1",
            summary_message=msg,
        )

    def reconcile_stints(
        self,
        df_openf1_stints: pd.DataFrame,
        df_fastf1_stints: pd.DataFrame,
        session_key: int = 0,
        season: int = 2023,
    ) -> OpenF1ReconciliationReport:
        """Compare tyre compounds and stint start/end lap bounds."""
        if df_openf1_stints.empty and df_fastf1_stints.empty:
            return OpenF1ReconciliationReport(
                session_key=session_key,
                season=season,
                round_num=None,
                category="stints",
                openf1_record_count=0,
                fastf1_record_count=0,
                overlap_count=0,
                identical_count=0,
                summary_message="Both stint datasets are empty.",
            )

        o_map = {
            (str(r["driver_id"]), int(r["stint_number"])): r
            for _, r in df_openf1_stints.iterrows()
            if pd.notna(r.get("driver_id")) and pd.notna(r.get("stint_number"))
        }
        f_map = {
            (str(r["driver_id"]), int(r["stint_number"])): r
            for _, r in df_fastf1_stints.iterrows()
            if pd.notna(r.get("driver_id")) and pd.notna(r.get("stint_number"))
        }

        overlap_keys = set(o_map.keys()).intersection(set(f_map.keys()))
        identical_count = 0
        conflicts: List[OpenF1Discrepancy] = []

        for key in sorted(list(overlap_keys)):
            o_row = o_map[key]
            f_row = f_map[key]

            o_comp = str(o_row.get("compound") or "").upper()
            f_comp = str(f_row.get("compound") or "").upper()

            if o_comp == f_comp or not o_comp or not f_comp:
                identical_count += 1
            else:
                conflicts.append(OpenF1Discrepancy(
                    category="stints",
                    entity_id=key[0],
                    lap_number=int(o_row.get("lap_start", 0)),
                    field_name="compound",
                    openf1_value=o_comp,
                    fastf1_value=f_comp,
                    absolute_delta=None,
                    tolerance_applied=0.0,
                    authoritative_source="FastF1",
                    resolution_action="FastF1 official Pirelli compound classification retained",
                ))

        msg = (
            f"Stints reconciliation across session {session_key}: {len(overlap_keys)} overlapping stints, "
            f"{identical_count} compound matches ({((identical_count/len(overlap_keys))*100 if overlap_keys else 100):.2f}%), "
            f"{len(conflicts)} discrepancies detected."
        )

        return OpenF1ReconciliationReport(
            session_key=session_key,
            season=season,
            round_num=None,
            category="stints",
            openf1_record_count=len(df_openf1_stints),
            fastf1_record_count=len(df_fastf1_stints),
            overlap_count=len(overlap_keys),
            identical_count=identical_count,
            conflicts_detected=conflicts,
            openf1_only_count=len(set(o_map.keys()) - overlap_keys),
            fastf1_only_count=len(set(f_map.keys()) - overlap_keys),
            authoritative_precedence="FastF1",
            summary_message=msg,
        )
