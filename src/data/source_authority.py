"""PACE: Source Authority Matrix & Precedence Framework.

Establishes formal, domain-specific authority rules across PACE's five external
data providers (FastF1, Kaggle Historical, Jolpica, OpenF1, TUM Racetrack Database).

Enforces:
1. Domain-level source authority hierarchy.
2. Field-level precision, units, and tolerance specifications.
3. Principled discrepancy arbitration without silent overwrites.
4. Layout and era-dependent authority switching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Optional, Set, Tuple, Any

from src.data.unified_contracts import DiscrepancyType, DiscrepancyRecord

logger = logging.getLogger(__name__)


class SourceDomain(str, Enum):
    """Broad data domain categories governing authority precedence."""
    HIGH_RES_TIMING_TELEMETRY = "timing_telemetry"
    HISTORICAL_RESULTS = "historical_results"
    CHAMPIONSHIP_STANDINGS = "championship_standings"
    CONTINUOUS_STREAMS_MESSAGES = "continuous_streams_messages"
    CIRCUIT_GEOMETRY = "circuit_geometry"


@dataclass(frozen=True)
class FieldAuthorityRule:
    """Authority rule governing a specific field within a data domain."""
    domain: str
    field_name: str
    preferred_source: str
    secondary_source: str
    tolerance: float                        # Numerical difference threshold
    unit: str
    rationale: str
    era_cutoff_season: Optional[int] = None # e.g. 2023 for FastF1 primary


class SourceAuthorityMatrix:
    """Evaluates field-level precedence, applies calibrated tolerances, and arbitrates conflicts."""

    # Domain Precedence Defaults
    DOMAIN_AUTHORITY: Dict[str, str] = {
        SourceDomain.HIGH_RES_TIMING_TELEMETRY.value: "FastF1 (direct FOM transponder telemetry feed)",
        SourceDomain.HISTORICAL_RESULTS.value: "Jolpica (official FIA classification) for modern; Kaggle archive for 1950-2022",
        SourceDomain.CHAMPIONSHIP_STANDINGS.value: "Jolpica (official Ergast successor with verified tie-breakers)",
        SourceDomain.CONTINUOUS_STREAMS_MESSAGES.value: "OpenF1 (high-frequency intervals and verbatim FIA race director text)",
        SourceDomain.CIRCUIT_GEOMETRY.value: "TUM Racetrack Database (where covered and layout-valid); Canonical/FIA for missing",
    }

    # Field-Specific Authority Rules
    FIELD_RULES: Dict[Tuple[str, str], FieldAuthorityRule] = {
        ("results", "finish_position"): FieldAuthorityRule(
            domain="results",
            field_name="finish_position",
            preferred_source="jolpica",
            secondary_source="kaggle",
            tolerance=0.0,
            unit="integer",
            rationale="Jolpica represents official FIA final classifications after post-race penalties and disqualifications.",
        ),
        ("results", "points"): FieldAuthorityRule(
            domain="results",
            field_name="points",
            preferred_source="jolpica",
            secondary_source="kaggle",
            tolerance=0.05,
            unit="points",
            rationale="Official FIA points table; 0.05 tolerance accounts for IEEE-754 floating-point half-points.",
        ),
        ("results", "grid_position"): FieldAuthorityRule(
            domain="results",
            field_name="grid_position",
            preferred_source="fastf1",
            secondary_source="jolpica",
            tolerance=0.0,
            unit="integer",
            rationale="FastF1 records official published starting grid positions after pit lane start penalties.",
            era_cutoff_season=2023,
        ),
        ("standings", "points"): FieldAuthorityRule(
            domain="standings",
            field_name="points",
            preferred_source="jolpica",
            secondary_source="kaggle",
            tolerance=0.05,
            unit="points",
            rationale="Jolpica provides authoritative multi-round points tally including sprint points and fastest lap bonuses.",
        ),
        ("standings", "position"): FieldAuthorityRule(
            domain="standings",
            field_name="position",
            preferred_source="jolpica",
            secondary_source="kaggle",
            tolerance=0.0,
            unit="integer",
            rationale="Authoritative tie-breaking regulations (countback on 1st, 2nd, 3rd places).",
        ),
        ("timing", "lap_time_ms"): FieldAuthorityRule(
            domain="timing",
            field_name="lap_time_ms",
            preferred_source="fastf1",
            secondary_source="openf1",
            tolerance=10.0,                 # 10 ms tolerance on flying laps
            unit="milliseconds",
            rationale="FastF1 directly ingests FOM timing feed millisecond transponder events. OpenF1 floating-point duration has rounding noise.",
        ),
        ("timing", "s1_ms"): FieldAuthorityRule(
            domain="timing",
            field_name="s1_ms",
            preferred_source="fastf1",
            secondary_source="openf1",
            tolerance=10.0,
            unit="milliseconds",
            rationale="FastF1 sector timing loops are synchronized directly with FIA beam crossings.",
        ),
        ("weather", "track_temp_c"): FieldAuthorityRule(
            domain="weather",
            field_name="track_temp_c",
            preferred_source="openf1",
            secondary_source="fastf1",
            tolerance=1.5,                  # 1.5 deg C tolerance
            unit="Celsius",
            rationale="OpenF1 streams 1-minute continuous track sensor samples; FastF1 provides session summary averages.",
        ),
        ("weather", "air_temp_c"): FieldAuthorityRule(
            domain="weather",
            field_name="air_temp_c",
            preferred_source="openf1",
            secondary_source="fastf1",
            tolerance=1.0,
            unit="Celsius",
            rationale="OpenF1 1-minute time series captures rapid ambient shifts (e.g. cloud cover, sunset).",
        ),
        ("stints", "compound"): FieldAuthorityRule(
            domain="stints",
            field_name="compound",
            preferred_source="fastf1",
            secondary_source="openf1",
            tolerance=0.0,
            unit="categorical",
            rationale="FastF1 provides definitive tyre compound coding (C1-C5 and Soft/Medium/Hard/Intermediate/Wet).",
        ),
        ("stints", "lap_start"): FieldAuthorityRule(
            domain="stints",
            field_name="lap_start",
            preferred_source="fastf1",
            secondary_source="openf1",
            tolerance=1.0,                  # Within 1 lap
            unit="lap_number",
            rationale="Pit stop in-lap vs out-lap definitions can cause a 1-lap offset between providers.",
        ),
        ("geometry", "track_length_m"): FieldAuthorityRule(
            domain="geometry",
            field_name="track_length_m",
            preferred_source="tum",
            secondary_source="canonical_fia",
            tolerance=15.0,                 # 15 meter tolerance for centerline calculation
            unit="meters",
            rationale="TUM centerline 2D differential geometry arc length; must match within 15m unless reconfigured.",
        ),
    }

    def __init__(self) -> None:
        pass

    def get_preferred_source(
        self,
        domain: str,
        field_name: Optional[str] = None,
        season: Optional[int] = None,
    ) -> str:
        """Determine preferred source considering domain, field, and historical season."""
        # 1. Field-specific rule check
        if field_name:
            key = (domain, field_name)
            if key in self.FIELD_RULES:
                rule = self.FIELD_RULES[key]
                if rule.era_cutoff_season is not None and season is not None:
                    if season >= rule.era_cutoff_season:
                        return rule.preferred_source
                    else:
                        return rule.secondary_source
                return rule.preferred_source

        # 2. Domain-level rule check
        if domain == "results":
            if season is not None and season <= 2022:
                return "kaggle"
            return "jolpica"
        elif domain == "standings":
            if season is not None and season <= 2022:
                return "kaggle"
            return "jolpica"
        elif domain in {"timing", "telemetry"}:
            return "fastf1"
        elif domain in {"intervals", "messages", "race_control"}:
            return "openf1"
        elif domain == "geometry":
            return "tum"

        return "fastf1"

    def evaluate_discrepancy(
        self,
        domain: str,
        field_name: str,
        val_a: Any,
        val_b: Any,
        source_a: str,
        source_b: str,
        entity_id: str = "",
        season: int = 2024,
        round_num: int = 1,
        session_type: str = "race",
        is_flying_lap: bool = True,
    ) -> DiscrepancyRecord:
        """Systematically evaluate two source values, apply tolerances, and select authoritative value."""
        rule = self.FIELD_RULES.get((domain, field_name))
        tolerance = rule.tolerance if rule else 0.0
        preferred = self.get_preferred_source(domain, field_name, season=season)

        # 1. Check for Exact Match
        if val_a == val_b or (val_a is None and val_b is None):
            return DiscrepancyRecord(
                domain=domain,
                entity_id=entity_id,
                season=season,
                round=round_num,
                session_type=session_type,
                field_name=field_name,
                source_a=source_a,
                val_a_str=str(val_a),
                source_b=source_b,
                val_b_str=str(val_b),
                discrepancy_type=DiscrepancyType.EXACT_MATCH,
                absolute_delta=0.0,
                tolerance_applied=tolerance,
                selected_source=source_a,
                selection_reason="Values are identical across sources.",
            )

        # 2. Check for Missing from one source
        if val_a is None or val_a == "" or (isinstance(val_a, float) and val_a != val_a):
            return DiscrepancyRecord(
                domain=domain,
                entity_id=entity_id,
                season=season,
                round=round_num,
                session_type=session_type,
                field_name=field_name,
                source_a=source_a,
                val_a_str="NULL",
                source_b=source_b,
                val_b_str=str(val_b),
                discrepancy_type=DiscrepancyType.MISSING_FROM_SOURCE,
                absolute_delta=None,
                tolerance_applied=tolerance,
                selected_source=source_b,
                selection_reason=f"Value missing in {source_a}; populated from {source_b}.",
            )
        if val_b is None or val_b == "" or (isinstance(val_b, float) and val_b != val_b):
            return DiscrepancyRecord(
                domain=domain,
                entity_id=entity_id,
                season=season,
                round=round_num,
                session_type=session_type,
                field_name=field_name,
                source_a=source_a,
                val_a_str=str(val_a),
                source_b=source_b,
                val_b_str="NULL",
                discrepancy_type=DiscrepancyType.MISSING_FROM_SOURCE,
                absolute_delta=None,
                tolerance_applied=tolerance,
                selected_source=source_a,
                selection_reason=f"Value present in {source_a}; missing in {source_b}.",
            )

        # 3. Check for Numeric comparison & Tolerance
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            delta = abs(num_a - num_b)

            if delta <= tolerance:
                # If within tolerance, it's equivalent converted
                winning_source = source_a if preferred.lower() in source_a.lower() else source_b
                return DiscrepancyRecord(
                    domain=domain,
                    entity_id=entity_id,
                    season=season,
                    round=round_num,
                    session_type=session_type,
                    field_name=field_name,
                    source_a=source_a,
                    val_a_str=str(val_a),
                    source_b=source_b,
                    val_b_str=str(val_b),
                    discrepancy_type=DiscrepancyType.EQUIVALENT_CONVERTED,
                    absolute_delta=delta,
                    tolerance_applied=tolerance,
                    selected_source=winning_source,
                    selection_reason=f"Delta ({delta:.3f}) is within calibrated tolerance ({tolerance:.3f}). Selected preferred {winning_source}.",
                )
            else:
                # Exceeds tolerance
                # If timing on in-laps / out-laps: different definition
                if domain == "timing" and not is_flying_lap:
                    return DiscrepancyRecord(
                        domain=domain,
                        entity_id=entity_id,
                        season=season,
                        round=round_num,
                        session_type=session_type,
                        field_name=field_name,
                        source_a=source_a,
                        val_a_str=str(val_a),
                        source_b=source_b,
                        val_b_str=str(val_b),
                        discrepancy_type=DiscrepancyType.DIFFERENT_DEFINITION,
                        absolute_delta=delta,
                        tolerance_applied=tolerance,
                        selected_source=source_a if "fastf1" in source_a.lower() else source_b,
                        selection_reason="Pit in/out lap timing protocols differ between FOM timing loop and provider intervals.",
                    )

                winning_source = source_a if preferred.lower() in source_a.lower() else source_b
                reason = rule.rationale if rule else f"{winning_source} is authoritative for {domain}."
                return DiscrepancyRecord(
                    domain=domain,
                    entity_id=entity_id,
                    season=season,
                    round=round_num,
                    session_type=session_type,
                    field_name=field_name,
                    source_a=source_a,
                    val_a_str=str(val_a),
                    source_b=source_b,
                    val_b_str=str(val_b),
                    discrepancy_type=DiscrepancyType.CONFLICTING_VALUE,
                    absolute_delta=delta,
                    tolerance_applied=tolerance,
                    selected_source=winning_source,
                    selection_reason=f"Values conflict (delta={delta:.3f} > tol {tolerance:.3f}). {reason}",
                )
        except (ValueError, TypeError):
            pass

        # 4. String / Categorical Discrepancy
        winning_source = source_a if preferred.lower() in source_a.lower() else source_b
        return DiscrepancyRecord(
            domain=domain,
            entity_id=entity_id,
            season=season,
            round=round_num,
            session_type=session_type,
            field_name=field_name,
            source_a=source_a,
            val_a_str=str(val_a),
            source_b=source_b,
            val_b_str=str(val_b),
            discrepancy_type=DiscrepancyType.CONFLICTING_VALUE,
            absolute_delta=None,
            tolerance_applied=tolerance,
            selected_source=winning_source,
            selection_reason=f"Categorical values differ. Selected preferred source {winning_source}.",
        )
