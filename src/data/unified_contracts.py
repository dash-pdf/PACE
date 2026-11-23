"""PACE: Unified Data Contracts and Domain Specifications.

Establishes the common canonical event, session, and reconciliation data models
governing cross-source integration across FastF1, Kaggle, Jolpica, OpenF1,
and the TUM Racetrack Database.

Defines:
1. Standard session types and session status enums.
2. Missing value semantics and discrepancy classification taxonomies.
3. Unified event and session data classes with ISO-8601 UTC timestamp normalization.
4. Formal data contracts for overlapping domains (keys, units, semantics, authority).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union

from src.data.canonical_registry import normalize_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SessionType(str, Enum):
    """Authoritative Formula 1 session classifications."""
    FP1 = "fp1"
    FP2 = "fp2"
    FP3 = "fp3"
    QUALIFYING = "qualifying"
    SPRINT_QUALIFYING = "sprint_qualifying"  # Includes 2023 Sprint Shootout
    SPRINT = "sprint"
    RACE = "race"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, val: Any) -> SessionType:
        """Parse raw session title or code to standard SessionType."""
        if val is None:
            return cls.UNKNOWN
        s = str(val).strip().lower()
        s_norm = re.sub(r"[\s\-_]+", "", s)

        if s_norm in {"fp1", "practice1", "free_practice_1", "p1"}:
            return cls.FP1
        if s_norm in {"fp2", "practice2", "free_practice_2", "p2"}:
            return cls.FP2
        if s_norm in {"fp3", "practice3", "free_practice_3", "p3"}:
            return cls.FP3
        if s_norm in {"sprintqualifying", "sprintshootout", "sprintqualy", "sq"}:
            return cls.SPRINT_QUALIFYING
        if s_norm in {"sprint", "sprintrace", "s"}:
            return cls.SPRINT
        if s_norm in {"qualifying", "qualy", "q", "qual"}:
            return cls.QUALIFYING
        if s_norm in {"race", "grandprix", "gp", "r"}:
            return cls.RACE
        return cls.UNKNOWN


class SessionStatus(str, Enum):
    """Session execution and data availability state."""
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ABANDONED = "ABANDONED"


class MissingValueSemantics(str, Enum):
    """Explicit taxonomy distinguishing why a datum is absent."""
    UNAVAILABLE = "UNAVAILABLE"        # Source schema physically does not record this attribute (e.g. TUM 3D elevation)
    NOT_APPLICABLE = "NOT_APPLICABLE"  # Attribute does not apply (e.g. Sprint points in non-sprint race, Q3 for knocked-out driver)
    NOT_SUPPLIED = "NOT_SUPPLIED"      # Attribute supported by source, but omitted in this specific record or stream
    UNKNOWN = "UNKNOWN"                # Value cannot be definitively resolved from available evidence


class DiscrepancyType(str, Enum):
    """Taxonomy of discrepancies between overlapping data sources."""
    EXACT_MATCH = "EXACT_MATCH"                      # Values are strictly identical
    EQUIVALENT_CONVERTED = "EQUIVALENT_CONVERTED"    # Identical after unit conversion or numerical float rounding (<= 10ms for timing)
    TIMESTAMP_OFFSET = "TIMESTAMP_OFFSET"            # Same physical event with offset due to timezone or trigger latency
    DIFFERENT_DEFINITION = "DIFFERENT_DEFINITION"    # Attributes have different operational definitions (e.g. pit-stop vs transit loss)
    HISTORICAL_REVISION = "HISTORICAL_REVISION"      # Official retroactive FIA modification (penalties, re-classifications)
    MISSING_FROM_SOURCE = "MISSING_FROM_SOURCE"      # Present in one source but absent in comparator
    CONFLICTING_VALUE = "CONFLICTING_VALUE"          # Unreconciled contradictory data
    UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"      # Ambiguous entity resolution
    UNRESOLVED_DISCREPANCY = "UNRESOLVED_DISCREPANCY"# Generic discrepancy awaiting human/expert review


# ---------------------------------------------------------------------------
# Timestamp Normalization Helper
# ---------------------------------------------------------------------------

def parse_iso_utc(ts_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parse a timestamp string into normalized (UTC_ISO_8601, original_tz_offset).
    
    Returns:
        (utc_iso, original_offset_str). If cannot parse, returns (None, None).
    """
    if not ts_str or not isinstance(ts_str, str):
        return None, None
    s = ts_str.strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None, None

    try:
        # Check if already ISO format with offset or Z
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
            return dt.astimezone(timezone.utc).isoformat(), "+00:00"
        
        # Match offset like +02:00 or -05:00
        offset_match = re.search(r"([+-]\d{2}:\d{2})$", s)
        if offset_match:
            offset_str = offset_match.group(1)
            dt = datetime.fromisoformat(s)
            return dt.astimezone(timezone.utc).isoformat(), offset_str

        # Match date only: YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            # Do NOT infer a fake time of day if only date is provided
            return f"{s}T00:00:00+00:00", None

        # Try standard parsing
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).isoformat(), str(dt.tzinfo)
        else:
            # Naive datetime: do not guess timezone, preserve as-is with None offset
            return dt.replace(tzinfo=timezone.utc).isoformat(), "naive"
    except Exception as e:
        logger.debug(f"Unable to parse timestamp '{ts_str}': {e}")
        return None, None


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

@dataclass
class UnifiedEvent:
    """Canonical representation of a Formula 1 Grand Prix weekend event."""
    season: int
    round: int
    canonical_event_id: str                      # '{season}_{round:02d}_{circuit_id}'
    official_name: str                           # e.g., 'Formula 1 Heineken Silver Las Vegas Grand Prix 2024'
    event_name: str                              # e.g., 'Las Vegas Grand Prix'
    circuit_id: str                              # Canonical circuit identifier (e.g., 'vegas')
    circuit_layout_id: str                       # Versioned circuit layout (e.g., 'vegas:2023_present')
    country: str
    locality: str
    event_date: str                              # Sunday race date YYYY-MM-DD
    has_sprint: bool = False
    source_event_keys: Dict[str, Any] = field(default_factory=dict)
    source_apis: List[str] = field(default_factory=list)
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to PyArrow-compatible dictionary."""
        return {
            "canonical_event_id": self.canonical_event_id,
            "season": int(self.season),
            "round": int(self.round),
            "official_name": str(self.official_name),
            "event_name": str(self.event_name),
            "circuit_id": str(self.circuit_id),
            "circuit_layout_id": str(self.circuit_layout_id),
            "country": str(self.country),
            "locality": str(self.locality),
            "event_date": str(self.event_date),
            "has_sprint": bool(self.has_sprint),
            "source_event_keys_json": json.dumps(self.source_event_keys),
            "source_apis": ",".join(sorted(self.source_apis)),
            "retrieved_at": str(self.retrieved_at),
        }


@dataclass
class UnifiedSession:
    """Canonical representation of an individual on-track session."""
    canonical_session_id: str                    # '{season}_{round:02d}_{session_type}'
    canonical_event_id: str
    season: int
    round: int
    session_type: SessionType
    session_name: str                            # e.g., 'Practice 1', 'Qualifying', 'Sprint', 'Race'
    circuit_id: str
    circuit_layout_id: str
    start_time_utc: Optional[str] = None         # ISO-8601 UTC
    end_time_utc: Optional[str] = None           # ISO-8601 UTC
    original_tz: Optional[str] = None            # Preserved timezone offset (e.g. '+03:00')
    original_start_time: Optional[str] = None    # Raw unmodified source timestamp
    status: SessionStatus = SessionStatus.COMPLETED
    source_session_keys: Dict[str, Any] = field(default_factory=dict)
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to PyArrow-compatible dictionary."""
        return {
            "canonical_session_id": self.canonical_session_id,
            "canonical_event_id": self.canonical_event_id,
            "season": int(self.season),
            "round": int(self.round),
            "session_type": self.session_type.value,
            "session_name": str(self.session_name),
            "circuit_id": str(self.circuit_id),
            "circuit_layout_id": str(self.circuit_layout_id),
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "original_tz": self.original_tz,
            "original_start_time": self.original_start_time,
            "status": self.status.value,
            "source_session_keys_json": json.dumps(self.source_session_keys),
            "retrieved_at": str(self.retrieved_at),
        }


@dataclass
class DiscrepancyRecord:
    """Detailed record of a discrepancy detected across overlapping data sources."""
    domain: str                                  # 'results', 'standings', 'timing', 'stints', etc.
    entity_id: str                               # driver_id, constructor_id, or composite key
    season: int
    round: int
    session_type: str                            # 'race', 'qualifying', etc.
    field_name: str                              # 'finish_position', 'lap_time_ms', 'points'
    source_a: str                                # Primary source
    val_a_str: str
    source_b: str                                # Comparator source
    val_b_str: str
    discrepancy_type: DiscrepancyType
    absolute_delta: Optional[float] = None
    tolerance_applied: float = 0.0
    selected_source: str = ""
    selection_reason: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to PyArrow-compatible dictionary."""
        return {
            "domain": str(self.domain),
            "entity_id": str(self.entity_id),
            "season": int(self.season),
            "round": int(self.round),
            "session_type": str(self.session_type),
            "field_name": str(self.field_name),
            "source_a": str(self.source_a),
            "val_a_str": str(self.val_a_str),
            "source_b": str(self.source_b),
            "val_b_str": str(self.val_b_str),
            "discrepancy_type": self.discrepancy_type.value,
            "absolute_delta": float(self.absolute_delta) if self.absolute_delta is not None else None,
            "tolerance_applied": float(self.tolerance_applied),
            "selected_source": str(self.selected_source),
            "selection_reason": str(self.selection_reason),
            "recorded_at": str(self.recorded_at),
        }


@dataclass
class QualityScorecard:
    """Multi-dimensional data quality evaluation for a specific source and dataset category."""
    source_name: str                             # 'fastf1', 'kaggle', 'jolpica', 'openf1', 'tum'
    dataset_category: str                        # 'race_results', 'driver_standings', etc.
    records_evaluated: int
    completeness_pct: float                      # Required non-null completeness (0.0 to 100.0)
    consistency_pct: float                       # Referential integrity / cross-field consistency
    validity_pct: float                          # Constraint / range validity
    reconciliation_coverage_pct: float           # Overlap percentage with secondary sources
    unresolved_count: int = 0                    # Number of unresolved identities or conflicting items
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": str(self.source_name),
            "dataset_category": str(self.dataset_category),
            "records_evaluated": int(self.records_evaluated),
            "completeness_pct": float(self.completeness_pct),
            "consistency_pct": float(self.consistency_pct),
            "validity_pct": float(self.validity_pct),
            "reconciliation_coverage_pct": float(self.reconciliation_coverage_pct),
            "unresolved_count": int(self.unresolved_count),
            "evaluated_at": str(self.evaluated_at),
        }


@dataclass
class DataLineageRecord:
    """Audit record capturing complete provenance and transformation lineage."""
    canonical_record_id: str
    domain: str
    canonical_event_id: str
    source_api: str
    source_record_id: str
    ingestion_timestamp: str
    transformation_pipeline: str
    reconciliation_outcome: str
    selected_source: str
    selection_reason: str
    pipeline_version: str = "PACE-1.0-Stage5.6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_record_id": str(self.canonical_record_id),
            "domain": str(self.domain),
            "canonical_event_id": str(self.canonical_event_id),
            "source_api": str(self.source_api),
            "source_record_id": str(self.source_record_id),
            "ingestion_timestamp": str(self.ingestion_timestamp),
            "transformation_pipeline": str(self.transformation_pipeline),
            "reconciliation_outcome": str(self.reconciliation_outcome),
            "selected_source": str(self.selected_source),
            "selection_reason": str(self.selection_reason),
            "pipeline_version": str(self.pipeline_version),
        }


# ---------------------------------------------------------------------------
# Data Contract Specifications (13 Required Domains)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataContractMetadata:
    """Formal contract defining schema keys, units, semantics, and authority."""
    domain: str
    canonical_schema_name: str
    primary_key: List[str]
    natural_key: List[str]
    units: Dict[str, str]
    timestamp_semantics: str
    missing_value_semantics: str
    preferred_source: str
    reconciliation_rule: str


DATA_CONTRACTS_CATALOG: Dict[str, DataContractMetadata] = {
    "race_results": DataContractMetadata(
        domain="race_results",
        canonical_schema_name="RACE_RESULTS_SCHEMA",
        primary_key=["season", "round", "driver_id"],
        natural_key=["season", "round", "driver_number"],
        units={"points": "points", "total_time_ms": "milliseconds", "fastest_lap_speed": "km/h"},
        timestamp_semantics="Session start and duration in UTC; fastest lap time in MM:SS.sss",
        missing_value_semantics="NOT_APPLICABLE for DNS/DSQ times; NOT_SUPPLIED for legacy races lacking micro-sector times",
        preferred_source="Jolpica (official FIA classification) for modern; Kaggle archive for 1950-2022",
        reconciliation_rule="Positions and points must match exactly. Where provisional differs from final, authoritative FIA Jolpica applies.",
    ),
    "qualifying_results": DataContractMetadata(
        domain="qualifying_results",
        canonical_schema_name="QUALIFYING_RESULTS_SCHEMA",
        primary_key=["season", "round", "driver_id"],
        natural_key=["season", "round", "driver_number"],
        units={"q1_time_ms": "milliseconds", "q2_time_ms": "milliseconds", "q3_time_ms": "milliseconds"},
        timestamp_semantics="Session timestamps in UTC; lap times recorded as elapsed lap durations in milliseconds",
        missing_value_semantics="NOT_APPLICABLE for drivers eliminated in prior qualifying segments (Q2/Q3)",
        preferred_source="FastF1 for 2023-2026; Kaggle/Jolpica for historical eras",
        reconciliation_rule="Times must reconcile within 10 ms tolerance. Knockout stages align to regulatory eras.",
    ),
    "sprint_results": DataContractMetadata(
        domain="sprint_results",
        canonical_schema_name="SPRINT_RESULTS_SCHEMA",
        primary_key=["season", "round", "driver_id"],
        natural_key=["season", "round", "driver_number"],
        units={"points": "points", "total_time_ms": "milliseconds"},
        timestamp_semantics="Saturday Sprint session start in UTC; elapsed duration in milliseconds",
        missing_value_semantics="NOT_APPLICABLE for non-Sprint events (seasons < 2021 or standard Grand Prix weekends)",
        preferred_source="Jolpica for official points distribution; FastF1 for timing and telemetry",
        reconciliation_rule="Points distribution evaluated against regulatory era tables (3-2-1 in 2021; 8-7-...-1 in 2022-2026).",
    ),
    "driver_standings": DataContractMetadata(
        domain="driver_standings",
        canonical_schema_name="DRIVER_STANDINGS_SCHEMA",
        primary_key=["season", "round", "driver_id"],
        natural_key=["season", "round", "position"],
        units={"points": "points", "wins": "count"},
        timestamp_semantics="Cumulative championship point accumulation as of the conclusion of specified round",
        missing_value_semantics="UNAVAILABLE for non-participating drivers in season; NOT_APPLICABLE for unclassified competitors",
        preferred_source="Jolpica (authoritative Ergast successor with verified tie-breakers); Kaggle archive for historical",
        reconciliation_rule="Points and positions must match 100%. Idempotent primary-key deduplication replaces existing records without row inflation.",
    ),
    "constructor_standings": DataContractMetadata(
        domain="constructor_standings",
        canonical_schema_name="CONSTRUCTOR_STANDINGS_SCHEMA",
        primary_key=["season", "round", "constructor_id"],
        natural_key=["season", "round", "position"],
        units={"points": "points", "wins": "count"},
        timestamp_semantics="Cumulative constructor championship point accumulation as of the conclusion of specified round",
        missing_value_semantics="UNAVAILABLE prior to 1958 inaugural International Cup for F1 Manufacturers",
        preferred_source="Jolpica for modern and historical standings; Kaggle archive for multi-decade persistence",
        reconciliation_rule="Scoring rules account for single-car constructor scoring (1958-1978) vs two-car scoring (1979-2026).",
    ),
    "lap_timing": DataContractMetadata(
        domain="lap_timing",
        canonical_schema_name="LAPS_SCHEMA",
        primary_key=["season", "round", "session_type", "driver_id", "lap_number"],
        natural_key=["season", "round", "driver_number", "lap_number"],
        units={"lap_time_ms": "milliseconds", "s1_ms": "milliseconds", "s2_ms": "milliseconds", "s3_ms": "milliseconds"},
        timestamp_semantics="Transponder loop crossing times normalized to UTC ISO-8601",
        missing_value_semantics="NOT_SUPPLIED for deleted/track-limits laps; NOT_APPLICABLE for in-laps / out-laps sector 3",
        preferred_source="FastF1 (FOM transponder precision); OpenF1 as continuous streaming comparator",
        reconciliation_rule="Clean flying laps match within 10 ms. In-laps and pit laps flagged as DIFFERENT_DEFINITION.",
    ),
    "pit_stops": DataContractMetadata(
        domain="pit_stops",
        canonical_schema_name="PIT_STOPS_SCHEMA",
        primary_key=["season", "round", "driver_id", "stop_number"],
        natural_key=["season", "round", "driver_number", "lap_number"],
        units={"stop_duration_ms": "milliseconds", "transit_duration_ms": "milliseconds"},
        timestamp_semantics="Pit entry and exit transponder loop crossing times in UTC",
        missing_value_semantics="UNAVAILABLE prior to official 2011 pit stop timing transponder mandate",
        preferred_source="FastF1 for pit lane transit duration; OpenF1 for stationary stop duration",
        reconciliation_rule="Explicitly distinguish stationary box stop duration from total pit lane transit loss.",
    ),
    "tyre_stints": DataContractMetadata(
        domain="tyre_stints",
        canonical_schema_name="OPENF1_STINTS_SCHEMA",
        primary_key=["session_key", "driver_id", "stint_number"],
        natural_key=["season", "round", "driver_number", "stint_number"],
        units={"lap_start": "lap_count", "lap_end": "lap_count", "tyre_age_at_start": "laps"},
        timestamp_semantics="Stint start and conclusion tied to session lap intervals",
        missing_value_semantics="UNKNOWN for unmarked test compounds; NOT_SUPPLIED where telemetry telemetry drops",
        preferred_source="FastF1 for authoritative compound classification (C1-C5 / Soft-Med-Hard); OpenF1 for live stint bounds",
        reconciliation_rule="Stint start and end lap bounds must agree within 1 lap across pit stop records.",
    ),
    "weather": DataContractMetadata(
        domain="weather",
        canonical_schema_name="OPENF1_WEATHER_SCHEMA",
        primary_key=["session_key", "timestamp_utc"],
        natural_key=["season", "round", "session_type", "timestamp_utc"],
        units={"air_temp_c": "Celsius", "track_temp_c": "Celsius", "humidity_pct": "percent", "wind_speed_ms": "m/s"},
        timestamp_semantics="Sampled periodic time series in UTC (nominally 1-minute intervals)",
        missing_value_semantics="NOT_SUPPLIED if specific weather station sensor drops offline",
        preferred_source="OpenF1 for high-frequency time series (1-min intervals); FastF1 for session average summary",
        reconciliation_rule="Track temperatures must reconcile within 1.5 deg C between station telemetry and FOM feed.",
    ),
    "race_control": DataContractMetadata(
        domain="race_control",
        canonical_schema_name="OPENF1_RACE_CONTROL_SCHEMA",
        primary_key=["session_key", "timestamp_utc", "category"],
        natural_key=["season", "round", "timestamp_utc", "category"],
        units={"flag": "categorical", "scope": "categorical"},
        timestamp_semantics="Verbatim FIA Race Control server notification timestamp in UTC",
        missing_value_semantics="NOT_APPLICABLE for track-wide messages not targeting a specific driver",
        preferred_source="OpenF1 (verbatim FIA race director message text and millisecond timestamps)",
        reconciliation_rule="Safety Car, VSC, and Red Flag state transitions must align with FastF1 track status codes.",
    ),
    "driver_positions": DataContractMetadata(
        domain="driver_positions",
        canonical_schema_name="OPENF1_INTERVALS_SCHEMA",
        primary_key=["session_key", "timestamp_utc", "driver_id"],
        natural_key=["season", "round", "lap_number", "driver_number"],
        units={"gap_to_leader_s": "seconds", "interval_s": "seconds"},
        timestamp_semantics="High-frequency time series generated at sector boundary transponder loops",
        missing_value_semantics="NOT_APPLICABLE for race leader (interval=0, gap=0); UNAVAILABLE for retired cars",
        preferred_source="OpenF1 (continuous interval stream); FastF1 as transponder ground-truth at finish line",
        reconciliation_rule="Lap crossing order must match classified race lap order exactly.",
    ),
    "overtakes": DataContractMetadata(
        domain="overtakes",
        canonical_schema_name="OPENF1_OVERTAKES_SCHEMA",
        primary_key=["session_key", "lap_number", "overtaking_driver_id", "overtaken_driver_id"],
        natural_key=["season", "round", "lap_number", "overtaking_driver_id"],
        units={"position_change": "count"},
        timestamp_semantics="Transponder loop timestamp where position change was recorded in UTC",
        missing_value_semantics="NOT_APPLICABLE for pit lane position swaps or lapping slower traffic",
        preferred_source="OpenF1 (corroborative on-track overtakes stream); cross-checked against FastF1 lap position diffs",
        reconciliation_rule="Position improvements during green flag conditions confirmed against timing sheets.",
    ),
    "circuit_geometry": DataContractMetadata(
        domain="circuit_geometry",
        canonical_schema_name="TUM_TRACK_GEOMETRY_SCHEMA",
        primary_key=["circuit_id", "circuit_layout_id", "point_index"],
        natural_key=["circuit_id", "s_m"],
        units={"x_m": "meters", "y_m": "meters", "track_width_m": "meters", "curvature_radpm": "rad/m", "radius_m": "meters"},
        timestamp_semantics="Spatial coordinate parameterization along cumulative arc length s; static geometric model",
        missing_value_semantics="UNAVAILABLE for 3D elevation and banking in TUM; UNAVAILABLE for non-covered circuits (e.g. Las Vegas)",
        preferred_source="TUM Racetrack Database (where covered and layout-valid); Canonical/FIA metrics for missing circuits",
        reconciliation_rule="Layout validity intervals must match season: pre-2021 Yas Marina and pre-2020 Zandvoort restricted to valid years.",
    ),
}


def get_data_contract(domain: str) -> Optional[DataContractMetadata]:
    """Retrieve formal data contract specification by domain name."""
    return DATA_CONTRACTS_CATALOG.get(domain)


def list_data_contracts() -> List[DataContractMetadata]:
    """List all registered data contracts across PACE."""
    return list(DATA_CONTRACTS_CATALOG.values())
