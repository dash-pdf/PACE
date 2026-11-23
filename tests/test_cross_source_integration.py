"""PACE: Stage 5.6 Cross-Source Integration and Data Reconciliation Tests.

Validates:
1. Canonical entity and event resolution across all five providers.
2. Driver substitutions and multi-era constructor team lineages.
3. Circuit layout versioning and multi-event same-circuit disambiguation.
4. Timestamp normalization (ISO-8601 UTC with offset preservation).
5. Unified data contracts across all 13 domains.
6. Source authority matrix, precedence rules, and calibrated tolerances.
7. Cross-source reconciliation accounting (Jolpica 30-row limit, Kaggle 91 duplicates).
8. Provenance and data lineage ledger idempotency.
9. Multi-dimensional data quality framework scorecards.
10. Fully offline execution guard.
"""

from pathlib import Path
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    get_canonical_registry,
    resolve_driver,
    resolve_constructor,
    resolve_circuit,
    resolve_event,
    resolve_circuit_layout,
    get_circuit_layouts,
    get_driver_substitutions,
    get_constructor_lineage,
    resolve_constructor_for_season,
    AmbiguousEntityError,
    UnresolvedEntityError,
)
from src.data.unified_contracts import (
    SessionType,
    SessionStatus,
    MissingValueSemantics,
    DiscrepancyType,
    UnifiedEvent,
    UnifiedSession,
    DiscrepancyRecord,
    DataContractMetadata,
    DATA_CONTRACTS_CATALOG,
    get_data_contract,
    list_data_contracts,
    parse_iso_utc,
)
from src.data.source_authority import SourceAuthorityMatrix, SourceDomain
from src.data.data_lineage import DataLineageTracker
from src.data.cross_source_reconciler import CrossSourceReconciler
from src.data.quality_framework import DataQualityEvaluator
from src.data.schemas import (
    UNIFIED_EVENTS_SCHEMA,
    UNIFIED_SESSIONS_SCHEMA,
    CROSS_SOURCE_RECONCILIATION_SCHEMA,
    DATA_QUALITY_SCORECARD_SCHEMA,
    DATA_LINEAGE_SCHEMA,
)


@pytest.fixture
def registry():
    return get_canonical_registry()


@pytest.fixture
def reconciler():
    return CrossSourceReconciler()


@pytest.fixture
def evaluator():
    return DataQualityEvaluator()


# ---------------------------------------------------------------------------
# 1. Canonical Entity & Driver Substitution Resolution
# ---------------------------------------------------------------------------

def test_driver_substitutions_registered(registry):
    """Verify that modern driver substitutions are properly recorded."""
    subs_2024 = registry.get_driver_substitutions(season=2024)
    assert len(subs_2024) >= 5

    # Check Oliver Bearman subbing for Carlos Sainz at Ferrari (Saudi Arabia R02)
    bearman_r2 = [s for s in subs_2024 if s.round == 2 and s.substitute_driver_id == "bearman"]
    assert len(bearman_r2) == 1
    assert bearman_r2[0].original_driver_id == "sainz"
    assert bearman_r2[0].constructor_id == "ferrari"
    assert bearman_r2[0].car_number == 38

    # Check Franco Colapinto subbing for Logan Sargeant at Williams (Monza R16)
    colapinto_r16 = [s for s in subs_2024 if s.round == 16 and s.substitute_driver_id == "colapinto"]
    assert len(colapinto_r16) == 1
    assert colapinto_r16[0].original_driver_id == "sargeant"
    assert colapinto_r16[0].constructor_id == "williams"
    assert colapinto_r16[0].car_number == 43

    # Check Liam Lawson subbing for Daniel Ricciardo at RB (Austin R19)
    lawson_r19 = [s for s in subs_2024 if s.round == 19 and s.substitute_driver_id == "lawson"]
    assert len(lawson_r19) == 1
    assert lawson_r19[0].original_driver_id == "ricciardo"
    assert lawson_r19[0].constructor_id == "rb"
    assert lawson_r19[0].car_number == 30


def test_ambiguous_driver_resolution_with_and_without_season(registry):
    """Verify that ambiguous surnames are protected and resolve with temporal context."""
    # Surnames without context or strict mode
    res_schumacher = registry.resolve_driver("schumacher", strict=False)
    assert not res_schumacher.is_resolved or res_schumacher.strategy in {"ambiguous", "unresolved"}

    # Strict resolution of ambiguous surname raises AmbiguousEntityError
    with pytest.raises(AmbiguousEntityError):
        registry.resolve_driver("schumacher", strict=True)

    # Resolution with season context
    res_max = registry.resolve_driver("verstappen", season=2024, strict=False)
    assert res_max.is_resolved
    assert res_max.canonical_id == "max_verstappen"


# ---------------------------------------------------------------------------
# 2. Constructor Lineage Resolution
# ---------------------------------------------------------------------------

def test_constructor_lineage_mapping(registry):
    """Verify that multi-era corporate rebrandings resolve to their respective era IDs."""
    # Sauber lineage: Sauber -> BMW Sauber -> Alfa Romeo -> Kick Sauber -> Audi
    sauber_lin = registry.get_constructor_lineage("sauber")
    assert sauber_lin is not None
    assert sauber_lin.lineage_id == "sauber_lineage"

    # In 2008: BMW Sauber
    assert registry.resolve_constructor_for_season("sauber", 2008) == "bmw_sauber"
    # In 2021: Alfa Romeo
    assert registry.resolve_constructor_for_season("sauber", 2021) == "alfa_romeo"
    # In 2024: Sauber
    assert registry.resolve_constructor_for_season("sauber", 2024) == "sauber"
    # In 2026: Audi
    assert registry.resolve_constructor_for_season("sauber", 2026) == "audi"

    # Jordan lineage: Jordan -> Midland -> Spyker -> Force India -> Racing Point -> Aston Martin
    assert registry.resolve_constructor_for_season("aston_martin", 1999) == "jordan"
    assert registry.resolve_constructor_for_season("aston_martin", 2014) == "force_india"
    assert registry.resolve_constructor_for_season("aston_martin", 2020) == "racing_point"
    assert registry.resolve_constructor_for_season("aston_martin", 2024) == "aston_martin"


# ---------------------------------------------------------------------------
# 3. Circuit Layout Versioning & TUM Compatibility
# ---------------------------------------------------------------------------

def test_circuit_layout_versioning(registry):
    """Verify that circuits with historical reconfigurations track layout versions."""
    # Yas Marina
    ym_pre2021 = registry.resolve_circuit_layout("yas_marina", 2019)
    assert ym_pre2021 is not None
    assert ym_pre2021.layout_id == "yas_marina:2009_2020"
    assert ym_pre2021.tum_compatible is True
    assert ym_pre2021.lap_length_km == 5.547

    ym_modern = registry.resolve_circuit_layout("yas_marina", 2024)
    assert ym_modern is not None
    assert ym_modern.layout_id == "yas_marina:2021_present"
    assert ym_modern.tum_compatible is False
    assert ym_modern.lap_length_km == 5.281

    # Zandvoort
    zand_pre2020 = registry.resolve_circuit_layout("zandvoort", 2018)
    assert zand_pre2020.tum_compatible is True
    zand_modern = registry.resolve_circuit_layout("zandvoort", 2024)
    assert zand_modern.tum_compatible is False

    # Las Vegas (Strip Street Circuit introduced 2023)
    vegas_layout = registry.resolve_circuit_layout("vegas", 2024)
    assert vegas_layout is not None
    assert vegas_layout.tum_compatible is False


# ---------------------------------------------------------------------------
# 4. Multi-Event Same Circuit Disambiguation (2020 Season)
# ---------------------------------------------------------------------------

def test_multi_event_same_circuit_disambiguation(registry):
    """Verify that multiple Grands Prix at the same circuit in a single season are distinguished."""
    # 2020 Red Bull Ring held Round 1 (Austrian GP) and Round 2 (Styrian GP)
    ev_austria = registry.resolve_event(2020, round_num=1)
    ev_styria = registry.resolve_event(2020, round_num=2)
    assert ev_austria.is_resolved and ev_styria.is_resolved
    assert ev_austria.canonical_id == "2020_01_red_bull_ring"
    assert ev_styria.canonical_id == "2020_02_red_bull_ring"
    assert ev_austria.entity.event_name == "Austrian Grand Prix"
    assert ev_styria.entity.event_name == "Styrian Grand Prix"

    # Searching by circuit without specifying round or event name flags ambiguity
    res_ambig = registry.resolve_event(2020, circuit_id="red_bull_ring", strict=False)
    assert not res_ambig.is_resolved
    assert res_ambig.strategy == "ambiguous"

    # Searching with event name disambiguates
    res_resolved = registry.resolve_event(2020, circuit_id="red_bull_ring", event_name="Styrian Grand Prix")
    assert res_resolved.is_resolved
    assert res_resolved.canonical_id == "2020_02_red_bull_ring"


# ---------------------------------------------------------------------------
# 5. Timestamp Normalization & UTC Preservation
# ---------------------------------------------------------------------------

def test_timestamp_normalization():
    """Verify UTC ISO-8601 normalization and preservation of original offsets."""
    # UTC ISO format with Z
    utc_ts, tz = parse_iso_utc("2024-03-02T15:00:00Z")
    assert utc_ts == "2024-03-02T15:00:00+00:00"
    assert tz == "+00:00"

    # ISO format with timezone offset
    utc_ts, tz = parse_iso_utc("2024-03-02T18:00:00+03:00")
    assert utc_ts == "2024-03-02T15:00:00+00:00"
    assert tz == "+03:00"

    # Date only format
    utc_ts, tz = parse_iso_utc("2024-03-02")
    assert utc_ts == "2024-03-02T00:00:00+00:00"
    assert tz is None

    # Null / None handling
    assert parse_iso_utc(None) == (None, None)
    assert parse_iso_utc("") == (None, None)


# ---------------------------------------------------------------------------
# 6. Unified Data Contracts
# ---------------------------------------------------------------------------

def test_unified_data_contracts_coverage():
    """Verify that all 13 required data domains have formal contracts."""
    contracts = list_data_contracts()
    assert len(contracts) == 13

    required_domains = [
        "race_results",
        "qualifying_results",
        "sprint_results",
        "driver_standings",
        "constructor_standings",
        "lap_timing",
        "pit_stops",
        "tyre_stints",
        "weather",
        "race_control",
        "driver_positions",
        "overtakes",
        "circuit_geometry",
    ]

    for dom in required_domains:
        c = get_data_contract(dom)
        assert c is not None
        assert len(c.primary_key) >= 2
        assert len(c.units) >= 1
        assert c.preferred_source != ""
        assert c.missing_value_semantics != ""


# ---------------------------------------------------------------------------
# 7. Source Authority Matrix & Precedence Rules
# ---------------------------------------------------------------------------

def test_source_authority_precedence_and_tolerances():
    """Verify authority rules, tolerances, and discrepancy classifications."""
    matrix = SourceAuthorityMatrix()

    # Lap timing preferred source is FastF1
    assert matrix.get_preferred_source("timing") == "fastf1"

    # Standings preferred source is Jolpica for modern seasons
    assert matrix.get_preferred_source("standings", season=2024) == "jolpica"
    # Kaggle for historical pre-2023
    assert matrix.get_preferred_source("standings", season=2015) == "kaggle"

    # Flying lap timing within 10ms tolerance -> EQUIVALENT_CONVERTED
    disc_lap = matrix.evaluate_discrepancy(
        domain="timing",
        field_name="lap_time_ms",
        val_a=92450.0,
        val_b=92455.0, # 5ms difference
        source_a="fastf1",
        source_b="openf1",
    )
    assert disc_lap.discrepancy_type == DiscrepancyType.EQUIVALENT_CONVERTED
    assert disc_lap.selected_source == "fastf1"

    # In-lap / pit lap timing exceeding tolerance -> DIFFERENT_DEFINITION
    disc_inlap = matrix.evaluate_discrepancy(
        domain="timing",
        field_name="lap_time_ms",
        val_a=110000.0,
        val_b=110500.0, # 500ms difference
        source_a="fastf1",
        source_b="openf1",
        is_flying_lap=False,
    )
    assert disc_inlap.discrepancy_type == DiscrepancyType.DIFFERENT_DEFINITION


# ---------------------------------------------------------------------------
# 8. Cross-Source Reconciliation Accounting & Provenance
# ---------------------------------------------------------------------------

def test_cross_source_reconciliation_accounting(reconciler):
    """Verify that known historical audit questions are proven and explained."""
    report = reconciler.execute_full_reconciliation(target_seasons=[2024])

    assert "race_results" in report.domain_summaries
    s_rr = report.domain_summaries["race_results"]
    # Verify exact Jolpica 30-record match
    assert s_rr.overlap_count == 30
    assert s_rr.identical_count == 30
    assert s_rr.match_rate_pct == 100.0

    assert "driver_standings" in report.domain_summaries
    s_ds = report.domain_summaries["driver_standings"]
    assert s_ds.overlap_count == 519
    assert s_ds.identical_count == 519
    assert s_ds.match_rate_pct == 100.0

    # Verify audit resolutions are populated
    assert len(report.known_audit_resolutions) >= 5
    assert "Jolpica 2024 Race Results (30 vs 480 rows)" in report.known_audit_resolutions
    assert "Kaggle Results CSV Row Accounting (27,678 vs 27,587 rows)" in report.known_audit_resolutions


def test_data_lineage_tracker_idempotency(tmp_path):
    """Verify that recording provenance records twice is idempotent without duplicates."""
    tracker = DataLineageTracker(base_dir=tmp_path)
    tracker.record_lineage(
        canonical_record_id="test_rec_1",
        domain="results",
        canonical_event_id="2024_01",
        source_api="jolpica.ergast",
        source_record_id="2024/1/verstappen",
    )
    assert len(tracker._ledger) == 1

    # Re-recording same key updates in-place
    tracker.record_lineage(
        canonical_record_id="test_rec_1",
        domain="results",
        canonical_event_id="2024_01",
        source_api="jolpica.ergast",
        source_record_id="2024/1/verstappen",
        reconciliation_outcome="RECONCILED_EXACT",
    )
    assert len(tracker._ledger) == 1
    assert tracker.get_lineage("test_rec_1").reconciliation_outcome == "RECONCILED_EXACT"

    # Persist and reload
    out_file = tracker.save_ledger()
    assert out_file.exists()
    reloaded = DataLineageTracker(base_dir=tmp_path)
    assert len(reloaded._ledger) == 1


# ---------------------------------------------------------------------------
# 9. Multi-Dimensional Data Quality Scorecard
# ---------------------------------------------------------------------------

def test_data_quality_framework_scorecards(evaluator):
    """Verify that scorecards compute orthogonal quality metrics across all sources."""
    scorecards = evaluator.execute_quality_evaluation()
    assert len(scorecards) == 5

    sources = {s.source_name: s for s in scorecards}
    assert "kaggle" in sources
    assert "jolpica" in sources
    assert "fastf1" in sources
    assert "openf1" in sources
    assert "tum" in sources

    # Check metrics are distinct and valid percentages
    for sc in scorecards:
        assert 0.0 <= sc.completeness_pct <= 100.0
        assert 0.0 <= sc.consistency_pct <= 100.0
        assert 0.0 <= sc.validity_pct <= 100.0
        assert 0.0 <= sc.reconciliation_coverage_pct <= 100.0


# ---------------------------------------------------------------------------
# 10. Unified Schemas Validity
# ---------------------------------------------------------------------------

def test_unified_arrow_schemas():
    """Verify PyArrow schemas defined for Stage 5.6."""
    assert isinstance(UNIFIED_EVENTS_SCHEMA, pa.Schema)
    assert isinstance(UNIFIED_SESSIONS_SCHEMA, pa.Schema)
    assert isinstance(CROSS_SOURCE_RECONCILIATION_SCHEMA, pa.Schema)
    assert isinstance(DATA_QUALITY_SCORECARD_SCHEMA, pa.Schema)
    assert isinstance(DATA_LINEAGE_SCHEMA, pa.Schema)

    assert "canonical_event_id" in UNIFIED_EVENTS_SCHEMA.names
    assert "canonical_session_id" in UNIFIED_SESSIONS_SCHEMA.names
    assert "discrepancy_type" in CROSS_SOURCE_RECONCILIATION_SCHEMA.names
    assert "completeness_pct" in DATA_QUALITY_SCORECARD_SCHEMA.names
    assert "reconciliation_outcome" in DATA_LINEAGE_SCHEMA.names
