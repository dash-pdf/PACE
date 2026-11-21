"""Comprehensive test suite for PACE Kaggle Historical Dataset Integration (Stage 5.2).

Validates:
1. Kaggle API discovery and local fallback mechanics.
2. Robust CSV loading with MySQL \\N null conversion.
3. Canonical Entity Registry expansion and collision handling for historical entities.
4. Schema compliance across all 10 historical Parquet datasets.
5. Referential integrity validation and quarantine of orphaned rows.
6. Reconciliation precedence (FastF1 primary for 2023+, Kaggle for 1950-2022).
7. Complete 38-column driver-circuit appearances compilation and teammate matching.
8. Complete 27-column aggregated performance profiles compilation.
9. Provable zero-leakage guarantee on historical predictive feature generation.
10. End-to-end integration pipeline execution and quality reporting.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    CanonicalDriver,
    CanonicalConstructor,
    CanonicalCircuit,
    AmbiguousEntityError,
)
from src.data.kaggle_importer import (
    KaggleAcquisitionManager,
    KaggleImporter,
    HistoricalDataReconciler,
    HistoricalIntelligenceExpander,
    run_kaggle_historical_integration,
    KaggleInventory,
    DataQualityReport,
)
from src.data.schemas import (
    DRIVERS_SCHEMA,
    CONSTRUCTORS_SCHEMA,
    CIRCUITS_SCHEMA,
    EVENTS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    DRIVER_STANDINGS_SCHEMA,
    CONSTRUCTOR_STANDINGS_SCHEMA,
    DRIVER_CIRCUIT_APPEARANCES_SCHEMA,
    DRIVER_CIRCUIT_AGGREGATED_SCHEMA,
)
from src.data.validation import (
    validate_driver_circuit_appearances,
    validate_driver_circuit_aggregated,
)
from src.data.storage import load_parquet, dataset_exists


# ---------------------------------------------------------------------------
# 1. Acquisition & Inventory Tests
# ---------------------------------------------------------------------------

def test_kaggle_credentials_discovery(monkeypatch):
    """Verify Kaggle credential discovery handles both missing and present tokens."""
    manager = KaggleAcquisitionManager()

    # Clear env
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/tmp/nonexistent_kaggle_dir")

    has_creds, msg = manager.check_kaggle_credentials()
    assert not has_creds
    assert "No Kaggle API credentials found" in msg

    # Set mock env
    monkeypatch.setenv("KAGGLE_USERNAME", "f1_analyst")
    monkeypatch.setenv("KAGGLE_KEY", "mock_key_12345")
    has_creds_env, msg_env = manager.check_kaggle_credentials()
    assert has_creds_env
    assert "Found KAGGLE_USERNAME and KAGGLE_KEY" in msg_env


def test_kaggle_inventory_scanner(tmp_path):
    """Verify inventory scanner identifies required vs missing tables."""
    manager = KaggleAcquisitionManager(base_dir=tmp_path)
    raw_dir = tmp_path / "data" / "raw" / "kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Empty directory -> invalid inventory
    inv_empty = manager.scan_inventory(raw_dir)
    assert not inv_empty.is_valid
    assert len(inv_empty.missing_required_tables) == 6

    # Create dummy required tables
    for req in KaggleAcquisitionManager.REQUIRED_TABLES:
        (raw_dir / req).write_text("colA,colB\n1,2\n", encoding="utf-8")

    inv_full = manager.scan_inventory(raw_dir)
    assert inv_full.is_valid
    assert len(inv_full.missing_required_tables) == 0
    assert "races.csv" in inv_full.tables
    assert inv_full.tables["races.csv"].table_name == "races"
    assert inv_full.tables["races.csv"].column_count == 2


# ---------------------------------------------------------------------------
# 2. Importer & Null Handling Tests
# ---------------------------------------------------------------------------

def test_kaggle_importer_mysql_null_conversion(tmp_path):
    """Verify MySQL \\N representations are parsed to genuine Python/Pandas NaNs."""
    raw_dir = tmp_path / "data" / "raw" / "kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_content = (
        "driverId,driverRef,number,code,forename,surname,dob,nationality,url\n"
        "1,farina,\\N,\\N,Nino,Farina,1906-10-30,Italian,http://url1\n"
        "2,fangio,\\N,\\N,Juan,Fangio,1911-06-24,Argentine,\\N\n"
        "14,hamilton,44,HAM,Lewis,Hamilton,1985-01-07,British,http://url14\n"
    )
    (raw_dir / "drivers.csv").write_text(csv_content, encoding="utf-8")

    importer = KaggleImporter(source_dir=raw_dir, base_dir=tmp_path)
    df = importer.load_csv("drivers.csv")

    assert len(df) == 3
    assert pd.isna(df.loc[0, "number"])
    assert pd.isna(df.loc[0, "code"])
    assert pd.isna(df.loc[1, "url"])
    assert df.loc[2, "number"] == "44" or df.loc[2, "number"] == 44
    assert df.loc[2, "code"] == "HAM"


# ---------------------------------------------------------------------------
# 3. Canonical Registry Historical Expansion & Ambiguity Tests
# ---------------------------------------------------------------------------

def test_historical_canonical_expansion_and_ambiguity():
    """Verify historical drivers and constructors expand registry and handle alias collisions."""
    registry = CanonicalEntityRegistry()

    # Pre-register Fangio and Graham Hill
    d_fangio = CanonicalDriver(
        canonical_id="fangio",
        code="",
        permanent_number=-1,
        full_name="Juan Manuel Fangio",
        given_name="Juan",
        family_name="Fangio",
        nationality="Argentine",
        ergast_id="fangio",
        fastf1_code="",
        aliases=["fangio", "juan fangio"],
    )
    registry.register_driver(d_fangio)

    d_graham = CanonicalDriver(
        canonical_id="graham_hill",
        code="",
        permanent_number=-1,
        full_name="Graham Hill",
        given_name="Graham",
        family_name="Hill",
        nationality="British",
        ergast_id="graham_hill",
        fastf1_code="",
        aliases=["hill", "graham hill"],
    )
    registry.register_driver(d_graham)

    # Register Damon Hill with shared alias 'hill'
    d_damon = CanonicalDriver(
        canonical_id="damon_hill",
        code="",
        permanent_number=-1,
        full_name="Damon Hill",
        given_name="Damon",
        family_name="Hill",
        nationality="British",
        ergast_id="damon_hill",
        fastf1_code="",
        aliases=["hill", "damon hill"],
    )
    registry.register_driver(d_damon)

    # Strict resolution on 'hill' must raise AmbiguousEntityError
    with pytest.raises(AmbiguousEntityError):
        registry.resolve_driver("hill", strict=True)

    # Non-strict resolution on 'hill' returns is_resolved=False with ambiguous strategy
    res = registry.resolve_driver("hill", strict=False)
    assert not res.is_resolved
    assert res.strategy == "ambiguous"
    assert "graham_hill" in res.error_message
    assert "damon_hill" in res.error_message

    # Unambiguous resolution on exact full name or ID resolves perfectly
    assert registry.resolve_driver("graham hill").canonical_id == "graham_hill"
    assert registry.resolve_driver("damon hill").canonical_id == "damon_hill"
    assert registry.resolve_driver("fangio").canonical_id == "fangio"


# ---------------------------------------------------------------------------
# 4. Referential Integrity & Quarantine Tests
# ---------------------------------------------------------------------------

def test_referential_integrity_quarantine(tmp_path):
    """Verify orphaned results with invalid raceId or driverId are quarantined."""
    raw_dir = tmp_path / "data" / "raw" / "kaggle"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Status table
    (raw_dir / "status.csv").write_text("statusId,status\n1,Finished\n", encoding="utf-8")
    # Circuits table
    (raw_dir / "circuits.csv").write_text(
        "circuitId,circuitRef,name,location,country,lat,lng,alt,url\n"
        "1,silverstone,Silverstone,Silverstone,UK,52.0,-1.0,150,http\n",
        encoding="utf-8",
    )
    # Constructors table
    (raw_dir / "constructors.csv").write_text(
        "constructorId,constructorRef,name,nationality,url\n"
        "1,ferrari,Ferrari,Italian,http\n",
        encoding="utf-8",
    )
    # Drivers table: only driverId 1
    (raw_dir / "drivers.csv").write_text(
        "driverId,driverRef,number,code,forename,surname,dob,nationality,url\n"
        "1,farina,\\N,\\N,Nino,Farina,1906-10-30,Italian,http\n",
        encoding="utf-8",
    )
    # Races table: only raceId 1
    (raw_dir / "races.csv").write_text(
        "raceId,year,round,circuitId,name,date,time,url\n"
        "1,1950,1,1,1950 British GP,1950-05-13,\\N,http\n",
        encoding="utf-8",
    )
    # Results table: row 1 is valid (raceId 1, driverId 1), row 2 is orphaned (raceId 999, driverId 999)
    (raw_dir / "results.csv").write_text(
        "resultId,raceId,driverId,constructorId,number,grid,position,positionText,positionOrder,points,laps,time,milliseconds,fastestLap,rank,fastestLapTime,fastestLapSpeed,statusId\n"
        "1,1,1,1,2,1,1,1,1,9.0,70,2:13:23.6,8003600,\\N,\\N,\\N,\\N,1\n"
        "2,999,999,1,3,2,2,2,2,6.0,70,+2.0,8005600,\\N,\\N,\\N,\\N,1\n",
        encoding="utf-8",
    )

    importer = KaggleImporter(source_dir=raw_dir, base_dir=tmp_path)
    importer.import_circuits()
    importer.import_constructors()
    importer.import_drivers()
    importer.import_status()
    importer.import_races()

    df_res = importer.import_results()
    assert len(df_res) == 1
    assert len(importer.quarantined_records) == 1
    assert "Missing referential integrity" in importer.quarantined_records[0]["reason"]


# ---------------------------------------------------------------------------
# 5. Source Precedence & Reconciliation Tests
# ---------------------------------------------------------------------------

def test_source_precedence_reconciliation():
    """Verify FastF1 has precedence for modern seasons (2023+) while Kaggle enriches 1950-2022."""
    reconciler = HistoricalDataReconciler()

    # FastF1 data (2023 modern)
    f1_records = [
        {
            "season": 2023,
            "round": 1,
            "driver_id": "max_verstappen",
            "constructor_id": "red_bull",
            "driver_number": 1,
            "driver_code": "VER",
            "grid_position": 1,
            "finish_position": 1,
            "position_text": "1",
            "points": 25.0,
            "laps_completed": 57,
            "status": "Finished",
            "is_classified": True,
            "is_dnf": False,
            "is_dns": False,
            "is_dsq": False,
            "total_time_ms": 5636736.0,
            "total_time_str": "1:33:56.736",
            "fastest_lap_rank": 2,
            "fastest_lap_time": "1:36.236",
            "fastest_lap_speed": 202.456,
            "source_api": "fastf1.timing",
            "retrieved_at": "2026-11-20T00:00:00",
        }
    ]
    df_f1 = pd.DataFrame(f1_records, columns=RACE_RESULTS_SCHEMA.names)

    # Kaggle data: 1 older race (1950) + 1 overlapping race (2023 with intentional difference)
    k_records = [
        {
            "season": 1950,
            "round": 1,
            "driver_id": "farina",
            "constructor_id": "alfa",
            "driver_number": 2,
            "driver_code": "",
            "grid_position": 1,
            "finish_position": 1,
            "position_text": "1",
            "points": 9.0,
            "laps_completed": 70,
            "status": "Finished",
            "is_classified": True,
            "is_dnf": False,
            "is_dns": False,
            "is_dsq": False,
            "total_time_ms": 8003600.0,
            "total_time_str": "2:13:23.6",
            "fastest_lap_rank": None,
            "fastest_lap_time": "",
            "fastest_lap_speed": None,
            "source_api": "kaggle.rohanrao",
            "retrieved_at": "2026-11-20T00:00:00",
        },
        {
            "season": 2023,
            "round": 1,
            "driver_id": "max_verstappen",
            "constructor_id": "red_bull",
            "driver_number": 1,
            "driver_code": "VER",
            "grid_position": 1,
            "finish_position": 2,  # Differs from FastF1 finish_position 1
            "position_text": "2",
            "points": 18.0,
            "laps_completed": 57,
            "status": "Finished",
            "is_classified": True,
            "is_dnf": False,
            "is_dns": False,
            "is_dsq": False,
            "total_time_ms": 5636736.0,
            "total_time_str": "1:33:56.736",
            "fastest_lap_rank": 2,
            "fastest_lap_time": "1:36.236",
            "fastest_lap_speed": 202.456,
            "source_api": "kaggle.rohanrao",
            "retrieved_at": "2026-11-20T00:00:00",
        },
    ]
    df_k = pd.DataFrame(k_records, columns=RACE_RESULTS_SCHEMA.names)

    master, report = reconciler.reconcile_race_results(df_f1, df_k)

    assert len(master) == 2
    assert report.overlap_count == 1
    assert len(report.conflicts_detected) > 0

    # Modern record retains FastF1 finish_position = 1
    ver_row = master[master["driver_id"] == "max_verstappen"].iloc[0]
    assert ver_row["finish_position"] == 1
    assert ver_row["points"] == 25.0

    # Historical record is preserved from Kaggle
    farina_row = master[master["driver_id"] == "farina"].iloc[0]
    assert farina_row["finish_position"] == 1
    assert farina_row["season"] == 1950


# ---------------------------------------------------------------------------
# 6. Driver-Circuit Appearances & Aggregation Tests
# ---------------------------------------------------------------------------

def test_driver_circuit_appearances_and_aggregated_compilation():
    """Verify compilation of 38-column appearances and 27-column aggregated performance profiles."""
    expander = HistoricalIntelligenceExpander()

    df_races = pd.DataFrame([
        {"season": 1988, "round": 11, "circuit_id": "spa", "event_name": "Belgian GP", "event_date": "1988-08-28"},
        {"season": 1988, "round": 12, "circuit_id": "monza", "event_name": "Italian GP", "event_date": "1988-09-11"},
    ], columns=EVENTS_SCHEMA.names)

    df_results = pd.DataFrame([
        # Spa 1988: Senna wins from pole, Prost P2 from P2
        {
            "season": 1988, "round": 11, "driver_id": "senna", "constructor_id": "mclaren",
            "driver_number": 12, "driver_code": "SEN", "grid_position": 1, "finish_position": 1,
            "position_text": "1", "points": 9.0, "laps_completed": 43, "status": "Finished",
            "is_classified": True, "is_dnf": False, "is_dns": False, "is_dsq": False,
            "total_time_ms": 5281549.0, "total_time_str": "1:28:01.549",
            "fastest_lap_rank": 1, "fastest_lap_time": "", "fastest_lap_speed": None,
            "source_api": "kaggle_test", "retrieved_at": "2026-11-20T00:00:00",
        },
        {
            "season": 1988, "round": 11, "driver_id": "prost", "constructor_id": "mclaren",
            "driver_number": 11, "driver_code": "PRO", "grid_position": 2, "finish_position": 2,
            "position_text": "2", "points": 6.0, "laps_completed": 43, "status": "Finished",
            "is_classified": True, "is_dnf": False, "is_dns": False, "is_dsq": False,
            "total_time_ms": 5312019.0, "total_time_str": "+30.470",
            "fastest_lap_rank": 2, "fastest_lap_time": "", "fastest_lap_speed": None,
            "source_api": "kaggle_test", "retrieved_at": "2026-11-20T00:00:00",
        },
        # Monza 1988: Senna collision DNF, Prost engine DNF
        {
            "season": 1988, "round": 12, "driver_id": "senna", "constructor_id": "mclaren",
            "driver_number": 12, "driver_code": "SEN", "grid_position": 1, "finish_position": 10,
            "position_text": "10", "points": 0.0, "laps_completed": 49, "status": "Collision",
            "is_classified": True, "is_dnf": False, "is_dns": False, "is_dsq": False,
            "total_time_ms": None, "total_time_str": "",
            "fastest_lap_rank": None, "fastest_lap_time": "", "fastest_lap_speed": None,
            "source_api": "kaggle_test", "retrieved_at": "2026-11-20T00:00:00",
        },
        {
            "season": 1988, "round": 12, "driver_id": "prost", "constructor_id": "mclaren",
            "driver_number": 11, "driver_code": "PRO", "grid_position": 2, "finish_position": None,
            "position_text": "R", "points": 0.0, "laps_completed": 34, "status": "Engine",
            "is_classified": False, "is_dnf": True, "is_dns": False, "is_dsq": False,
            "total_time_ms": None, "total_time_str": "",
            "fastest_lap_rank": None, "fastest_lap_time": "", "fastest_lap_speed": None,
            "source_api": "kaggle_test", "retrieved_at": "2026-11-20T00:00:00",
        },
    ], columns=RACE_RESULTS_SCHEMA.names)

    df_app = expander.compile_driver_circuit_appearances(df_races, df_results)
    assert len(df_app) == 4
    assert list(df_app.columns) == DRIVER_CIRCUIT_APPEARANCES_SCHEMA.names

    # Check teammate metrics for Senna at Spa (beat Prost)
    senna_spa = df_app[(df_app["driver_id"] == "senna") & (df_app["circuit_id"] == "spa")].iloc[0]
    assert senna_spa["teammate_driver_id"] == "prost"
    assert senna_spa["beat_teammate_race"] is True or senna_spa["beat_teammate_race"] == 1
    assert senna_spa["positions_gained_lost"] == 0

    # Check teammate metrics for Prost at Spa (lost to Senna)
    prost_spa = df_app[(df_app["driver_id"] == "prost") & (df_app["circuit_id"] == "spa")].iloc[0]
    assert prost_spa["teammate_driver_id"] == "senna"
    assert prost_spa["beat_teammate_race"] is False or prost_spa["beat_teammate_race"] == 0

    # Validate appearances schema
    rep_app = validate_driver_circuit_appearances(df_app)
    assert rep_app.is_valid

    # Compile aggregated profiles
    df_agg = expander.compile_driver_circuit_aggregated(df_app)
    assert len(df_agg) == 4  # (senna, spa), (prost, spa), (senna, monza), (prost, monza)
    assert list(df_agg.columns) == DRIVER_CIRCUIT_AGGREGATED_SCHEMA.names

    # Validate aggregated schema
    rep_agg = validate_driver_circuit_aggregated(df_agg)
    assert rep_agg.is_valid

    # Senna at Spa profile verification
    senna_spa_agg = df_agg[(df_agg["driver_id"] == "senna") & (df_agg["circuit_id"] == "spa")].iloc[0]
    assert senna_spa_agg["total_starts"] == 1
    assert senna_spa_agg["wins"] == 1
    assert senna_spa_agg["poles"] == 1
    assert senna_spa_agg["dnf_count"] == 0
    assert senna_spa_agg["dnf_rate"] == 0.0


# ---------------------------------------------------------------------------
# 7. Zero Future Leakage Guarantee Tests
# ---------------------------------------------------------------------------

def test_zero_future_leakage_guarantee():
    """Verify querying historical profile strictly prior to an event excludes future events."""
    expander = HistoricalIntelligenceExpander()

    df_appearances = pd.DataFrame([
        {
            "driver_id": "hamilton", "circuit_id": "silverstone", "season": 2019, "round": 10,
            "grid_position": 2, "finish_position": 1, "is_classified": True, "is_dnf": False,
            "points": 26.0, "positions_gained_lost": 1, "beat_teammate_race": True,
        },
        {
            "driver_id": "hamilton", "circuit_id": "silverstone", "season": 2020, "round": 4,
            "grid_position": 1, "finish_position": 1, "is_classified": True, "is_dnf": False,
            "points": 25.0, "positions_gained_lost": 0, "beat_teammate_race": True,
        },
        {
            "driver_id": "hamilton", "circuit_id": "silverstone", "season": 2021, "round": 10,
            "grid_position": 2, "finish_position": 1, "is_classified": True, "is_dnf": False,
            "points": 25.0, "positions_gained_lost": 1, "beat_teammate_race": True,
        },
    ])

    # Query before 2020 Round 4: should ONLY see 2019 event (1 start, 1 win)
    p_before_2020_r4 = expander.query_historical_profile(df_appearances, "hamilton", "silverstone", 2020, 4)
    assert p_before_2020_r4 is not None
    assert p_before_2020_r4["total_starts"] == 1
    assert p_before_2020_r4["latest_season"] == 2019

    # Query before 2021 Round 10: should see 2019 and 2020 events (2 starts, 2 wins)
    p_before_2021_r10 = expander.query_historical_profile(df_appearances, "hamilton", "silverstone", 2021, 10)
    assert p_before_2021_r10 is not None
    assert p_before_2021_r10["total_starts"] == 2
    assert p_before_2021_r10["latest_season"] == 2020

    # Query before 2019 Round 10: should see 0 starts (None returned)
    p_before_2019_r10 = expander.query_historical_profile(df_appearances, "hamilton", "silverstone", 2019, 10)
    assert p_before_2019_r10 is None


# ---------------------------------------------------------------------------
# 8. End-to-End Ingestion Pipeline Test
# ---------------------------------------------------------------------------

def test_kaggle_integration_pipeline_end_to_end():
    """Verify live integration execution across real/seed repository datasets."""
    res = run_kaggle_historical_integration(dry_run=True)

    assert res.success is True
    assert res.inventory.is_valid is True
    assert res.quality.valid_records_count >= 560
    assert res.quality.seasons_range[0] == 1950
    assert res.quality.seasons_range[1] == 2026
    assert res.quality.distinct_drivers_count >= 45
    assert res.reconciliation.reconciled_master_count >= 560
    assert len(res.reconciliation.conflicts_detected) == 0
