"""PACE: Comprehensive Unit & Integration Tests for OpenF1 Client & Race Intelligence.

Covers:
- OpenF1Client HTTP transport, request construction, throttling, retries, and backoff
- Rate limiting adherence (3 req/sec and 30 req/min rolling cap)
- HTTP 429 and Retry-After handling
- HTTP 404 "No results found" graceful empty list handling
- Local JSON disk cache hits, misses, and corruption recovery
- Canonical entity resolution (driver numbers, constructors, circuits)
- PyArrow schema validation across OpenF1 datasets
- Race intelligence features: neutralisation reconstruction, traffic exposure, weather analysis
- Selective telemetry ingestion and subsampling
- FastF1 vs OpenF1 cross-source reconciliation
"""

import json
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd
import requests

from src.data.openf1_client import (
    OpenF1Client,
    OpenF1Config,
    OpenF1Stats,
    OpenF1RateLimiter,
    OpenF1IngestionEngine,
    OpenF1Error,
    OpenF1APIError,
    OpenF1RateLimitError,
    OpenF1ConnectionError,
)
from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    CanonicalDriver,
    CanonicalConstructor,
    CanonicalCircuit,
)
from src.data.openf1_analytics import (
    OpenF1RaceIntelligence,
    NeutralisationEvent,
    DriverStintProfile,
    SessionWeatherProfile,
)
from src.data.openf1_reconciler import (
    OpenF1Reconciler,
    OpenF1ReconciliationReport,
    OpenF1Discrepancy,
)
from src.data.schemas import (
    OPENF1_MEETINGS_SCHEMA,
    OPENF1_SESSIONS_SCHEMA,
    OPENF1_DRIVERS_SCHEMA,
    OPENF1_INTERVALS_SCHEMA,
    OPENF1_STINTS_SCHEMA,
    OPENF1_RACE_CONTROL_SCHEMA,
    OPENF1_WEATHER_SCHEMA,
    OPENF1_OVERTAKES_SCHEMA,
    OPENF1_CAR_DATA_SCHEMA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary disk cache directory for OpenF1 tests."""
    c_dir = tmp_path / "openf1_cache"
    c_dir.mkdir(parents=True, exist_ok=True)
    return c_dir


@pytest.fixture
def mock_meetings_payload():
    """Sample OpenF1 meetings response."""
    return [
        {
            "meeting_key": 1141,
            "meeting_name": "Bahrain Grand Prix",
            "meeting_official_name": "FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2023",
            "location": "Sakhir",
            "country_key": 36,
            "country_code": "BRN",
            "country_name": "Bahrain",
            "circuit_key": 63,
            "circuit_short_name": "Sakhir",
            "year": 2023,
            "date_start": "2023-03-03T11:30:00+00:00",
            "date_end": "2023-03-05T17:00:00+00:00",
            "gmt_offset": "03:00:00",
        }
    ]


@pytest.fixture
def mock_drivers_payload():
    """Sample OpenF1 drivers response."""
    return [
        {
            "session_key": 7953,
            "meeting_key": 1141,
            "driver_number": 1,
            "broadcast_name": "M VERSTAPPEN",
            "full_name": "Max VERSTAPPEN",
            "name_acronym": "VER",
            "team_name": "Red Bull Racing",
            "team_colour": "3671C6",
            "first_name": "Max",
            "last_name": "Verstappen",
            "country_code": "NED",
        },
        {
            "session_key": 7953,
            "meeting_key": 1141,
            "driver_number": 16,
            "broadcast_name": "C LECLERC",
            "full_name": "Charles LECLERC",
            "name_acronym": "LEC",
            "team_name": "Ferrari",
            "team_colour": "F91536",
            "first_name": "Charles",
            "last_name": "Leclerc",
            "country_code": "MON",
        },
    ]


# ---------------------------------------------------------------------------
# 1. Client Transport, Rate Limiting & Resilience Tests
# ---------------------------------------------------------------------------

def test_openf1_cache_hit_and_miss(temp_cache_dir, mock_meetings_payload):
    """Verify first call hits network and writes cache; second call reads from disk cache."""
    mock_session = Mock(spec=requests.Session)
    mock_resp = Mock(status_code=200)
    mock_resp.json.return_value = mock_meetings_payload
    mock_resp.content = b'[{"meeting_key": 1141}]'
    mock_session.get.return_value = mock_resp

    config = OpenF1Config(cache_dir=temp_cache_dir, use_cache=True, rate_limit_pause=0.0)
    client = OpenF1Client(config=config, session=mock_session)

    # Call 1: Miss
    data1 = client.fetch_meetings(year=2023)
    assert len(data1) == 1
    assert client.stats.cache_misses == 1
    assert client.stats.cache_hits == 0
    assert mock_session.get.call_count == 1

    # Call 2: Hit
    data2 = client.fetch_meetings(year=2023)
    assert len(data2) == 1
    assert client.stats.cache_hits == 1
    assert mock_session.get.call_count == 1  # No second HTTP call


def test_openf1_rate_limiter_pauses():
    """Verify rate limiter enforces minimum interval spacing."""
    limiter = OpenF1RateLimiter(min_pause=0.05, max_per_minute=100)
    t0 = time.time()
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    t1 = time.time()
    assert (t1 - t0) >= 0.045


def test_openf1_http_429_retry_and_backoff(temp_cache_dir, mock_meetings_payload):
    """Verify HTTP 429 response triggers backoff and Retry-After header adherence."""
    mock_session = Mock(spec=requests.Session)
    resp_429 = Mock(status_code=429, headers={"Retry-After": "0"})
    resp_200 = Mock(status_code=200, content=b'[]')
    resp_200.json.return_value = mock_meetings_payload

    # First request 429, second request 200
    mock_session.get.side_effect = [resp_429, resp_200]

    config = OpenF1Config(cache_dir=temp_cache_dir, max_retries=2, backoff_base=0.01, rate_limit_pause=0.0)
    client = OpenF1Client(config=config, session=mock_session)

    data = client.fetch_meetings(year=2023)
    assert len(data) == 1
    assert client.stats.rate_limits_encountered == 1
    assert client.stats.retries_attempted == 1
    assert mock_session.get.call_count == 2


def test_openf1_http_404_no_results_found(temp_cache_dir):
    """Verify OpenF1 404 'No results found' is handled gracefully as empty list []."""
    mock_session = Mock(spec=requests.Session)
    resp_404 = Mock(status_code=404)
    resp_404.json.return_value = {"detail": "No results found."}
    resp_404.text = '{"detail":"No results found."}'
    mock_session.get.return_value = resp_404

    config = OpenF1Config(cache_dir=temp_cache_dir, rate_limit_pause=0.0)
    client = OpenF1Client(config=config, session=mock_session)

    res = client.fetch_pit(session_key=7953)
    assert res == []
    assert client.stats.errors_encountered == 0


def test_openf1_server_error_exhaustion(temp_cache_dir):
    """Verify persistent 500 server errors exhaust retries and raise OpenF1APIError."""
    mock_session = Mock(spec=requests.Session)
    resp_500 = Mock(status_code=500, text="Internal Server Error")
    mock_session.get.return_value = resp_500

    config = OpenF1Config(cache_dir=temp_cache_dir, max_retries=2, backoff_base=0.01, rate_limit_pause=0.0)
    client = OpenF1Client(config=config, session=mock_session)

    with pytest.raises(OpenF1APIError) as exc_info:
        client.fetch_meetings(year=2023)
    assert "OpenF1 API Error [500]" in str(exc_info.value)
    assert client.stats.retries_attempted == 2


def test_openf1_corrupted_cache_recovery(temp_cache_dir, mock_meetings_payload):
    """Verify client recovers gracefully from corrupted disk cache files."""
    cache_file = temp_cache_dir / "corrupted.json"
    with open(cache_file, "w") as f:
        f.write("{invalid_json: true, broken")

    mock_session = Mock(spec=requests.Session)
    resp_200 = Mock(status_code=200, content=b'[]')
    resp_200.json.return_value = mock_meetings_payload
    mock_session.get.return_value = resp_200

    config = OpenF1Config(cache_dir=temp_cache_dir, use_cache=True, rate_limit_pause=0.0)
    client = OpenF1Client(config=config, session=mock_session)

    # 1. Directly verify _read_cache purges corrupted file
    assert client._read_cache(cache_file) is None
    assert not cache_file.exists()

    # 2. Re-create corrupted file and verify recovery via request_endpoint rewriting valid JSON
    with open(cache_file, "w") as f:
        f.write("{invalid_json: true, broken")

    with patch.object(client, "_get_cache_path", return_value=cache_file):
        data = client.fetch_meetings(year=2023)
        assert len(data) == 1
        assert cache_file.exists()
        with open(cache_file, "r") as f:
            assert json.load(f) == mock_meetings_payload


# ---------------------------------------------------------------------------
# 2. Canonical Resolution & Ingestion Tests
# ---------------------------------------------------------------------------

def test_ingest_meetings_schema_and_canonical_circuit(mock_meetings_payload):
    """Verify conversion of OpenF1 meetings into OPENF1_MEETINGS_SCHEMA with canonical circuit ID."""
    mock_client = Mock(spec=OpenF1Client)
    mock_client.fetch_meetings.return_value = mock_meetings_payload

    registry = CanonicalEntityRegistry()
    registry.register_circuit(CanonicalCircuit(
        canonical_id="bahrain",
        name="Bahrain International Circuit",
        country="Bahrain",
        locality="Sakhir",
        ergast_id="bahrain",
        fastf1_id="bahrain",
        lap_length_km=5.412,
        typical_laps=57,
        default_transit_loss_s=23.5,
        aliases=["sakhir", "bahrain"],
    ))

    engine = OpenF1IngestionEngine(client=mock_client, registry=registry)
    df = engine.ingest_meetings_for_year(2023)

    assert len(df) == 1
    assert list(df.columns) == OPENF1_MEETINGS_SCHEMA.names
    assert df["meeting_key"].iloc[0] == 1141
    assert df["circuit_id"].iloc[0] == "bahrain"  # Resolved canonical ID
    assert df["source_api"].iloc[0] == "openf1"


def test_ingest_session_drivers_canonical_resolution(mock_drivers_payload):
    """Verify session drivers resolve canonical driver and constructor identities."""
    mock_client = Mock(spec=OpenF1Client)
    mock_client.fetch_drivers.return_value = mock_drivers_payload

    registry = CanonicalEntityRegistry()
    registry.register_driver(CanonicalDriver(
        canonical_id="max_verstappen",
        code="VER",
        permanent_number=1,
        full_name="Max Verstappen",
        given_name="Max",
        family_name="Verstappen",
        nationality="Dutch",
        ergast_id="max_verstappen",
        fastf1_code="VER",
    ))
    registry.register_driver(CanonicalDriver(
        canonical_id="leclerc",
        code="LEC",
        permanent_number=16,
        full_name="Charles Leclerc",
        given_name="Charles",
        family_name="Leclerc",
        nationality="Monegasque",
        ergast_id="leclerc",
        fastf1_code="LEC",
    ))
    registry.register_constructor(CanonicalConstructor(
        canonical_id="red_bull",
        name="Red Bull Racing",
        ergast_id="red_bull",
        fastf1_name="Red Bull",
        nationality="Austrian",
        aliases=["red bull racing"],
    ))
    registry.register_constructor(CanonicalConstructor(
        canonical_id="ferrari",
        name="Ferrari",
        ergast_id="ferrari",
        fastf1_name="Ferrari",
        nationality="Italian",
        aliases=["ferrari"],
    ))

    engine = OpenF1IngestionEngine(client=mock_client, registry=registry)
    df = engine.ingest_session_drivers(session_key=7953, year=2023)

    assert len(df) == 2
    assert list(df.columns) == OPENF1_DRIVERS_SCHEMA.names
    assert df.loc[df["driver_number"] == 1, "driver_id"].iloc[0] == "max_verstappen"
    assert df.loc[df["driver_number"] == 1, "constructor_id"].iloc[0] == "red_bull"
    assert df.loc[df["driver_number"] == 16, "driver_id"].iloc[0] == "leclerc"
    assert df.loc[df["driver_number"] == 16, "constructor_id"].iloc[0] == "ferrari"


def test_ingest_intervals_schema():
    """Verify intervals payload conversion into OPENF1_INTERVALS_SCHEMA."""
    payload = [
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:02:00Z", "driver_number": 1, "gap_to_leader": 0.0, "interval": 0.0},
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:02:00Z", "driver_number": 16, "gap_to_leader": 1.250, "interval": 1.250},
    ]
    mock_client = Mock(spec=OpenF1Client)
    mock_client.fetch_intervals.return_value = payload

    engine = OpenF1IngestionEngine(client=mock_client)
    driver_map = {
        1: {"driver_id": "max_verstappen"},
        16: {"driver_id": "leclerc"},
    }
    df = engine.ingest_session_intervals(session_key=7953, driver_map=driver_map)

    assert len(df) == 2
    assert list(df.columns) == OPENF1_INTERVALS_SCHEMA.names
    assert df.loc[df["driver_number"] == 16, "driver_id"].iloc[0] == "leclerc"
    assert df.loc[df["driver_number"] == 16, "gap_to_leader"].iloc[0] == 1.250


def test_ingest_stints_schema():
    """Verify stints payload conversion into OPENF1_STINTS_SCHEMA."""
    payload = [
        {"session_key": 7953, "meeting_key": 1141, "stint_number": 1, "driver_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": 14, "tyre_age_at_start": 0},
        {"session_key": 7953, "meeting_key": 1141, "stint_number": 2, "driver_number": 1, "compound": "HARD", "lap_start": 15, "lap_end": 36, "tyre_age_at_start": 0},
    ]
    mock_client = Mock(spec=OpenF1Client)
    mock_client.fetch_stints.return_value = payload

    engine = OpenF1IngestionEngine(client=mock_client)
    driver_map = {1: {"driver_id": "max_verstappen"}}
    df = engine.ingest_session_stints(session_key=7953, driver_map=driver_map)

    assert len(df) == 2
    assert list(df.columns) == OPENF1_STINTS_SCHEMA.names
    assert df["stint_length"].tolist() == [14, 22]  # (14 - 1 + 1) and (36 - 15 + 1)
    assert df["compound"].tolist() == ["SOFT", "HARD"]


def test_selective_car_data_telemetry_subsampling():
    """Verify car_data selective ingestion applies speed threshold and subsampling."""
    payload = [
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:10:01Z", "driver_number": 1, "speed": 310, "rpm": 11500, "n_gear": 8, "throttle": 100, "brake": 0, "drs": 1},
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:10:02Z", "driver_number": 1, "speed": 315, "rpm": 11600, "n_gear": 8, "throttle": 100, "brake": 0, "drs": 1},
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:10:03Z", "driver_number": 1, "speed": 320, "rpm": 11700, "n_gear": 8, "throttle": 100, "brake": 0, "drs": 1},
        {"session_key": 7953, "meeting_key": 1141, "date": "2023-03-05T15:10:04Z", "driver_number": 1, "speed": 325, "rpm": 11800, "n_gear": 8, "throttle": 100, "brake": 0, "drs": 1},
    ]
    mock_client = Mock(spec=OpenF1Client)
    mock_client.fetch_car_data.return_value = payload

    engine = OpenF1IngestionEngine(client=mock_client)
    driver_map = {1: {"driver_id": "max_verstappen"}}

    # sample_step=2 should take records 0 and 2
    df = engine.ingest_selective_car_data(session_key=7953, driver_numbers=[1], speed_min=300, sample_step=2, driver_map=driver_map)

    assert len(df) == 2
    assert list(df.columns) == OPENF1_CAR_DATA_SCHEMA.names
    assert df["speed"].tolist() == [310, 320]
    assert df["drs"].tolist() == [1, 1]


# ---------------------------------------------------------------------------
# 3. Race Intelligence Analytics Tests
# ---------------------------------------------------------------------------

def test_neutralisation_period_reconstruction():
    """Verify reconstruction of continuous Safety Car and VSC periods from race control messages."""
    df_rc = pd.DataFrame([
        {"date": "2023-03-05T15:20:00Z", "lap_number": 10, "category": "SafetyCar", "flag": "CLEAR", "message": "SAFETY CAR DEPLOYED"},
        {"date": "2023-03-05T15:26:00Z", "lap_number": 13, "category": "SafetyCar", "flag": "GREEN", "message": "SAFETY CAR IN THIS LAP"},
        {"date": "2023-03-05T15:45:00Z", "lap_number": 28, "category": "Flag", "flag": "CLEAR", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "2023-03-05T15:47:30Z", "lap_number": 29, "category": "Flag", "flag": "GREEN", "message": "VSC ENDING - TRACK CLEAR"},
    ])

    intel = OpenF1RaceIntelligence(df_race_control=df_rc)
    events = intel.extract_neutralisation_periods()

    assert len(events) == 2
    sc = events[0]
    assert sc.event_type == "SAFETY_CAR"
    assert sc.start_lap == 10
    assert sc.end_lap == 13
    assert sc.laps_affected == 4  # Laps 10, 11, 12, 13
    assert sc.duration_seconds == 360.0  # 6 minutes

    vsc = events[1]
    assert vsc.event_type == "VSC"
    assert vsc.start_lap == 28
    assert vsc.end_lap == 29
    assert vsc.duration_seconds == 150.0  # 2.5 minutes


def test_traffic_exposure_calculation():
    """Verify traffic exposure percentage (% of intervals <= 1.0s) calculation."""
    df_inv = pd.DataFrame([
        {"driver_id": "norris", "interval": 0.45},   # In traffic
        {"driver_id": "norris", "interval": 0.80},   # In traffic
        {"driver_id": "norris", "interval": 1.50},   # Clean air
        {"driver_id": "norris", "interval": 3.20},   # Clean air
        {"driver_id": "verstappen", "interval": 4.5}, # Clean air
        {"driver_id": "verstappen", "interval": 6.2}, # Clean air
    ])
    intel = OpenF1RaceIntelligence(df_intervals=df_inv)
    exposure = intel.calculate_traffic_exposure(drs_threshold_s=1.0)

    assert exposure["norris"] == 50.0  # 2 out of 4 intervals in DRS window
    assert exposure["verstappen"] == 0.0  # Clean air


def test_session_weather_profile_analytics():
    """Verify calculation of weather statistics, gradients, and rain detection."""
    df_w = pd.DataFrame([
        {"session_key": 7953, "date": "2023-03-05T15:00:00Z", "track_temperature": 32.0, "air_temperature": 24.0, "humidity": 45.0, "wind_speed": 3.2, "rainfall": 0},
        {"session_key": 7953, "date": "2023-03-05T15:30:00Z", "track_temperature": 30.0, "air_temperature": 23.5, "humidity": 48.0, "wind_speed": 4.0, "rainfall": 0},
        {"session_key": 7953, "date": "2023-03-05T16:00:00Z", "track_temperature": 28.0, "air_temperature": 23.0, "humidity": 52.0, "wind_speed": 2.8, "rainfall": 0},
    ])
    intel = OpenF1RaceIntelligence(df_weather=df_w)
    wp = intel.analyze_session_weather(session_key=7953)

    assert wp.session_key == 7953
    assert wp.mean_track_temp == 30.0
    assert wp.min_track_temp == 28.0
    assert wp.max_track_temp == 32.0
    assert wp.track_temp_delta == 4.0
    assert wp.rainfall_detected is False


def test_driver_race_intelligence_summary():
    """Verify aggregated driver summary combining stints, traffic exposure, and overtakes."""
    df_st = pd.DataFrame([
        {"driver_id": "norris", "driver_number": 4, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 20, "stint_length": 20, "tyre_age_at_start": 0},
        {"driver_id": "norris", "driver_number": 4, "stint_number": 2, "compound": "HARD", "lap_start": 21, "lap_end": 57, "stint_length": 37, "tyre_age_at_start": 0},
    ])
    df_inv = pd.DataFrame([
        {"driver_id": "norris", "interval": 0.5, "gap_to_leader": 12.0},
        {"driver_id": "norris", "interval": 1.2, "gap_to_leader": 14.0},
    ])
    df_ov = pd.DataFrame([
        {"overtaking_driver_id": "norris", "overtaken_driver_id": "hulkenberg"},
        {"overtaking_driver_id": "norris", "overtaken_driver_id": "tsunoda"},
        {"overtaking_driver_id": "hamilton", "overtaken_driver_id": "norris"},
    ])
    df_drivers = pd.DataFrame([
        {"driver_id": "norris", "driver_number": 4, "constructor_id": "mclaren"},
    ])

    intel = OpenF1RaceIntelligence(
        df_stints=df_st,
        df_intervals=df_inv,
        df_overtakes=df_ov,
        df_drivers=df_drivers,
    )
    summary_df = intel.compile_driver_race_intelligence()

    assert len(summary_df) == 1
    r = summary_df.iloc[0]
    assert r["driver_id"] == "norris"
    assert r["constructor_id"] == "mclaren"
    assert r["stints_count"] == 2
    assert r["compounds_used"] == "MEDIUM, HARD"
    assert r["traffic_exposure_pct"] == 50.0
    assert r["overtakes_made"] == 2
    assert r["times_overtaken"] == 1
    assert r["net_overtakes"] == 1


# ---------------------------------------------------------------------------
# 4. Cross-Source Reconciliation Tests
# ---------------------------------------------------------------------------

def test_openf1_fastf1_laps_reconciliation():
    """Verify lap time reconciliation between OpenF1 and FastF1 within tolerance."""
    df_openf1 = pd.DataFrame([
        {"driver_id": "max_verstappen", "lap_number": 1, "lap_time_ms": 94500.0},
        {"driver_id": "max_verstappen", "lap_number": 2, "lap_time_ms": 93205.0},
        {"driver_id": "leclerc", "lap_number": 1, "lap_time_ms": 95100.0},
    ])
    df_fastf1 = pd.DataFrame([
        {"driver_id": "max_verstappen", "lap_number": 1, "lap_time_ms": 94505.0},  # Delta = 5ms <= 15ms tolerance
        {"driver_id": "max_verstappen", "lap_number": 2, "lap_time_ms": 93200.0},  # Delta = 5ms <= 15ms tolerance
        {"driver_id": "leclerc", "lap_number": 1, "lap_time_ms": 95150.0},         # Delta = 50ms > 15ms (conflict)
    ])

    reconciler = OpenF1Reconciler()
    rep = reconciler.reconcile_laps(df_openf1, df_fastf1, session_key=7953, timing_tolerance_ms=15.0)

    assert rep.overlap_count == 3
    assert rep.identical_count == 2
    assert rep.conflict_count == 1
    assert rep.conflicts_detected[0].entity_id == "leclerc"
    assert rep.conflicts_detected[0].authoritative_source == "FastF1"


def test_openf1_fastf1_stints_reconciliation():
    """Verify tyre stint compound reconciliation between OpenF1 and FastF1."""
    df_openf1 = pd.DataFrame([
        {"driver_id": "max_verstappen", "stint_number": 1, "compound": "SOFT", "lap_start": 1},
        {"driver_id": "max_verstappen", "stint_number": 2, "compound": "HARD", "lap_start": 15},
    ])
    df_fastf1 = pd.DataFrame([
        {"driver_id": "max_verstappen", "stint_number": 1, "compound": "SOFT", "lap_start": 1},
        {"driver_id": "max_verstappen", "stint_number": 2, "compound": "MEDIUM", "lap_start": 15},  # Conflict
    ])

    reconciler = OpenF1Reconciler()
    rep = reconciler.reconcile_stints(df_openf1, df_fastf1, session_key=7953)

    assert rep.overlap_count == 2
    assert rep.identical_count == 1
    assert rep.conflict_count == 1
    assert rep.conflicts_detected[0].field_name == "compound"
