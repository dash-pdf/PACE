"""PACE: Comprehensive Unit & Integration Tests for Jolpica Client & Championship Analytics.

Covers:
- JolpicaClient HTTP transport, retries, exponential backoff, and timeouts
- HTTP 429 rate limiting & Retry-After header adherence
- Local JSON disk cache hits, misses, and corruption recovery
- Auto-pagination across multi-page Ergast result sets
- Ingestion engine DataFrame schema conformance
- Canonical entity resolution and unresolved entity auditing
- Championship analytics as-of queries (zero temporal leakage)
- Points progression and gap calculation
- Mathematical championship elimination thresholds
- Monte Carlo title probability simulation contract
- Cross-source reconciliation reporting
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import pytest
import pandas as pd
import requests

from src.data.jolpica_client import (
    JolpicaClient,
    JolpicaConfig,
    JolpicaStats,
    JolpicaIngestionEngine,
    JolpicaAPIError,
    JolpicaRateLimitError,
    JolpicaConnectionError,
)
from src.data.canonical_registry import CanonicalEntityRegistry, CanonicalDriver, CanonicalConstructor
from src.data.championship_analytics import ChampionshipAnalytics, F1ScoringRules, MathematicalEliminationReport
from src.data.jolpica_reconciler import JolpicaReconciler, JolpicaConflict
from src.data.schemas import (
    DRIVER_STANDINGS_SCHEMA,
    CONSTRUCTOR_STANDINGS_SCHEMA,
    RACE_RESULTS_SCHEMA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for Jolpica client testing."""
    cache_dir = tmp_path / "jolpica_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_driver_standings_payload():
    """Sample Ergast-compatible driver standings JSON payload."""
    return {
        "MRData": {
            "xmlns": "http://ergast.com/mrd/1.5",
            "series": "f1",
            "url": "https://api.jolpi.ca/ergast/f1/2024/1/driverStandings.json",
            "limit": "100",
            "offset": "0",
            "total": "2",
            "StandingsTable": {
                "season": "2024",
                "round": "1",
                "StandingsLists": [
                    {
                        "season": "2024",
                        "round": "1",
                        "DriverStandings": [
                            {
                                "position": "1",
                                "positionText": "1",
                                "points": "26",
                                "wins": "1",
                                "Driver": {
                                    "driverId": "max_verstappen",
                                    "code": "VER",
                                    "givenName": "Max",
                                    "familyName": "Verstappen",
                                },
                                "Constructors": [{"constructorId": "red_bull", "name": "Red Bull"}],
                            },
                            {
                                "position": "2",
                                "positionText": "2",
                                "points": "18",
                                "wins": "0",
                                "Driver": {
                                    "driverId": "perez",
                                    "code": "PER",
                                    "givenName": "Sergio",
                                    "familyName": "Perez",
                                },
                                "Constructors": [{"constructorId": "red_bull", "name": "Red Bull"}],
                            },
                        ],
                    }
                ],
            },
        }
    }


# ---------------------------------------------------------------------------
# 1. HTTP Transport, Caching, and Error Recovery Tests
# ---------------------------------------------------------------------------

def test_jolpica_cache_hit_and_miss(temp_cache_dir, mock_driver_standings_payload):
    """Verify that first request misses cache and persists response, and second request hits cache."""
    mock_session = Mock(spec=requests.Session)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_driver_standings_payload
    mock_session.get.return_value = mock_response

    config = JolpicaConfig(cache_dir=temp_cache_dir, use_cache=True, rate_limit_pause=0.0)
    client = JolpicaClient(config=config, session=mock_session)

    # First call: Cache Miss -> HTTP Get
    data1 = client.request_endpoint("2024/1/driverStandings")
    assert client.stats.cache_misses == 1
    assert client.stats.cache_hits == 0
    assert mock_session.get.call_count == 1
    assert data1["MRData"]["total"] == "2"

    # Second call: Cache Hit -> Zero HTTP Get
    data2 = client.request_endpoint("2024/1/driverStandings")
    assert client.stats.cache_hits == 1
    assert mock_session.get.call_count == 1
    assert data2 == data1


def test_jolpica_http_429_retry_and_backoff(temp_cache_dir, mock_driver_standings_payload):
    """Verify HTTP 429 response triggers backoff and succeeds on subsequent attempt."""
    mock_session = Mock(spec=requests.Session)
    
    resp_429 = Mock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0"}

    resp_200 = Mock()
    resp_200.status_code = 200
    resp_200.json.return_value = mock_driver_standings_payload

    # First attempt 429, second attempt 200
    mock_session.get.side_effect = [resp_429, resp_200]

    config = JolpicaConfig(cache_dir=temp_cache_dir, max_retries=2, backoff_base=0.01, rate_limit_pause=0.0)
    client = JolpicaClient(config=config, session=mock_session)

    data = client.request_endpoint("2024/1/driverStandings")
    assert data["MRData"]["total"] == "2"
    assert client.stats.rate_limits_encountered == 1
    assert client.stats.retries_attempted == 1
    assert mock_session.get.call_count == 2


def test_jolpica_server_error_exhaustion(temp_cache_dir):
    """Verify persistent 500 server errors exhaust retries and raise JolpicaAPIError."""
    mock_session = Mock(spec=requests.Session)
    resp_500 = Mock()
    resp_500.status_code = 500
    mock_session.get.return_value = resp_500

    config = JolpicaConfig(cache_dir=temp_cache_dir, max_retries=2, backoff_base=0.01, rate_limit_pause=0.0)
    client = JolpicaClient(config=config, session=mock_session)

    with pytest.raises(JolpicaAPIError) as exc_info:
        client.request_endpoint("2024/1/driverStandings")

    assert "HTTP 500" in str(exc_info.value)
    assert client.stats.retries_attempted == 2


def test_jolpica_malformed_json_handling(temp_cache_dir):
    """Verify non-JSON payload raises JolpicaAPIError."""
    mock_session = Mock(spec=requests.Session)
    resp = Mock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("No JSON object could be decoded")
    mock_session.get.return_value = resp

    config = JolpicaConfig(cache_dir=temp_cache_dir, rate_limit_pause=0.0)
    client = JolpicaClient(config=config, session=mock_session)

    with pytest.raises(JolpicaAPIError) as exc_info:
        client.request_endpoint("2024/1/driverStandings")
    assert "Malformed JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Pagination Engine Tests
# ---------------------------------------------------------------------------

def test_jolpica_multi_page_pagination(temp_cache_dir):
    """Verify paginate_endpoint correctly accumulates multiple pages using limit and offset."""
    page1 = {
        "MRData": {
            "total": "3",
            "limit": "2",
            "offset": "0",
            "StandingsTable": {
                "StandingsLists": [{"DriverStandings": [{"position": "1", "points": "25"}, {"position": "2", "points": "18"}]}]
            },
        }
    }
    page2 = {
        "MRData": {
            "total": "3",
            "limit": "2",
            "offset": "2",
            "StandingsTable": {
                "StandingsLists": [{"DriverStandings": [{"position": "3", "points": "15"}]}]
            },
        }
    }

    mock_session = Mock(spec=requests.Session)
    mock_resp1 = Mock(status_code=200)
    mock_resp1.json.return_value = page1
    mock_resp2 = Mock(status_code=200)
    mock_resp2.json.return_value = page2
    mock_session.get.side_effect = [mock_resp1, mock_resp2]

    config = JolpicaConfig(cache_dir=temp_cache_dir, use_cache=False, rate_limit_pause=0.0)
    client = JolpicaClient(config=config, session=mock_session)

    items = client.paginate_endpoint(
        "2024/1/driverStandings",
        extract_key=("StandingsTable", "StandingsLists", "DriverStandings"),
        page_size=2,
    )

    assert len(items) == 3
    assert items[0]["position"] == "1"
    assert items[2]["position"] == "3"
    assert mock_session.get.call_count == 2


# ---------------------------------------------------------------------------
# 3. Ingestion Engine & Canonical Resolution Tests
# ---------------------------------------------------------------------------

def test_ingest_driver_standings_schema_and_canonical_mapping(temp_cache_dir, mock_driver_standings_payload):
    """Verify conversion of raw Jolpica driver standings to DRIVER_STANDINGS_SCHEMA with canonical IDs."""
    mock_client = Mock(spec=JolpicaClient)
    mock_client.fetch_driver_standings.return_value = (
        mock_driver_standings_payload["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    )

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
        canonical_id="perez",
        code="PER",
        permanent_number=11,
        full_name="Sergio Perez",
        given_name="Sergio",
        family_name="Perez",
        nationality="Mexican",
        ergast_id="perez",
        fastf1_code="PER",
    ))

    engine = JolpicaIngestionEngine(client=mock_client, registry=registry)
    df = engine.ingest_driver_standings_for_round(2024, 1)

    assert len(df) == 2
    assert list(df.columns) == DRIVER_STANDINGS_SCHEMA.names
    assert df["driver_id"].tolist() == ["max_verstappen", "perez"]
    assert df["points"].tolist() == [26.0, 18.0]
    assert df["position"].tolist() == [1, 2]
    assert df["wins"].tolist() == [1, 0]
    assert df["source_api"].iloc[0] == "jolpica.ergast"


def test_unknown_driver_recorded_for_audit(temp_cache_dir):
    """Verify drivers not in registry are recorded in unresolved audit set without crashing."""
    payload = [
        {
            "position": "1",
            "points": "25",
            "wins": "1",
            "Driver": {"driverId": "future_unknown_rookie", "givenName": "Jane", "familyName": "Doe"},
        }
    ]
    mock_client = Mock(spec=JolpicaClient)
    mock_client.fetch_driver_standings.return_value = payload

    registry = CanonicalEntityRegistry()
    engine = JolpicaIngestionEngine(client=mock_client, registry=registry)

    df = engine.ingest_driver_standings_for_round(2026, 1)
    assert len(df) == 1
    assert df["driver_id"].iloc[0] == "future_unknown_rookie"
    assert "future_unknown_rookie" in engine.unresolved["drivers"]


# ---------------------------------------------------------------------------
# 4. Championship Analytics & As-Of Temporal Integrity Tests
# ---------------------------------------------------------------------------

def test_championship_analytics_as_of_temporal_isolation():
    """Verify standings queries at round R strictly exclude data from round R+1 (Zero Leakage)."""
    df_stand = pd.DataFrame([
        # Round 1
        {"season": 2024, "round": 1, "driver_id": "max_verstappen", "points": 26.0, "position": 1, "wins": 1, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 1, "driver_id": "norris", "points": 18.0, "position": 2, "wins": 0, "position_text": "2", "source_api": "test", "retrieved_at": ""},
        # Round 2
        {"season": 2024, "round": 2, "driver_id": "max_verstappen", "points": 51.0, "position": 1, "wins": 2, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 2, "driver_id": "norris", "points": 43.0, "position": 2, "wins": 0, "position_text": "2", "source_api": "test", "retrieved_at": ""},
    ])

    analytics = ChampionshipAnalytics(df_driver_standings=df_stand)

    # As of Round 1
    r1_stand = analytics.get_driver_standings_as_of(season=2024, as_of_round=1)
    assert len(r1_stand) == 2
    assert r1_stand.loc[r1_stand["driver_id"] == "max_verstappen", "points"].iloc[0] == 26.0
    assert r1_stand.loc[r1_stand["driver_id"] == "norris", "gap_to_leader"].iloc[0] == 8.0

    # Before event Round 2 -> Evaluates Round 1
    pre_r2_stand = analytics.get_driver_standings_as_of(season=2024, as_of_round=2, before_event=True)
    assert pre_r2_stand.loc[pre_r2_stand["driver_id"] == "max_verstappen", "points"].iloc[0] == 26.0

    # As of Round 2
    r2_stand = analytics.get_driver_standings_as_of(season=2024, as_of_round=2)
    assert r2_stand.loc[r2_stand["driver_id"] == "max_verstappen", "points"].iloc[0] == 51.0


def test_mathematical_elimination_calculation():
    """Verify mathematical elimination report correctly identifies eliminated drivers."""
    # Season with 24 rounds. As of Round 22, 2 rounds remaining.
    # Max points in 2 rounds (no sprint, modern): 2 * 25 = 50 pts
    # Leader has 350 pts. Threshold = 350 - 50 = 300 pts.
    df_stand = pd.DataFrame([
        {"season": 2024, "round": 22, "driver_id": "verstappen", "points": 350.0, "position": 1, "wins": 8, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 22, "driver_id": "norris", "points": 310.0, "position": 2, "wins": 3, "position_text": "2", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 22, "driver_id": "leclerc", "points": 290.0, "position": 3, "wins": 2, "position_text": "3", "source_api": "test", "retrieved_at": ""},
    ])

    analytics = ChampionshipAnalytics(df_driver_standings=df_stand)
    report = analytics.compute_elimination_status(season=2024, as_of_round=22, total_rounds=24, remaining_sprints=0)

    assert report.leader_id == "verstappen"
    assert report.leader_points == 350.0
    assert report.max_remaining_points == 52.0  # 2 races * 26 (25 win + 1 FL in 2024) = 52
    # Norris has 310 + 52 = 362 >= 350 -> CONTENDER
    # Leclerc has 290 + 52 = 342 < 350 -> ELIMINATED
    assert "norris" in report.active_contenders
    assert "leclerc" in report.eliminated_drivers
    assert report.active_contenders_count == 2
    assert report.eliminated_count == 1


def test_mathematical_elimination_2024_round_20_piastri_contender():
    """Verify that in 2024 Round 20 (Mexico City GP), Piastri remains an active CONTENDER.
    
    4 Grands Prix remaining (4 * 26 = 104 pts).
    2 Sprints remaining: São Paulo R21 and Qatar R23 (2 * 8 = 16 pts).
    Total maximum remaining points = 120.0 pts.
    Verstappen has 362.0 pts.
    Piastri has 251.0 pts: 251 + 120 = 371.0 >= 362.0 -> CONTENDER.
    Sainz has 240.0 pts: 240 + 120 = 360.0 < 362.0 -> ELIMINATED this round.
    """
    df_stand_r20 = pd.DataFrame([
        {"season": 2024, "round": 20, "driver_id": "max_verstappen", "points": 362.0, "position": 1, "wins": 7, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "norris", "points": 315.0, "position": 2, "wins": 3, "position_text": "2", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "leclerc", "points": 291.0, "position": 3, "wins": 3, "position_text": "3", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "piastri", "points": 251.0, "position": 4, "wins": 2, "position_text": "4", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "sainz", "points": 240.0, "position": 5, "wins": 2, "position_text": "5", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "hamilton", "points": 189.0, "position": 6, "wins": 2, "position_text": "6", "source_api": "test", "retrieved_at": ""},
    ])
    # Standings after Round 19 for prior-elimination delta tracking
    df_stand_r19 = pd.DataFrame([
        {"season": 2024, "round": 19, "driver_id": "max_verstappen", "points": 354.0, "position": 1, "wins": 7, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 19, "driver_id": "norris", "points": 297.0, "position": 2, "wins": 3, "position_text": "2", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 19, "driver_id": "leclerc", "points": 275.0, "position": 3, "wins": 3, "position_text": "3", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 19, "driver_id": "piastri", "points": 247.0, "position": 4, "wins": 2, "position_text": "4", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 19, "driver_id": "sainz", "points": 215.0, "position": 5, "wins": 1, "position_text": "5", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 19, "driver_id": "hamilton", "points": 177.0, "position": 6, "wins": 2, "position_text": "6", "source_api": "test", "retrieved_at": ""},
    ])
    df_sprints = pd.DataFrame([
        {"season": 2024, "round": 5},
        {"season": 2024, "round": 6},
        {"season": 2024, "round": 11},
        {"season": 2024, "round": 19},
        {"season": 2024, "round": 21},  # São Paulo
        {"season": 2024, "round": 23},  # Qatar
    ])

    analytics = ChampionshipAnalytics(
        df_driver_standings=pd.concat([df_stand_r19, df_stand_r20], ignore_index=True),
        df_sprints=df_sprints,
    )

    report = analytics.compute_elimination_status(season=2024, as_of_round=20, total_rounds=24)

    assert report.remaining_rounds == 4
    assert report.remaining_sprints == 2
    assert report.max_remaining_points == 120.0
    assert report.leader_id == "max_verstappen"
    assert report.leader_points == 362.0

    # Contenders verification
    assert report.active_contenders_count == 4
    assert "max_verstappen" in report.active_contenders
    assert "norris" in report.active_contenders
    assert "leclerc" in report.active_contenders
    assert "piastri" in report.active_contenders  # CRITICAL: Piastri must be a contender!

    # Elimination verification
    assert "sainz" in report.eliminated_drivers
    assert "hamilton" in report.eliminated_drivers
    assert report.eliminated_this_round == ["sainz"]  # Sainz newly eliminated at R20; Hamilton was eliminated at R19


def test_constructor_elimination_2024_round_20():
    """Verify constructor elimination after 2024 Round 20 with Sprint points included.
    
    4 Grands Prix remaining (4 * 44 = 176 pts).
    2 Sprints remaining (2 * 15 = 30 pts).
    Total maximum constructor points = 206.0 pts.
    McLaren: 566 pts.
    Mercedes: 366 pts: 366 + 206 = 572.0 >= 566.0 -> CONTENDER.
    Aston Martin: 86 pts: 86 + 206 = 292.0 < 566.0 -> ELIMINATED.
    """
    df_c_r20 = pd.DataFrame([
        {"season": 2024, "round": 20, "constructor_id": "mclaren", "points": 566.0, "position": 1, "wins": 5, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "constructor_id": "ferrari", "points": 537.0, "position": 2, "wins": 5, "position_text": "2", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "constructor_id": "red_bull", "points": 512.0, "position": 3, "wins": 7, "position_text": "3", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "constructor_id": "mercedes", "points": 366.0, "position": 4, "wins": 3, "position_text": "4", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "constructor_id": "aston_martin", "points": 86.0, "position": 5, "wins": 0, "position_text": "5", "source_api": "test", "retrieved_at": ""},
    ])
    df_sprints = pd.DataFrame([
        {"season": 2024, "round": 5},
        {"season": 2024, "round": 6},
        {"season": 2024, "round": 11},
        {"season": 2024, "round": 19},
        {"season": 2024, "round": 21},
        {"season": 2024, "round": 23},
    ])

    analytics = ChampionshipAnalytics(
        df_constructor_standings=df_c_r20,
        df_sprints=df_sprints,
    )

    c_rep = analytics.compute_constructor_elimination_status(season=2024, as_of_round=20, total_rounds=24)

    assert c_rep.remaining_rounds == 4
    assert c_rep.remaining_sprints == 2
    assert c_rep.max_remaining_points == 206.0
    assert c_rep.leader_id == "mclaren"
    assert c_rep.active_contenders_count == 4
    assert "mercedes" in c_rep.active_contenders  # Mercedes active because 366 + 206 = 572 > 566
    assert "aston_martin" in c_rep.eliminated_drivers


def test_scoring_rules_era_coverage():
    """Verify exact regulatory scoring distributions across all historical and modern eras."""
    # 1. 1950-1959 era: 8-6-4-3-2 + 1 FL
    assert F1ScoringRules.get_gp_points_distribution(1955) == [8.0, 6.0, 4.0, 3.0, 2.0]
    assert F1ScoringRules.has_fastest_lap_point(1955) is True
    assert F1ScoringRules.fastest_lap_requires_top_10(1955) is False
    assert F1ScoringRules.get_max_points_per_event(1955) == 9.0
    assert F1ScoringRules.get_max_constructor_points_per_event(1955) == 0.0  # WCC started 1958

    # 2. 1960: 8-6-4-3-2-1, no FL
    assert F1ScoringRules.get_gp_points_distribution(1960) == [8.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    assert F1ScoringRules.has_fastest_lap_point(1960) is False
    assert F1ScoringRules.get_max_points_per_event(1960) == 8.0

    # 3. 1961-1978: 9-6-4-3-2-1, only 1 car scored for WCC
    assert F1ScoringRules.get_gp_points_distribution(1975) == [9.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    assert F1ScoringRules.has_fastest_lap_point(1975) is False
    assert F1ScoringRules.get_max_points_per_event(1975) == 9.0
    assert F1ScoringRules.get_max_constructor_points_per_event(1975) == 9.0

    # 4. 1979-1990: 9-6-4-3-2-1, 2 cars scored for WCC
    assert F1ScoringRules.get_max_constructor_points_per_event(1985) == 15.0  # 9 + 6

    # 5. 1991-2002: 10-6-4-3-2-1
    assert F1ScoringRules.get_gp_points_distribution(1998) == [10.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    assert F1ScoringRules.get_max_points_per_event(1998) == 10.0
    assert F1ScoringRules.get_max_constructor_points_per_event(1998) == 16.0  # 10 + 6

    # 6. 2003-2009: 10-8-6-5-4-3-2-1
    assert F1ScoringRules.get_gp_points_distribution(2005) == [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert F1ScoringRules.get_max_points_per_event(2005) == 10.0
    assert F1ScoringRules.get_max_constructor_points_per_event(2005) == 18.0  # 10 + 8

    # 7. 2010-2018: 25-18-15-12-10-8-6-4-2-1, no FL
    assert F1ScoringRules.get_gp_points_distribution(2015) == [25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
    assert F1ScoringRules.has_fastest_lap_point(2015) is False
    assert F1ScoringRules.get_max_points_per_event(2015) == 25.0
    assert F1ScoringRules.get_max_constructor_points_per_event(2015) == 43.0  # 25 + 18

    # 8. 2019-2024: 25-18-15-12-10-8-6-4-2-1 + 1 FL (top 10 finish)
    assert F1ScoringRules.has_fastest_lap_point(2024) is True
    assert F1ScoringRules.fastest_lap_requires_top_10(2024) is True
    assert F1ScoringRules.get_max_points_per_event(2024, has_sprint=False) == 26.0
    assert F1ScoringRules.get_max_points_per_event(2024, has_sprint=True) == 34.0  # 26 + 8
    assert F1ScoringRules.get_max_constructor_points_per_event(2024, has_sprint=False) == 44.0  # 25 + 18 + 1
    assert F1ScoringRules.get_max_constructor_points_per_event(2024, has_sprint=True) == 59.0   # 44 + 15

    # 9. 2025+: Fastest lap officially abolished by FIA starting in 2025
    assert F1ScoringRules.has_fastest_lap_point(2025) is False
    assert F1ScoringRules.has_fastest_lap_point(2026) is False
    assert F1ScoringRules.get_max_points_per_event(2025, has_sprint=False) == 25.0
    assert F1ScoringRules.get_max_points_per_event(2025, has_sprint=True) == 33.0  # 25 + 8
    assert F1ScoringRules.get_max_constructor_points_per_event(2025, has_sprint=False) == 43.0
    assert F1ScoringRules.get_max_constructor_points_per_event(2025, has_sprint=True) == 58.0   # 43 + 15

    # 10. Sprints: 2021 (top 3) vs 2022+ (top 8)
    assert F1ScoringRules.get_sprint_points_distribution(2021) == [3.0, 2.0, 1.0]
    assert F1ScoringRules.get_sprint_points_distribution(2024) == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


def test_scoring_rules_variations():
    """Verify regulation scoring differences across historical and modern eras."""
    # Modern 2024: 25 win + 1 FL = 26 pts
    assert F1ScoringRules.get_max_points_per_event(2024, has_sprint=False) == 26.0
    # 2024 with Sprint: 26 + 8 = 34 pts
    assert F1ScoringRules.get_max_points_per_event(2024, has_sprint=True) == 34.0
    # 2026 (No Fastest Lap point): 25 win = 25 pts
    assert F1ScoringRules.get_max_points_per_event(2026, has_sprint=False) == 25.0
    # 2005: 10 pts
    assert F1ScoringRules.get_max_points_per_event(2005, has_sprint=False) == 10.0


def test_reconcile_constructor_standings():
    """Verify cross-source reconciler detects matches and conflicts for constructor standings."""
    df_j = pd.DataFrame([
        {"season": 2024, "round": 1, "constructor_id": "red_bull", "points": 44.0, "position": 1, "wins": 1},
        {"season": 2024, "round": 1, "constructor_id": "ferrari", "points": 27.0, "position": 2, "wins": 0},
    ])
    df_ref = pd.DataFrame([
        {"season": 2024, "round": 1, "constructor_id": "red_bull", "points": 44.0, "position": 1, "wins": 1},
        {"season": 2024, "round": 1, "constructor_id": "ferrari", "points": 25.0, "position": 3, "wins": 0},  # Discrepancy
    ])

    reconciler = JolpicaReconciler()
    rep = reconciler.reconcile_constructor_standings(df_j, df_ref, target_seasons=[2024])

    assert rep.overlap_count == 2
    assert rep.identical_count == 1
    assert rep.conflict_count == 2  # points and position differ for ferrari
    assert rep.conflicts_detected[0].entity_id == "ferrari"


def test_evaluate_championship_outcomes_simulation_contract():
    """Verify predictive simulation title probability evaluation contract."""
    df_stand = pd.DataFrame([
        {"season": 2024, "round": 20, "driver_id": "verstappen", "points": 300.0, "position": 1, "wins": 7, "position_text": "1", "source_api": "test", "retrieved_at": ""},
        {"season": 2024, "round": 20, "driver_id": "norris", "points": 280.0, "position": 2, "wins": 3, "position_text": "2", "source_api": "test", "retrieved_at": ""},
    ])
    analytics = ChampionshipAnalytics(df_driver_standings=df_stand)

    # 4 simulated scenarios for remaining races
    sim_points = {
        "verstappen": [50.0, 30.0, 40.0, 20.0],
        "norris":     [40.0, 60.0, 30.0, 50.0],
    }
    # Totals:
    # Sim 0: V = 350, N = 320 -> V wins
    # Sim 1: V = 330, N = 340 -> N wins
    # Sim 2: V = 340, N = 310 -> V wins
    # Sim 3: V = 320, N = 330 -> N wins
    # Probabilities: V: 0.5, N: 0.5
    probs = analytics.evaluate_championship_outcomes(2024, 20, sim_points)
    assert probs["verstappen"] == 0.5
    assert probs["norris"] == 0.5


# ---------------------------------------------------------------------------
# 5. Cross-Source Reconciliation Tests
# ---------------------------------------------------------------------------

def test_jolpica_reconciliation_conflict_detection():
    """Verify reconciler detects discrepancies between Jolpica and reference records."""
    df_jolpica = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "max_verstappen", "points": 26.0, "position": 1, "wins": 1},
        {"season": 2024, "round": 1, "driver_id": "norris", "points": 12.0, "position": 4, "wins": 0},
    ])
    df_reference = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "max_verstappen", "points": 26.0, "position": 1, "wins": 1},
        {"season": 2024, "round": 1, "driver_id": "norris", "points": 10.0, "position": 5, "wins": 0},  # Discrepancy
    ])

    reconciler = JolpicaReconciler()
    report = reconciler.reconcile_driver_standings(df_jolpica, df_reference, target_seasons=[2024])

    assert report.overlap_count == 2
    assert report.identical_count == 1
    assert report.conflict_count == 2  # points and position both differ for norris
    assert report.conflicts_detected[0].entity_id == "norris"
    assert report.conflicts_detected[0].authoritative_source == "jolpica.ergast"


# ---------------------------------------------------------------------------
# 6. Additional Schema, Idempotency & Edge Case Tests
# ---------------------------------------------------------------------------

def test_ingest_constructor_standings_schema():
    """Verify constructor standings mapping into CONSTRUCTOR_STANDINGS_SCHEMA."""
    payload = [
        {"position": "1", "positionText": "1", "points": "44", "wins": "1", "Constructor": {"constructorId": "red_bull", "name": "Red Bull"}},
        {"position": "2", "positionText": "2", "points": "27", "wins": "0", "Constructor": {"constructorId": "ferrari", "name": "Ferrari"}},
    ]
    mock_client = Mock(spec=JolpicaClient)
    mock_client.fetch_constructor_standings.return_value = payload

    registry = CanonicalEntityRegistry()
    registry.register_constructor(CanonicalConstructor(canonical_id="red_bull", name="Red Bull Racing", fastf1_name="Red Bull", nationality="Austrian", ergast_id="red_bull"))
    registry.register_constructor(CanonicalConstructor(canonical_id="ferrari", name="Ferrari", fastf1_name="Ferrari", nationality="Italian", ergast_id="ferrari"))

    engine = JolpicaIngestionEngine(client=mock_client, registry=registry)
    df = engine.ingest_constructor_standings_for_round(2024, 1)

    assert len(df) == 2
    assert list(df.columns) == CONSTRUCTOR_STANDINGS_SCHEMA.names
    assert df["constructor_id"].tolist() == ["red_bull", "ferrari"]
    assert df["points"].tolist() == [44.0, 27.0]
    assert df["position"].tolist() == [1, 2]


def test_ingest_race_results_schema():
    """Verify race results parsing into RACE_RESULTS_SCHEMA with status and time."""
    payload = [{
        "round": "1",
        "Results": [{
            "number": "1",
            "position": "1",
            "positionText": "1",
            "points": "26",
            "Driver": {"driverId": "max_verstappen"},
            "Constructor": {"constructorId": "red_bull"},
            "grid": "1",
            "laps": "57",
            "status": "Finished",
            "Time": {"millis": "5504742", "time": "1:31:44.742"},
            "FastestLap": {"rank": "1", "lap": "39", "Time": {"time": "1:32.608"}, "AverageSpeed": {"speed": "210.383"}},
        }],
    }]
    mock_client = Mock(spec=JolpicaClient)
    mock_client.fetch_race_results.return_value = payload

    registry = CanonicalEntityRegistry()
    engine = JolpicaIngestionEngine(client=mock_client, registry=registry)
    df = engine.ingest_season_race_results(2024)

    assert len(df) == 1
    assert list(df.columns) == RACE_RESULTS_SCHEMA.names
    assert df["driver_id"].iloc[0] == "max_verstappen"
    assert df["finish_position"].iloc[0] == 1
    assert bool(df["is_classified"].iloc[0]) is True
    assert bool(df["is_dnf"].iloc[0]) is False
    assert df["total_time_ms"].iloc[0] == 5504742
    assert df["fastest_lap_rank"].iloc[0] == 1


def test_points_progression_trajectory():
    """Verify points progression calculates delta points scored per round."""
    df_stand = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "norris", "points": 18.0, "position": 2, "wins": 0},
        {"season": 2024, "round": 2, "driver_id": "norris", "points": 33.0, "position": 2, "wins": 0},
        {"season": 2024, "round": 3, "driver_id": "norris", "points": 58.0, "position": 1, "wins": 1},
    ])
    analytics = ChampionshipAnalytics(df_driver_standings=df_stand)
    prog = analytics.get_points_progression(season=2024, driver_id="norris")

    assert len(prog) == 3
    assert prog["points"].tolist() == [18.0, 33.0, 58.0]
    assert prog["points_scored_this_round"].tolist() == [18.0, 15.0, 25.0]


def test_championship_as_of_round_1_preseason():
    """Verify before_event=True at Round 1 gracefully returns empty pre-season standings without crashing."""
    df_stand = pd.DataFrame([
        {"season": 2024, "round": 1, "driver_id": "max_verstappen", "points": 25.0, "position": 1, "wins": 1},
    ])
    analytics = ChampionshipAnalytics(df_driver_standings=df_stand)
    pre_s1 = analytics.get_driver_standings_as_of(season=2024, as_of_round=1, before_event=True)
    assert len(pre_s1) == 0


def test_idempotent_multi_season_ingestion(tmp_path):
    """Verify repeat ingestion produces identical persisted Parquet record counts."""
    mock_client = Mock(spec=JolpicaClient)
    mock_client.fetch_schedule.return_value = [{"round": "1"}]
    mock_client.fetch_driver_standings.return_value = [
        {"position": "1", "positionText": "1", "points": "25", "wins": "1", "Driver": {"driverId": "max_verstappen"}},
    ]
    mock_client.fetch_constructor_standings.return_value = [
        {"position": "1", "positionText": "1", "points": "25", "wins": "1", "Constructor": {"constructorId": "red_bull"}},
    ]
    mock_client.fetch_race_results.return_value = []
    mock_client.fetch_qualifying_results.return_value = []
    mock_client.fetch_sprint_results.return_value = []

    engine = JolpicaIngestionEngine(client=mock_client, base_dir=tmp_path)

    # First run
    res1 = engine.execute_season_ingestion([2024])
    assert res1.success is True
    assert res1.driver_standings_count == 1

    # Second run (identical repeat)
    res2 = engine.execute_season_ingestion([2024])
    assert res2.success is True
    assert res2.driver_standings_count == 1

    # Master table check in tmp_path
    master_d = pd.read_parquet(tmp_path / "data" / "historical" / "historical_driver_standings.parquet")
    assert len(master_d) == 1  # Deduplicated, no duplicates added

