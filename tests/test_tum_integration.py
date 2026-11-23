"""Unit and integration tests for TUM Racetrack Database integration in PACE.

Verifies:
1. Client inventory, raw file parsing, metadata, and offline mode.
2. Coordinate completeness, step sizes, and closed-loop distance calculations.
3. Differential geometry processing (heading, central difference curvature, radius).
4. Inferred feature extraction (corner detection, speed classification, curvature spectrum).
5. PyArrow schema conformity across all TUM tables.
6. Canonical circuit entity matching and ambiguity handling.
7. Discrepancy detection, track closure verification, and layout reconfiguration flags.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.canonical_registry import CanonicalEntityRegistry
from src.data.schemas import (
    TUM_TRACK_GEOMETRY_SCHEMA,
    TUM_RACELINE_SCHEMA,
    TUM_CIRCUIT_SUMMARY_SCHEMA,
    TUM_CORNER_PROFILE_SCHEMA,
)
from src.data.tum_client import (
    TUMClient,
    TUMClientConfig,
    TUMMetadata,
    TUMTrackNotFoundError,
    TUMRepositoryNotFoundError,
)
from src.data.tum_ingestion import TUMIngestionEngine
from src.data.tum_reconciler import TUMReconciler, TUMReconciliationReport
from src.features.circuit_geometry import (
    CircuitGeometryProcessor,
    CircuitGeometryAnalysis,
    CornerProfile,
    StraightSegment,
    CurvatureSpectrum,
)


@pytest.fixture
def tum_client() -> TUMClient:
    """Fixture providing initialized TUM client targeting the local workspace repo."""
    return TUMClient()


@pytest.fixture
def tum_engine(tum_client) -> TUMIngestionEngine:
    """Fixture providing initialized TUM ingestion engine."""
    return TUMIngestionEngine(client=tum_client)


@pytest.fixture
def tum_reconciler(tum_engine) -> TUMReconciler:
    """Fixture providing initialized TUM reconciler."""
    return TUMReconciler(ingestion_engine=tum_engine)


# ---------------------------------------------------------------------------
# 1. Client & Repository Inventory Tests
# ---------------------------------------------------------------------------

def test_tum_client_metadata(tum_client):
    """Verify repository provenance, commit hash, author, and LGPL-3.0 license."""
    meta = tum_client.get_metadata()
    assert meta.commit_hash == "e59595d1f3573b30d1ded6a08984935b957688e0"
    assert "alexander.heilmeier" in meta.author
    assert meta.license == "LGPL-3.0"
    assert meta.num_tracks == 25
    assert meta.num_racelines == 25


def test_tum_client_list_tracks(tum_client):
    """Verify inventory lists exactly 25 track files."""
    tracks = tum_client.list_tracks()
    assert len(tracks) == 25
    assert "Monza" in tracks
    assert "Austin" in tracks
    assert "Spa" in tracks
    assert "Silverstone" in tracks
    assert "SaoPaulo" in tracks


def test_tum_client_load_raw_track(tum_client):
    """Verify raw track CSV loads with expected columns and non-null values."""
    df = tum_client.load_raw_track("Monza")
    assert not df.empty
    assert list(df.columns) == ["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"]
    assert len(df) == 1159
    assert not df.isna().any().any()
    # Check plausible track width (right + left > 6m)
    total_w = df["w_tr_right_m"] + df["w_tr_left_m"]
    assert total_w.min() > 6.0
    assert total_w.max() < 25.0


def test_tum_client_load_raw_raceline(tum_client):
    """Verify raw raceline CSV loads with 2D coordinates."""
    df = tum_client.load_raw_raceline("Monza")
    assert not df.empty
    assert list(df.columns) == ["x_m", "y_m"]
    assert not df.isna().any().any()


def test_tum_client_track_not_found(tum_client):
    """Verify proper error raised when requesting a non-existent track."""
    with pytest.raises(TUMTrackNotFoundError):
        tum_client.load_raw_track("NonExistentCircuit123")


def test_tum_client_offline_mode_guard():
    """Verify offline mode raises TUMRepositoryNotFoundError when repo dir is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_empty_dir = Path(tmp_dir) / "empty_repo"
        cfg = TUMClientConfig(local_repo_dir=fake_empty_dir, allow_network=False)
        client = TUMClient(config=cfg)
        with pytest.raises(TUMRepositoryNotFoundError):
            client.ensure_repository()


# ---------------------------------------------------------------------------
# 2. Geometry Processing & Numerical Verification Tests
# ---------------------------------------------------------------------------

def test_geometry_processor_step_spacing_and_closure(tum_client):
    """Verify centerline step sizes are ~5.0m and closure step matches sampling."""
    df_raw = tum_client.load_raw_track("Monza")
    analysis = CircuitGeometryProcessor.process_track(
        df_raw_track=df_raw,
        circuit_id="monza",
        tum_circuit_name="Monza",
    )

    # Centerline length
    assert 5780.0 < analysis.track_length_m < 5800.0
    # Closure gap is ~5.0m
    assert 4.8 < analysis.closure_error_m < 5.2

    # Step sizes between consecutive points: nominal 5.0m spacing with small spline variance
    dx = np.diff(df_raw["x_m"].values)
    dy = np.diff(df_raw["y_m"].values)
    ds = np.sqrt(dx**2 + dy**2)
    assert 4.9 < np.mean(ds) < 5.1
    assert 4.9 < np.median(ds) < 5.1
    assert ds.min() > 4.0
    assert ds.max() < 6.0


def test_geometry_processor_curvature_and_heading(tum_client):
    """Verify heading angles and central difference curvature calculations."""
    df_raw = tum_client.load_raw_track("Monza")
    analysis = CircuitGeometryProcessor.process_track(
        df_raw_track=df_raw,
        circuit_id="monza",
        tum_circuit_name="Monza",
    )
    df_geo = analysis.df_geometry

    # Curvature columns present
    assert "kappa_radpm" in df_geo.columns
    assert "radius_m" in df_geo.columns
    assert "psi_rad" in df_geo.columns

    # Minimum radius at Rettifilo chicane is ~9.8m
    min_r = df_geo["radius_m"].min()
    assert 8.0 < min_r < 12.0

    # Max curvature is ~0.10 rad/m
    max_k = np.abs(df_geo["kappa_radpm"]).max()
    assert 0.08 < max_k < 0.12


def test_geometry_processor_corner_detection(tum_client):
    """Verify corner detection identifies real-world turns and speed categories."""
    df_raw = tum_client.load_raw_track("Monza")
    analysis = CircuitGeometryProcessor.process_track(
        df_raw_track=df_raw,
        circuit_id="monza",
        tum_circuit_name="Monza",
    )
    corners = analysis.corners
    assert len(corners) >= 10

    # Verify first corner is Rettifilo (low speed chicane)
    c1 = corners[0]
    assert c1.speed_category == "low"
    assert c1.apex_radius_m < 15.0

    # Verify speed category values are valid
    for c in corners:
        assert c.speed_category in ("low", "medium", "high")
        assert c.turn_direction in ("left", "right")
        assert c.corner_length_m >= 15.0


def test_geometry_processor_curvature_spectrum(tum_client):
    """Verify curvature spectrum percentages sum to 100%."""
    df_raw = tum_client.load_raw_track("Monza")
    analysis = CircuitGeometryProcessor.process_track(
        df_raw_track=df_raw,
        circuit_id="monza",
        tum_circuit_name="Monza",
    )
    spec = analysis.spectrum
    total_pct = spec.low_speed_pct + spec.medium_speed_pct + spec.high_speed_pct + spec.straight_pct
    assert pytest.approx(total_pct, abs=0.1) == 100.0
    # Monza is high straight percentage (> 70%)
    assert spec.straight_pct > 70.0


def test_geometry_processor_raceline(tum_client):
    """Verify minimum curvature raceline is shorter than centerline (apex clipping)."""
    df_raw_track = tum_client.load_raw_track("Monza")
    df_raw_race = tum_client.load_raw_raceline("Monza")

    analysis = CircuitGeometryProcessor.process_track(
        df_raw_track=df_raw_track,
        circuit_id="monza",
        tum_circuit_name="Monza",
        df_raw_raceline=df_raw_race,
    )
    assert analysis.raceline_length_m is not None
    # Raceline must be strictly shorter than centerline
    assert analysis.raceline_length_m < analysis.track_length_m
    delta_m = analysis.track_length_m - analysis.raceline_length_m
    assert delta_m > 20.0  # At least 20m cut across lap


# ---------------------------------------------------------------------------
# 3. Ingestion & Schema Conformity Tests
# ---------------------------------------------------------------------------

def test_ingest_single_circuit_schemas(tum_engine):
    """Verify single circuit ingestion strictly matches PyArrow schemas."""
    analysis = tum_engine.ingest_single_circuit("Spa")
    assert analysis.circuit_id == "spa"
    assert analysis.tum_circuit_name == "Spa"

    # Validate geometry DataFrame schema
    assert list(analysis.df_geometry.columns) == TUM_TRACK_GEOMETRY_SCHEMA.names
    # Validate raceline DataFrame schema
    if analysis.df_raceline is not None:
        assert list(analysis.df_raceline.columns) == TUM_RACELINE_SCHEMA.names

    # Check persisted files
    p_geo = tum_engine.output_dir / "spa_track_geometry.parquet"
    p_race = tum_engine.output_dir / "spa_raceline.parquet"
    p_corn = tum_engine.output_dir / "spa_corners.parquet"

    assert p_geo.exists()
    assert p_race.exists()
    assert p_corn.exists()


def test_ingest_all_and_summary_schema(tum_engine):
    """Verify master summary table matches TUM_CIRCUIT_SUMMARY_SCHEMA."""
    df_summary = tum_engine.load_summary()
    assert len(df_summary) == 25
    assert list(df_summary.columns) == TUM_CIRCUIT_SUMMARY_SCHEMA.names

    # 20 F1 circuits, 5 non-canonical
    assert int(df_summary["is_canonical_f1"].sum()) == 20
    assert int((~df_summary["is_canonical_f1"]).sum()) == 5


# ---------------------------------------------------------------------------
# 4. Canonical Circuit Matching & Reconciliation Tests
# ---------------------------------------------------------------------------

def test_tum_canonical_matching_coverage(tum_reconciler):
    """Verify 20 F1 circuits match canonical registry and 5 non-canonical are flagged."""
    report = tum_reconciler.reconcile_all()
    assert report.total_tum_circuits == 25
    assert report.matched_f1_circuits == 20
    assert report.non_canonical_circuits == 5

    # Check non-canonical circuits
    assert "BrandsHatch" in report.unmapped_tum_tracks
    assert "IMS" in report.unmapped_tum_tracks
    assert "Norisring" in report.unmapped_tum_tracks

    # Check missing canonical F1 circuits documented
    assert "vegas" in report.missing_canonical_f1
    assert "monaco" in report.missing_canonical_f1
    assert "baku" in report.missing_canonical_f1


def test_tum_reconciliation_accuracy_and_reconfigurations(tum_reconciler):
    """Verify length accuracy across modern tracks and flag historical reconfigurations."""
    report = tum_reconciler.reconcile_all()
    comp_map = {c.circuit_id: c for c in report.comparisons}

    # High fidelity modern circuits (|delta| < 10m)
    assert abs(comp_map["monza"].length_delta_m) < 5.0
    assert abs(comp_map["spa"].length_delta_m) < 5.0
    assert abs(comp_map["silverstone"].length_delta_m) < 5.0
    assert abs(comp_map["americas"].length_delta_m) < 6.0
    assert abs(comp_map["interlagos"].length_delta_m) < 5.0

    # Flagged layout reconfigurations
    assert comp_map["yas_marina"].match_status == "RECONFIGURED_LAYOUT"
    assert comp_map["yas_marina"].length_delta_m > 200.0  # Pre-2021 chicane layout

    assert comp_map["zandvoort"].match_status == "RECONFIGURED_LAYOUT"
    assert comp_map["zandvoort"].length_delta_m > 50.0  # Pre-banking layout


def test_circuit_geometry_loader_methods(tum_engine):
    """Verify convenience loader methods retrieve dataframes cleanly."""
    df_geo = tum_engine.load_circuit_geometry("silverstone")
    assert df_geo is not None
    assert len(df_geo) > 1000

    df_corners = tum_engine.load_circuit_corners("silverstone")
    assert df_corners is not None
    assert not df_corners.empty

    df_race = tum_engine.load_circuit_raceline("silverstone")
    assert df_race is not None
    assert not df_race.empty
