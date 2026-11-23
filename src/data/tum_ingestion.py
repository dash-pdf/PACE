"""TUM Racetrack Database Ingestion and Normalization Engine.

Processes all raw CSV track centerlines and race lines from the TUM database,
resolves circuit identities against CanonicalEntityRegistry, applies differential
geometry analysis via CircuitGeometryProcessor, and persists strongly-typed
PyArrow Parquet datasets to data/historical/tum/.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd

from src.data.canonical_registry import CanonicalEntityRegistry
from src.data.schemas import (
    TUM_TRACK_GEOMETRY_SCHEMA,
    TUM_RACELINE_SCHEMA,
    TUM_CIRCUIT_SUMMARY_SCHEMA,
    TUM_CORNER_PROFILE_SCHEMA,
)
from src.data.storage import (
    get_tum_dir,
    get_tum_path,
    save_parquet,
    load_parquet,
    dataset_exists,
)
from src.data.tum_client import TUMClient, TUMMetadata
from src.features.circuit_geometry import (
    CircuitGeometryProcessor,
    CircuitGeometryAnalysis,
    CornerProfile,
)

logger = logging.getLogger("PACE.TUM.Ingestion")


class TUMIngestionEngine:
    """Ingestion, normalization, and persistence engine for TUM circuit geometry."""

    # Explicit mapping for non-canonical tracks (DTM / historic non-active F1)
    # These retain their clean normalized identifier when not in the 32 F1 canonical calendar
    NON_CANONICAL_FALLBACKS: Dict[str, str] = {
        "BrandsHatch": "brands_hatch",
        "IMS": "indianapolis",
        "MoscowRaceway": "moscow_raceway",
        "Norisring": "norisring",
        "Oschersleben": "oschersleben",
    }

    def __init__(
        self,
        client: Optional[TUMClient] = None,
        registry: Optional[CanonicalEntityRegistry] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.client = client or TUMClient()
        self.registry = registry or CanonicalEntityRegistry()
        self.output_dir = get_tum_dir(base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unresolved: Set[str] = set()

    def resolve_circuit_identity(self, tum_name: str) -> Tuple[str, bool, str]:
        """Resolve TUM track filename to canonical circuit ID, F1 status, and strategy."""
        res = self.registry.resolve_circuit(tum_name)
        if res.is_resolved and res.canonical_id:
            return res.canonical_id, True, res.strategy

        # Check non-canonical fallback table
        if tum_name in self.NON_CANONICAL_FALLBACKS:
            fallback_id = self.NON_CANONICAL_FALLBACKS[tum_name]
            return fallback_id, False, "non_canonical_fallback"

        # Unresolved
        self.unresolved.add(tum_name)
        clean_id = tum_name.lower().replace(" ", "_")
        return clean_id, False, "unresolved_stem"

    def ingest_single_circuit(
        self,
        tum_name: str,
        force: bool = False,
    ) -> CircuitGeometryAnalysis:
        """Load, analyze, and persist geometric datasets for a single TUM circuit."""
        self.client.ensure_repository()
        canonical_id, is_canonical, _ = self.resolve_circuit_identity(tum_name)
        metadata = self.client.get_metadata()

        geo_file = self.output_dir / f"{canonical_id}_track_geometry.parquet"
        race_file = self.output_dir / f"{canonical_id}_raceline.parquet"
        corner_file = self.output_dir / f"{canonical_id}_corners.parquet"

        df_raw_track = self.client.load_raw_track(tum_name)
        df_raw_raceline = None
        try:
            df_raw_raceline = self.client.load_raw_raceline(tum_name)
        except Exception as e:
            logger.debug(f"Raceline not loaded for {tum_name}: {e}")

        analysis = CircuitGeometryProcessor.process_track(
            df_raw_track=df_raw_track,
            circuit_id=canonical_id,
            tum_circuit_name=tum_name,
            is_canonical_f1=is_canonical,
            source_commit=metadata.commit_hash,
            df_raw_raceline=df_raw_raceline,
        )

        # Persist track geometry
        save_parquet(analysis.df_geometry, geo_file, schema=TUM_TRACK_GEOMETRY_SCHEMA)

        # Persist raceline if present
        if analysis.df_raceline is not None and not analysis.df_raceline.empty:
            save_parquet(analysis.df_raceline, race_file, schema=TUM_RACELINE_SCHEMA)

        # Persist corner profiles
        corner_rows = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        for c in analysis.corners:
            corner_rows.append({
                "circuit_id": canonical_id,
                "corner_number": int(c.corner_number),
                "turn_direction": str(c.turn_direction),
                "speed_category": str(c.speed_category),
                "start_s_m": float(c.start_s_m),
                "apex_s_m": float(c.apex_s_m),
                "end_s_m": float(c.end_s_m),
                "corner_length_m": float(c.corner_length_m),
                "apex_radius_m": float(c.apex_radius_m),
                "apex_curvature_radpm": float(c.apex_curvature_radpm),
                "min_track_width_m": float(c.min_track_width_m),
                "source_api": "tum",
                "retrieved_at": retrieved_at,
            })

        if corner_rows:
            df_corners = pd.DataFrame(corner_rows)
            save_parquet(df_corners, corner_file, schema=TUM_CORNER_PROFILE_SCHEMA)
        else:
            df_empty_corners = pd.DataFrame(columns=TUM_CORNER_PROFILE_SCHEMA.names)
            save_parquet(df_empty_corners, corner_file, schema=TUM_CORNER_PROFILE_SCHEMA)

        return analysis

    def ingest_all(self, force: bool = False) -> Dict[str, Any]:
        """Ingest, process, and persist all 25 circuits in the TUM database."""
        self.client.ensure_repository()
        metadata = self.client.get_metadata()
        track_names = self.client.list_tracks()

        summary_rows: List[Dict[str, Any]] = []
        total_points = 0
        total_corners = 0
        canonical_count = 0
        non_canonical_count = 0

        for tum_name in track_names:
            analysis = self.ingest_single_circuit(tum_name, force=force)
            total_points += len(analysis.df_geometry)
            total_corners += len(analysis.corners)
            if analysis.is_canonical_f1:
                canonical_count += 1
            else:
                non_canonical_count += 1

            low_corners = sum(1 for c in analysis.corners if c.speed_category == "low")
            med_corners = sum(1 for c in analysis.corners if c.speed_category == "medium")
            high_corners = sum(1 for c in analysis.corners if c.speed_category == "high")

            summary_rows.append({
                "circuit_id": analysis.circuit_id,
                "tum_circuit_name": analysis.tum_circuit_name,
                "is_canonical_f1": bool(analysis.is_canonical_f1),
                "num_centerline_points": int(len(analysis.df_geometry)),
                "num_raceline_points": int(len(analysis.df_raceline)) if analysis.df_raceline is not None else 0,
                "track_length_m": float(analysis.track_length_m),
                "raceline_length_m": float(analysis.raceline_length_m) if analysis.raceline_length_m is not None else 0.0,
                "closure_error_m": float(analysis.closure_error_m),
                "min_width_m": float(analysis.min_width_m),
                "max_width_m": float(analysis.max_width_m),
                "mean_width_m": float(analysis.mean_width_m),
                "num_corners_detected": int(len(analysis.corners)),
                "num_low_speed_corners": int(low_corners),
                "num_medium_speed_corners": int(med_corners),
                "num_high_speed_corners": int(high_corners),
                "longest_straight_m": float(analysis.longest_straight_m),
                "low_speed_pct": float(analysis.spectrum.low_speed_pct),
                "medium_speed_pct": float(analysis.spectrum.medium_speed_pct),
                "high_speed_pct": float(analysis.spectrum.high_speed_pct),
                "straight_pct": float(analysis.spectrum.straight_pct),
                "source_commit": str(analysis.source_commit),
                "source_license": str(analysis.source_license),
                "retrieved_at": str(analysis.retrieved_at),
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_file = self.output_dir / "tum_circuits_summary.parquet"
        save_parquet(df_summary, summary_file, schema=TUM_CIRCUIT_SUMMARY_SCHEMA)

        metrics = {
            "total_circuits": len(summary_rows),
            "canonical_f1_matched": canonical_count,
            "non_canonical": non_canonical_count,
            "total_points_ingested": total_points,
            "total_corners_detected": total_corners,
            "commit_hash": metadata.commit_hash,
            "license": metadata.license,
            "unresolved_entities": sorted(list(self.unresolved)),
        }
        logger.info(
            f"TUM ingestion complete: {len(summary_rows)} circuits ({canonical_count} F1), "
            f"{total_points} centerline points, {total_corners} corners persisted."
        )
        return metrics

    def load_summary(self) -> pd.DataFrame:
        """Load the master summary table of all ingested TUM circuits."""
        summary_file = self.output_dir / "tum_circuits_summary.parquet"
        if not dataset_exists(summary_file):
            # Ingest if not yet persisted
            self.ingest_all()
        return load_parquet(summary_file)

    def load_circuit_geometry(self, circuit_id: str) -> Optional[pd.DataFrame]:
        """Load normalized point-by-point centerline and geometry for a circuit."""
        # Check canonical resolution if user passed TUM name
        res = self.registry.resolve_circuit(circuit_id)
        cid = res.canonical_id if res.is_resolved and res.canonical_id else circuit_id
        if cid in self.NON_CANONICAL_FALLBACKS.values():
            pass
        elif cid in self.NON_CANONICAL_FALLBACKS:
            cid = self.NON_CANONICAL_FALLBACKS[cid]

        p = self.output_dir / f"{cid}_track_geometry.parquet"
        if dataset_exists(p):
            return load_parquet(p)

        # Try ingesting if not yet present
        for t in self.client.list_tracks():
            resolved_id, _, _ = self.resolve_circuit_identity(t)
            if resolved_id == cid or t == circuit_id:
                self.ingest_single_circuit(t)
                if dataset_exists(p):
                    return load_parquet(p)

        return None

    def load_circuit_corners(self, circuit_id: str) -> Optional[pd.DataFrame]:
        """Load corner profiles for a circuit."""
        res = self.registry.resolve_circuit(circuit_id)
        cid = res.canonical_id if res.is_resolved and res.canonical_id else circuit_id
        if cid in self.NON_CANONICAL_FALLBACKS:
            cid = self.NON_CANONICAL_FALLBACKS[cid]

        p = self.output_dir / f"{cid}_corners.parquet"
        if dataset_exists(p):
            return load_parquet(p)
        return None

    def load_circuit_raceline(self, circuit_id: str) -> Optional[pd.DataFrame]:
        """Load minimum curvature raceline for a circuit."""
        res = self.registry.resolve_circuit(circuit_id)
        cid = res.canonical_id if res.is_resolved and res.canonical_id else circuit_id
        if cid in self.NON_CANONICAL_FALLBACKS:
            cid = self.NON_CANONICAL_FALLBACKS[cid]

        p = self.output_dir / f"{cid}_raceline.parquet"
        if dataset_exists(p):
            return load_parquet(p)
        return None
