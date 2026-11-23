"""Circuit Geometry and Trajectory Processing for PACE.

Analyzes raw 2D track geometries from the TUM Racetrack Database.
Calculates derived spatial measurements (arc length, heading, curvature,
curvature radius, track width) and infers engineering characteristics
(corner detection, speed classification, straight segments, curvature spectrum,
and braking severity index).

Methodological note:
- Source measurements: x_m, y_m, w_tr_right_m, w_tr_left_m (from TUM OSM/satellite data).
- Derived measurements: s_m, total_length_m, psi_rad, kappa_radpm, radius_m, w_total_m.
- Inferred values: detected corners, apex speed category, straights, curvature spectrum.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.schemas import (
    TUM_TRACK_GEOMETRY_SCHEMA,
    TUM_RACELINE_SCHEMA,
    TUM_CIRCUIT_SUMMARY_SCHEMA,
    TUM_CORNER_PROFILE_SCHEMA,
)

logger = logging.getLogger("PACE.Geometry")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CornerProfile:
    """Inferred corner metadata identified from localized curvature extrema."""
    corner_number: int                      # Sequential index (1, 2, ...)
    turn_direction: str                     # 'left' (kappa > 0) or 'right' (kappa < 0)
    speed_category: str                     # 'low' (R < 75m), 'medium' (75m <= R <= 150m), 'high' (R > 150m)
    start_s_m: float                        # Entry arc length (meters)
    apex_s_m: float                         # Apex arc length (meters)
    end_s_m: float                          # Exit arc length (meters)
    corner_length_m: float                  # Arc length of corner (meters)
    apex_radius_m: float                    # Minimum radius at apex (meters)
    apex_curvature_radpm: float             # Maximum curvature magnitude at apex (rad/m)
    min_track_width_m: float                # Minimum road width in corner (meters)


@dataclass(frozen=True)
class StraightSegment:
    """Inferred straight-line section identified by near-zero curvature."""
    straight_number: int
    start_s_m: float
    end_s_m: float
    length_m: float


@dataclass(frozen=True)
class CurvatureSpectrum:
    """Proportional breakdown of circuit distance by geometric turn regime."""
    low_speed_pct: float                    # % of lap in corners with R < 75m
    medium_speed_pct: float                 # % of lap in corners with 75m <= R <= 150m
    high_speed_pct: float                   # % of lap in corners with R > 150m
    straight_pct: float                     # % of lap in straights


@dataclass
class CircuitGeometryAnalysis:
    """Complete geometric intelligence output bundle for a single circuit."""
    circuit_id: str
    tum_circuit_name: str
    is_canonical_f1: bool
    df_geometry: pd.DataFrame               # Point-by-point centerline & derived fields
    df_raceline: Optional[pd.DataFrame]     # Point-by-point raceline if present
    corners: List[CornerProfile]
    straights: List[StraightSegment]
    spectrum: CurvatureSpectrum
    track_length_m: float                   # Closed centerline distance (meters)
    raceline_length_m: Optional[float]      # Closed raceline distance (meters)
    closure_error_m: float                  # Euclidean gap between end point and start point
    min_width_m: float
    max_width_m: float
    mean_width_m: float
    longest_straight_m: float
    source_commit: str
    source_license: str = "LGPL-3.0"
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Geometry Processor
# ---------------------------------------------------------------------------

class CircuitGeometryProcessor:
    """Processes raw track coordinates into rigorous derived and inferred features."""

    # Corner identification thresholds
    CORNER_RADIUS_THRESHOLD_M: float = 300.0   # Contiguous sections with R <= 300m are corners
    CORNER_MIN_LENGTH_M: float = 15.0          # Minimum length to qualify as a distinct corner
    STRAIGHT_RADIUS_THRESHOLD_M: float = 300.0 # Contiguous sections with R > 300m are straights
    STRAIGHT_MIN_LENGTH_M: float = 80.0        # Minimum length to qualify as a named straight

    # Speed classification boundaries
    LOW_SPEED_MAX_RADIUS_M: float = 75.0       # Tight corners, chicanes, hairpins (< 75m)
    MEDIUM_SPEED_MAX_RADIUS_M: float = 150.0   # Technical sweeping turns (75m - 150m)

    @classmethod
    def process_track(
        cls,
        df_raw_track: pd.DataFrame,
        circuit_id: str,
        tum_circuit_name: str,
        is_canonical_f1: bool = True,
        source_commit: str = "e59595d1f3573b30d1ded6a08984935b957688e0",
        df_raw_raceline: Optional[pd.DataFrame] = None,
    ) -> CircuitGeometryAnalysis:
        """Execute full geometric pipeline on raw TUM centerline and boundaries."""
        n = len(df_raw_track)
        if n < 4:
            raise ValueError(f"Insufficient track points ({n}) to compute differential geometry.")

        x = df_raw_track["x_m"].to_numpy(dtype=float)
        y = df_raw_track["y_m"].to_numpy(dtype=float)
        w_r = df_raw_track["w_tr_right_m"].to_numpy(dtype=float)
        w_l = df_raw_track["w_tr_left_m"].to_numpy(dtype=float)

        # 1. Step vectors & closure
        # Periodic wrap-around step from last point N-1 back to point 0
        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        ds = np.sqrt(dx**2 + dy**2)

        # Closure error is distance between final point and start point
        closure_error_m = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))

        # Total closed centerline length includes all steps
        total_length_m = float(np.sum(ds))

        # Cumulative distance s along centerline
        s_m = np.zeros(n, dtype=float)
        s_m[1:] = np.cumsum(ds[:-1])

        # 2. Total width
        w_total = w_r + w_l

        # 3. Heading angle psi (radians) along direction of travel
        psi = np.arctan2(dy, dx)

        # 4. Curvature kappa (rad/m) using 3-point periodic central differences
        # dpsi with proper circular modulo [-pi, pi] wrap-around
        dpsi = (np.roll(psi, -1) - np.roll(psi, 1) + np.pi) % (2.0 * np.pi) - np.pi
        ds_central = ds + np.roll(ds, 1)
        ds_central = np.where(ds_central > 1e-6, ds_central, 1e-6)
        kappa = dpsi / ds_central

        # 5. Radius of curvature R = 1 / |kappa|
        abs_k = np.abs(kappa)
        radius = np.where(abs_k > 1e-6, 1.0 / abs_k, 1e6)

        retrieved_at = datetime.now(timezone.utc).isoformat()

        # Build normalized track geometry DataFrame
        df_geo = pd.DataFrame({
            "circuit_id": circuit_id,
            "tum_circuit_name": tum_circuit_name,
            "point_index": np.arange(n, dtype=np.int32),
            "s_m": s_m,
            "x_m": x,
            "y_m": y,
            "w_tr_right_m": w_r,
            "w_tr_left_m": w_l,
            "w_total_m": w_total,
            "psi_rad": psi,
            "kappa_radpm": kappa,
            "radius_m": radius,
            "source_api": "tum",
            "retrieved_at": retrieved_at,
        })

        # 6. Corner identification
        corners = cls._detect_corners(s_m, kappa, radius, w_total)

        # 7. Straight-line identification
        straights = cls._detect_straights(s_m, radius, total_length_m)
        longest_straight = max([st.length_m for st in straights], default=0.0)

        # 8. Curvature spectrum
        spectrum = cls._calculate_curvature_spectrum(corners, total_length_m)

        # 9. Optional raceline processing
        df_race = None
        raceline_len = None
        if df_raw_raceline is not None and not df_raw_raceline.empty:
            df_race, raceline_len = cls._process_raceline(
                df_raw_raceline, circuit_id, tum_circuit_name, retrieved_at
            )

        return CircuitGeometryAnalysis(
            circuit_id=circuit_id,
            tum_circuit_name=tum_circuit_name,
            is_canonical_f1=is_canonical_f1,
            df_geometry=df_geo[TUM_TRACK_GEOMETRY_SCHEMA.names],
            df_raceline=df_race[TUM_RACELINE_SCHEMA.names] if df_race is not None else None,
            corners=corners,
            straights=straights,
            spectrum=spectrum,
            track_length_m=round(total_length_m, 2),
            raceline_length_m=round(raceline_len, 2) if raceline_len is not None else None,
            closure_error_m=round(closure_error_m, 3),
            min_width_m=round(float(np.min(w_total)), 2),
            max_width_m=round(float(np.max(w_total)), 2),
            mean_width_m=round(float(np.mean(w_total)), 2),
            longest_straight_m=round(longest_straight, 2),
            source_commit=source_commit,
            source_license="LGPL-3.0",
            retrieved_at=retrieved_at,
        )

    @classmethod
    def _detect_corners(
        cls,
        s_m: np.ndarray,
        kappa: np.ndarray,
        radius: np.ndarray,
        w_total: np.ndarray,
    ) -> List[CornerProfile]:
        """Detect and classify contiguous corners from local curvature extrema."""
        n = len(s_m)
        in_corner = radius <= cls.CORNER_RADIUS_THRESHOLD_M
        corners: List[CornerProfile] = []

        start_idx: Optional[int] = None

        for i in range(n):
            if in_corner[i] and start_idx is None:
                start_idx = i
            elif not in_corner[i] and start_idx is not None:
                end_idx = i - 1
                if end_idx >= start_idx:
                    c_len = s_m[end_idx] - s_m[start_idx]
                    if c_len >= cls.CORNER_MIN_LENGTH_M:
                        sub_k = kappa[start_idx : end_idx + 1]
                        sub_r = radius[start_idx : end_idx + 1]
                        sub_w = w_total[start_idx : end_idx + 1]

                        apex_local_idx = int(np.argmax(np.abs(sub_k)))
                        apex_idx = start_idx + apex_local_idx

                        apex_r = float(sub_r[apex_local_idx])
                        apex_k = float(sub_k[apex_local_idx])

                        # Speed category
                        if apex_r < cls.LOW_SPEED_MAX_RADIUS_M:
                            cat = "low"
                        elif apex_r <= cls.MEDIUM_SPEED_MAX_RADIUS_M:
                            cat = "medium"
                        else:
                            cat = "high"

                        # Direction: positive dpsi is left turn in counter-clockwise convention
                        direction = "left" if apex_k > 0 else "right"

                        corners.append(CornerProfile(
                            corner_number=len(corners) + 1,
                            turn_direction=direction,
                            speed_category=cat,
                            start_s_m=round(float(s_m[start_idx]), 2),
                            apex_s_m=round(float(s_m[apex_idx]), 2),
                            end_s_m=round(float(s_m[end_idx]), 2),
                            corner_length_m=round(float(c_len), 2),
                            apex_radius_m=round(apex_r, 2),
                            apex_curvature_radpm=round(apex_k, 5),
                            min_track_width_m=round(float(np.min(sub_w)), 2),
                        ))
                start_idx = None

        # Check trailing corner segment crossing start/finish boundary
        if start_idx is not None and start_idx < n:
            end_idx = n - 1
            c_len = s_m[end_idx] - s_m[start_idx]
            if c_len >= cls.CORNER_MIN_LENGTH_M:
                sub_k = kappa[start_idx : end_idx + 1]
                sub_r = radius[start_idx : end_idx + 1]
                sub_w = w_total[start_idx : end_idx + 1]
                apex_local_idx = int(np.argmax(np.abs(sub_k)))
                apex_idx = start_idx + apex_local_idx
                apex_r = float(sub_r[apex_local_idx])
                apex_k = float(sub_k[apex_local_idx])
                cat = "low" if apex_r < cls.LOW_SPEED_MAX_RADIUS_M else ("medium" if apex_r <= cls.MEDIUM_SPEED_MAX_RADIUS_M else "high")
                direction = "left" if apex_k > 0 else "right"

                corners.append(CornerProfile(
                    corner_number=len(corners) + 1,
                    turn_direction=direction,
                    speed_category=cat,
                    start_s_m=round(float(s_m[start_idx]), 2),
                    apex_s_m=round(float(s_m[apex_idx]), 2),
                    end_s_m=round(float(s_m[end_idx]), 2),
                    corner_length_m=round(float(c_len), 2),
                    apex_radius_m=round(apex_r, 2),
                    apex_curvature_radpm=round(apex_k, 5),
                    min_track_width_m=round(float(np.min(sub_w)), 2),
                ))

        return corners

    @classmethod
    def _detect_straights(
        cls,
        s_m: np.ndarray,
        radius: np.ndarray,
        total_length_m: float,
    ) -> List[StraightSegment]:
        """Detect straight sections with low curvature above minimum length threshold."""
        n = len(s_m)
        in_straight = radius > cls.STRAIGHT_RADIUS_THRESHOLD_M
        straights: List[StraightSegment] = []

        start_idx: Optional[int] = None
        for i in range(n):
            if in_straight[i] and start_idx is None:
                start_idx = i
            elif not in_straight[i] and start_idx is not None:
                end_idx = i - 1
                if end_idx >= start_idx:
                    st_len = s_m[end_idx] - s_m[start_idx]
                    if st_len >= cls.STRAIGHT_MIN_LENGTH_M:
                        straights.append(StraightSegment(
                            straight_number=len(straights) + 1,
                            start_s_m=round(float(s_m[start_idx]), 2),
                            end_s_m=round(float(s_m[end_idx]), 2),
                            length_m=round(float(st_len), 2),
                        ))
                start_idx = None

        if start_idx is not None and start_idx < n:
            end_idx = n - 1
            st_len = s_m[end_idx] - s_m[start_idx]
            if st_len >= cls.STRAIGHT_MIN_LENGTH_M:
                straights.append(StraightSegment(
                    straight_number=len(straights) + 1,
                    start_s_m=round(float(s_m[start_idx]), 2),
                    end_s_m=round(float(s_m[end_idx]), 2),
                    length_m=round(float(st_len), 2),
                ))

        return straights

    @classmethod
    def _calculate_curvature_spectrum(
        cls,
        corners: List[CornerProfile],
        total_length_m: float,
    ) -> CurvatureSpectrum:
        """Calculate percentage breakdown of track length spent in corner regimes vs straights."""
        if total_length_m <= 0:
            return CurvatureSpectrum(0.0, 0.0, 0.0, 100.0)

        low_len = sum(c.corner_length_m for c in corners if c.speed_category == "low")
        med_len = sum(c.corner_length_m for c in corners if c.speed_category == "medium")
        high_len = sum(c.corner_length_m for c in corners if c.speed_category == "high")

        total_corner_len = low_len + med_len + high_len
        straight_len = max(0.0, total_length_m - total_corner_len)

        low_pct = (low_len / total_length_m) * 100.0
        med_pct = (med_len / total_length_m) * 100.0
        high_pct = (high_len / total_length_m) * 100.0
        straight_pct = (straight_len / total_length_m) * 100.0

        return CurvatureSpectrum(
            low_speed_pct=round(low_pct, 2),
            medium_speed_pct=round(med_pct, 2),
            high_speed_pct=round(high_pct, 2),
            straight_pct=round(straight_pct, 2),
        )

    @classmethod
    def _process_raceline(
        cls,
        df_raw_raceline: pd.DataFrame,
        circuit_id: str,
        tum_circuit_name: str,
        retrieved_at: str,
    ) -> Tuple[pd.DataFrame, float]:
        """Compute cumulative raceline distance and total path length."""
        n = len(df_raw_raceline)
        x = df_raw_raceline["x_m"].to_numpy(dtype=float)
        y = df_raw_raceline["y_m"].to_numpy(dtype=float)

        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        ds = np.sqrt(dx**2 + dy**2)

        total_len = float(np.sum(ds))
        s_m = np.zeros(n, dtype=float)
        s_m[1:] = np.cumsum(ds[:-1])

        df_race = pd.DataFrame({
            "circuit_id": circuit_id,
            "tum_circuit_name": tum_circuit_name,
            "point_index": np.arange(n, dtype=np.int32),
            "s_m": s_m,
            "x_m": x,
            "y_m": y,
            "source_api": "tum",
            "retrieved_at": retrieved_at,
        })
        return df_race, total_len
