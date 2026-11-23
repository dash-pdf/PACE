"""TUM Racetrack Database Client for PACE.

Integrates the official TUM Racetrack Database (Chair of Automotive Technology,
Technical University of Munich): https://github.com/TUMFTM/racetrack-database
License: GNU Lesser General Public License v3 (LGPL-3.0).

Provides repository management, offline caching, and raw CSV parsing for
track centerlines, boundaries, and minimum curvature race lines.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from src.data.storage import get_tum_raw_dir

logger = logging.getLogger("PACE.TUM")


# ---------------------------------------------------------------------------
# Exceptions & Configuration
# ---------------------------------------------------------------------------

class TUMError(Exception):
    """Base exception for TUM Racetrack Database operations."""
    pass


class TUMRepositoryNotFoundError(TUMError):
    """Raised when the local TUM repository is missing and cannot be retrieved."""
    pass


class TUMTrackNotFoundError(TUMError):
    """Raised when a requested track name is not present in the TUM database."""
    pass


@dataclass
class TUMClientConfig:
    """Configuration for TUM Racetrack Database client."""
    repo_url: str = "https://github.com/TUMFTM/racetrack-database.git"
    local_repo_dir: Optional[Path] = None
    allow_network: bool = True
    timeout_s: int = 60


@dataclass(frozen=True)
class TUMMetadata:
    """Authoritative metadata and provenance for the TUM Racetrack Database."""
    commit_hash: str
    commit_date: str
    author: str
    license: str
    repo_url: str
    local_path: str
    num_tracks: int
    num_racelines: int


@dataclass(frozen=True)
class TUMRawTrackInfo:
    """Inventory metadata for a single circuit in the raw TUM database."""
    track_name: str
    track_csv: Path
    raceline_csv: Optional[Path]
    raceline_png: Optional[Path]
    curvature_png: Optional[Path]
    point_count: int
    file_size_bytes: int


# ---------------------------------------------------------------------------
# TUM Client
# ---------------------------------------------------------------------------

class TUMClient:
    """Client for inspecting and loading raw TUM Racetrack Database assets."""

    KNOWN_COMMIT_HASH = "e59595d1f3573b30d1ded6a08984935b957688e0"
    KNOWN_COMMIT_DATE = "2021-09-18T16:28:20+02:00"
    KNOWN_AUTHOR = "Alexander Heilmeier <alexander.heilmeier@tum.de>"
    KNOWN_LICENSE = "LGPL-3.0"

    def __init__(self, config: Optional[TUMClientConfig] = None) -> None:
        self.config = config or TUMClientConfig()
        self.repo_dir = self.config.local_repo_dir or get_tum_raw_dir()
        self.tracks_dir = self.repo_dir / "tracks"
        self.racelines_dir = self.repo_dir / "racelines"

    def ensure_repository(self) -> Path:
        """Verify the repository exists locally; attempt clone if missing and network allowed."""
        if self.is_available():
            return self.repo_dir

        if not self.config.allow_network:
            raise TUMRepositoryNotFoundError(
                f"TUM racetrack repository not found at {self.repo_dir} and allow_network=False. "
                "Please clone https://github.com/TUMFTM/racetrack-database into data/raw/tum/repo."
            )

        logger.info(f"Cloning TUM racetrack database from {self.config.repo_url} into {self.repo_dir}...")
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            cmd = ["git", "clone", "--depth", "1", self.config.repo_url, str(self.repo_dir)]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.config.timeout_s)
            logger.info("Successfully cloned TUM racetrack database repository.")
        except Exception as e:
            raise TUMRepositoryNotFoundError(
                f"Failed to clone TUM repository from {self.config.repo_url}: {e}"
            ) from e

        return self.repo_dir

    def is_available(self) -> bool:
        """Check whether local TUM tracks and racelines directories exist and contain CSV files."""
        return (
            self.tracks_dir.is_dir()
            and self.racelines_dir.is_dir()
            and len(list(self.tracks_dir.glob("*.csv"))) > 0
        )

    def get_metadata(self) -> TUMMetadata:
        """Extract git revision, license, and dataset counts from local repository."""
        commit_hash = self.KNOWN_COMMIT_HASH
        commit_date = self.KNOWN_COMMIT_DATE
        author = self.KNOWN_AUTHOR
        license_str = self.KNOWN_LICENSE

        if (self.repo_dir / ".git").exists():
            try:
                res = subprocess.run(
                    ["git", "-C", str(self.repo_dir), "log", "-1", "--format=%H|%cI|%an <%ae>"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                parts = res.stdout.strip().split("|")
                if len(parts) >= 3:
                    commit_hash = parts[0]
                    commit_date = parts[1]
                    author = parts[2]
            except Exception as e:
                logger.debug(f"Could not read git log directly: {e}. Using known commit metadata.")

        num_tracks = len(list(self.tracks_dir.glob("*.csv"))) if self.tracks_dir.exists() else 0
        num_racelines = len(list(self.racelines_dir.glob("*.csv"))) if self.racelines_dir.exists() else 0

        return TUMMetadata(
            commit_hash=commit_hash,
            commit_date=commit_date,
            author=author,
            license=license_str,
            repo_url=self.config.repo_url,
            local_path=str(self.repo_dir),
            num_tracks=num_tracks,
            num_racelines=num_racelines,
        )

    def list_tracks(self) -> List[str]:
        """Return alphabetically sorted list of all available TUM track names."""
        if not self.tracks_dir.exists():
            return []
        return sorted([f.stem for f in self.tracks_dir.glob("*.csv")])

    def get_track_info(self, track_name: str) -> TUMRawTrackInfo:
        """Retrieve file paths, point count, and size for a specific raw track."""
        track_csv = self.tracks_dir / f"{track_name}.csv"
        if not track_csv.exists():
            raise TUMTrackNotFoundError(f"Track '{track_name}' not found at {track_csv}")

        raceline_csv = self.racelines_dir / f"{track_name}.csv"
        if not raceline_csv.exists():
            raceline_csv = None

        raceline_png = self.racelines_dir / f"{track_name}_raceline.png"
        if not raceline_png.exists():
            raceline_png = None

        curvature_png = self.racelines_dir / f"{track_name}_curvature.png"
        if not curvature_png.exists():
            curvature_png = None

        # Point count from line count excluding header comment
        point_count = 0
        with open(track_csv, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("#"):
                    point_count += 1

        file_size = track_csv.stat().st_size

        return TUMRawTrackInfo(
            track_name=track_name,
            track_csv=track_csv,
            raceline_csv=raceline_csv,
            raceline_png=raceline_png,
            curvature_png=curvature_png,
            point_count=point_count,
            file_size_bytes=file_size,
        )

    def get_all_track_infos(self) -> List[TUMRawTrackInfo]:
        """Retrieve inventory metadata for all available circuits in the database."""
        tracks = self.list_tracks()
        return [self.get_track_info(t) for t in tracks]

    def load_raw_track(self, track_name: str) -> pd.DataFrame:
        """Load raw track centerline and boundary widths from CSV.
        
        Source attributes:
          - x_m: 2D Cartesian x-coordinate (meters)
          - y_m: 2D Cartesian y-coordinate (meters)
          - w_tr_right_m: track width to the right of centerline (meters)
          - w_tr_left_m: track width to the left of centerline (meters)
        """
        track_csv = self.tracks_dir / f"{track_name}.csv"
        if not track_csv.exists():
            raise TUMTrackNotFoundError(f"Raw track CSV '{track_name}.csv' does not exist in {self.tracks_dir}")

        df = pd.read_csv(
            track_csv,
            comment="#",
            header=None,
            names=["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"],
            dtype={"x_m": float, "y_m": float, "w_tr_right_m": float, "w_tr_left_m": float},
        )

        if df.empty:
            raise TUMError(f"Track file {track_csv} is empty.")
        if df.isna().any().any():
            raise TUMError(f"Track file {track_csv} contains NaN values.")

        return df

    def load_raw_raceline(self, track_name: str) -> pd.DataFrame:
        """Load raw minimum curvature race line coordinates from CSV.
        
        Source attributes:
          - x_m: 2D Cartesian x-coordinate (meters)
          - y_m: 2D Cartesian y-coordinate (meters)
        """
        raceline_csv = self.racelines_dir / f"{track_name}.csv"
        if not raceline_csv.exists():
            raise TUMTrackNotFoundError(f"Raw raceline CSV '{track_name}.csv' does not exist in {self.racelines_dir}")

        df = pd.read_csv(
            raceline_csv,
            comment="#",
            header=None,
            names=["x_m", "y_m"],
            dtype={"x_m": float, "y_m": float},
        )

        if df.empty:
            raise TUMError(f"Raceline file {raceline_csv} is empty.")
        if df.isna().any().any():
            raise TUMError(f"Raceline file {raceline_csv} contains NaN values.")

        return df
