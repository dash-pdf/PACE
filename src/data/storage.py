"""Storage utilities for PACE datasets.

Handles partitioning, Parquet persistence, metadata preservation, and
dataset compilation across raw, processed, and historical layers.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Base directory paths
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = DEFAULT_BASE_DIR / "data" / "raw"
PROCESSED_DIR = DEFAULT_BASE_DIR / "data" / "processed"
HISTORICAL_DIR = DEFAULT_BASE_DIR / "data" / "historical"


def ensure_directories(base_dir: Optional[Path] = None) -> None:
    """Create data directories if they do not exist."""
    base = base_dir or DEFAULT_BASE_DIR
    for sub in ["raw", "processed", "historical", "historical/driver_circuit"]:
        d = base / "data" / sub
        d.mkdir(parents=True, exist_ok=True)


def get_driver_circuit_dir(base_dir: Optional[Path] = None) -> Path:
    """Return path to driver-circuit intelligence directory."""
    base = base_dir or DEFAULT_BASE_DIR
    p = base / "data" / "historical" / "driver_circuit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_driver_circuit_path(filename: str, base_dir: Optional[Path] = None) -> Path:
    """Return destination path for driver-circuit intelligence file."""
    return get_driver_circuit_dir(base_dir=base_dir) / filename


def get_raw_path(category: str, season: int, filename: str, base_dir: Optional[Path] = None) -> Path:
    """Return destination path for raw data file."""
    base = base_dir or DEFAULT_BASE_DIR
    p = base / "data" / "raw" / str(season) / category
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def get_processed_path(category: str, season: int, base_dir: Optional[Path] = None) -> Path:
    """Return destination path for processed Parquet file by season."""
    base = base_dir or DEFAULT_BASE_DIR
    p = base / "data" / "processed" / str(season)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{category}_{season}.parquet"


def get_historical_path(category: str, base_dir: Optional[Path] = None) -> Path:
    """Return destination path for compiled multi-season historical dataset."""
    base = base_dir or DEFAULT_BASE_DIR
    p = base / "data" / "historical"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"historical_{category}.parquet"


def save_raw_json(data: Any, category: str, season: int, filename: str, base_dir: Optional[Path] = None) -> Path:
    """Save raw response as JSON."""
    path = get_raw_path(category, season, filename, base_dir=base_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def save_parquet(
    df: pd.DataFrame,
    destination: Path,
    schema: Optional[pa.Schema] = None,
    compression: str = "snappy",
) -> Path:
    """Write DataFrame to Parquet with schema enforcement and compression."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if schema is not None:
        if df.empty:
            table = pa.Table.from_batches([], schema=schema)
        else:
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_table(table, destination, compression=compression)
    logger.debug(f"Saved {len(df)} rows to {destination}")
    return destination


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {path}")
    return pq.read_table(path).to_pandas()


def dataset_exists(path: Path) -> bool:
    """Check whether a dataset file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def compile_historical_dataset(
    category: str,
    seasons: List[int],
    schema: Optional[pa.Schema] = None,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Concatenate processed seasonal Parquet datasets into a master historical file."""
    dfs: List[pd.DataFrame] = []
    for s in seasons:
        path = get_processed_path(category, s, base_dir=base_dir)
        if dataset_exists(path):
            dfs.append(load_parquet(path))

    if not dfs:
        logger.warning(f"No processed data found for category '{category}' across seasons {seasons}")
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate canonical reference tables across seasons
    if category == "drivers" and "driver_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["driver_id"], keep="last")
    elif category == "constructors" and "constructor_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["constructor_id"], keep="last")
    elif category == "circuits" and "circuit_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["circuit_id"], keep="last")
    elif category == "events" and {"season", "round"}.issubset(combined_df.columns):
        combined_df = combined_df.drop_duplicates(subset=["season", "round"], keep="last")

    target_path = get_historical_path(category, base_dir=base_dir)
    save_parquet(combined_df, target_path, schema=schema)
    logger.info(f"Compiled historical '{category}' dataset ({len(combined_df)} rows) at {target_path}")
    return target_path
