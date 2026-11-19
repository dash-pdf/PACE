"""Tests for storage layer and Parquet persistence."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.storage import (
    ensure_directories,
    save_parquet,
    load_parquet,
    dataset_exists,
    compile_historical_dataset,
    get_processed_path,
    get_historical_path,
)
from src.data.schemas import DRIVERS_SCHEMA


def test_ensure_directories(tmp_path: Path):
    """Verify data directory generation."""
    ensure_directories(tmp_path)
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "processed").exists()
    assert (tmp_path / "data" / "historical").exists()


def test_parquet_save_and_load(tmp_path: Path):
    """Verify saving and loading dataframe with PyArrow schema."""
    df = pd.DataFrame([{
        "driver_id": "norris",
        "driver_code": "NOR",
        "driver_number": 4,
        "given_name": "Lando",
        "family_name": "Norris",
        "full_name": "Lando Norris",
        "date_of_birth": "1999-11-13",
        "nationality": "British",
        "driver_url": "https://en.wikipedia.org/wiki/Lando_Norris",
        "source_api": "test",
        "retrieved_at": "2026-01-01T00:00:00Z",
    }])

    dest = tmp_path / "test_drivers.parquet"
    save_parquet(df, dest, schema=DRIVERS_SCHEMA)
    assert dataset_exists(dest)

    loaded_df = load_parquet(dest)
    assert len(loaded_df) == 1
    assert loaded_df["driver_id"].iloc[0] == "norris"
    assert loaded_df["driver_number"].iloc[0] == 4


def test_compile_historical_dataset(tmp_path: Path):
    """Verify multi-season compilation."""
    df_2023 = pd.DataFrame([{
        "driver_id": "verstappen",
        "driver_code": "VER",
        "driver_number": 1,
        "given_name": "Max",
        "family_name": "Verstappen",
        "full_name": "Max Verstappen",
        "date_of_birth": "1997-09-30",
        "nationality": "Dutch",
        "driver_url": "url",
        "source_api": "test",
        "retrieved_at": "2026-01-01",
    }])
    df_2024 = pd.DataFrame([{
        "driver_id": "leclerc",
        "driver_code": "LEC",
        "driver_number": 16,
        "given_name": "Charles",
        "family_name": "Leclerc",
        "full_name": "Charles Leclerc",
        "date_of_birth": "1997-10-16",
        "nationality": "Monegasque",
        "driver_url": "url",
        "source_api": "test",
        "retrieved_at": "2026-01-01",
    }])

    p23 = get_processed_path("drivers", 2023, base_dir=tmp_path)
    p24 = get_processed_path("drivers", 2024, base_dir=tmp_path)
    save_parquet(df_2023, p23, schema=DRIVERS_SCHEMA)
    save_parquet(df_2024, p24, schema=DRIVERS_SCHEMA)

    compiled = compile_historical_dataset("drivers", [2023, 2024], schema=DRIVERS_SCHEMA, base_dir=tmp_path)
    assert compiled is not None
    assert dataset_exists(compiled)

    loaded = load_parquet(compiled)
    assert len(loaded) == 2
    assert set(loaded["driver_id"]) == {"verstappen", "leclerc"}
