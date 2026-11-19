"""Tests for Driver-Circuit Historical Intelligence module."""

import pandas as pd
import pytest

from src.data.driver_circuit import (
    get_regulation_era,
    CIRCUIT_METRICS,
    DriverCircuitIntelligence,
)
from src.data.validation import (
    validate_driver_circuit_appearances,
    validate_driver_circuit_aggregated,
    validate_circuit_characteristics,
)


def test_get_regulation_era():
    """Verify correct categorization of Formula 1 technical regulation eras."""
    assert get_regulation_era(2025) == "ground_effect_2022_2026"
    assert get_regulation_era(2022) == "ground_effect_2022_2026"
    assert get_regulation_era(2020) == "turbo_hybrid_2014_2021"
    assert get_regulation_era(2014) == "turbo_hybrid_2014_2021"
    assert get_regulation_era(2010) == "v8_2006_2013"
    assert get_regulation_era(2006) == "v8_2006_2013"
    assert get_regulation_era(2003) == "v10_1996_2005"
    assert get_regulation_era(1996) == "v10_1996_2005"
    assert get_regulation_era(1990) == "early_pre_1996"


def test_circuit_metrics_presence():
    """Verify curated metrics exist for flagship F1 circuits."""
    for c in ["monaco", "silverstone", "spa", "monza", "interlagos", "suzuka"]:
        assert c in CIRCUIT_METRICS
        assert CIRCUIT_METRICS[c]["length_km"] > 0
        assert CIRCUIT_METRICS[c]["laps"] > 0
        assert CIRCUIT_METRICS[c]["overtake"] in ("Low", "Medium", "High")


def test_validate_driver_circuit_appearances_success():
    """Verify validation passes on clean appearance records."""
    df = pd.DataFrame([{
        "driver_id": "hamilton",
        "circuit_id": "silverstone",
        "season": 2021,
        "round": 10,
        "regulation_era": "turbo_hybrid_2014_2021",
        "finish_position": 1,
        "is_classified": True,
        "is_dnf": False,
    }])
    r = validate_driver_circuit_appearances(df)
    assert r.is_valid


def test_validate_driver_circuit_appearances_duplicate():
    """Verify validation detects duplicate event appearances."""
    df = pd.DataFrame([
        {
            "driver_id": "hamilton",
            "circuit_id": "silverstone",
            "season": 2021,
            "round": 10,
            "regulation_era": "turbo_hybrid_2014_2021",
            "finish_position": 1,
            "is_classified": True,
            "is_dnf": False,
        },
        {
            "driver_id": "hamilton",
            "circuit_id": "silverstone",
            "season": 2021,
            "round": 10,
            "regulation_era": "turbo_hybrid_2014_2021",
            "finish_position": 1,
            "is_classified": True,
            "is_dnf": False,
        },
    ])
    r = validate_driver_circuit_appearances(df)
    assert not r.is_valid


def test_validate_driver_circuit_aggregated_bounds():
    """Verify validation detects invalid statistics in aggregated profiles."""
    # Valid
    df_valid = pd.DataFrame([{
        "driver_id": "hamilton",
        "circuit_id": "silverstone",
        "total_starts": 18,
        "wins": 8,
        "dnf_rate": 0.055,
    }])
    r_valid = validate_driver_circuit_aggregated(df_valid)
    assert r_valid.is_valid

    # Invalid: wins > starts or DNF rate > 1
    df_invalid = pd.DataFrame([{
        "driver_id": "hamilton",
        "circuit_id": "silverstone",
        "total_starts": 5,
        "wins": 10,
        "dnf_rate": 1.5,
    }])
    r_invalid = validate_driver_circuit_aggregated(df_invalid)
    assert not r_invalid.is_valid


def test_circuit_metrics_all_39_present():
    """Verify curated metrics exist for all 39 circuits."""
    expected_circuits = [
        "albert_park", "americas", "bahrain", "baku", "buddh", "catalunya",
        "fuji", "hockenheimring", "hungaroring", "imola", "indianapolis",
        "interlagos", "istanbul", "jeddah", "losail", "madring", "magny_cours",
        "marina_bay", "miami", "monaco", "monza", "mugello", "nurburgring",
        "portimao", "red_bull_ring", "ricard", "rodriguez", "sepang",
        "shanghai", "silverstone", "sochi", "spa", "suzuka", "valencia",
        "vegas", "villeneuve", "yas_marina", "yeongam", "zandvoort",
    ]
    for c in expected_circuits:
        assert c in CIRCUIT_METRICS
        assert CIRCUIT_METRICS[c]["length_km"] > 0
        assert CIRCUIT_METRICS[c]["laps"] > 0
        assert CIRCUIT_METRICS[c]["overtake"] in ("Low", "Medium", "High")
        assert CIRCUIT_METRICS[c]["tyre"] in ("Low", "Medium", "High", "Very High")


def test_driver_career_disk_caching(tmp_path):
    """Verify raw disk cache saves and loads without network."""
    dc = DriverCircuitIntelligence(base_dir=tmp_path)
    # Manually seed raw cache
    cache_file = dc.raw_dir / "test_driver_races.json"
    dummy_data = [{"season": 2023, "round": 1, "circuit_id": "bahrain", "driver_id": "test_driver"}]
    with open(cache_file, "w", encoding="utf-8") as f:
        import json
        json.dump(dummy_data, f)

    res = dc.fetch_driver_career_races("test_driver", force=False)
    assert len(res) == 1
    assert res[0]["circuit_id"] == "bahrain"
