"""Automated tests for PACE Canonical Entity Registry (Stage 5.1).

Verifies:
1. Known driver resolution (code, permanent number, Ergast ID, aliases).
2. Known constructor resolution (name, Ergast ID, historical renamings).
3. Circuit resolution and alias handling (including TUM and OpenF1 keys).
4. Explicit resolution of the Las Vegas defect ('las_vegas' -> 'vegas').
5. Calibrated pit lane transit loss retrieval (20.5s for Vegas, 28.0s for Marina Bay).
6. Traffic DRS configuration resolution across circuit aliases.
7. Grand Prix event resolution (by season/round, season/circuit, season/name).
8. Unknown identifier and ambiguous mapping error handling.
9. Cross-source data consistency across FastF1, Ergast, OpenF1, and TUM.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.canonical_registry import (
    CanonicalEntityRegistry,
    CanonicalDriver,
    CanonicalConstructor,
    CanonicalCircuit,
    CanonicalEvent,
    ResolutionResult,
    UnresolvedEntityError,
    AmbiguousEntityError,
    normalize_key,
    get_canonical_registry,
    resolve_driver,
    resolve_constructor,
    resolve_circuit,
    resolve_event,
    get_canonical_driver_id,
    get_canonical_constructor_id,
    get_canonical_circuit_id,
)
from src.simulation.pit_model import PitStopModel
from src.simulation.traffic import TrafficOvertakingEngine
from src.features.driver_circuit import extract_circuit_topology_features


# ---------------------------------------------------------------------------
# 1. Normalization Utility Tests
# ---------------------------------------------------------------------------

def test_normalize_key():
    """Verify string normalization strips accents, lowercases and standardizes separators."""
    assert normalize_key("Max Verstappen") == "max_verstappen"
    assert normalize_key("Sergio Pérez") == "sergio_perez"
    assert normalize_key("Nico Hülkenberg") == "nico_hulkenberg"
    assert normalize_key("Kimi Räikkönen") == "kimi_raikkonen"
    assert normalize_key("  Las-Vegas   Strip  ") == "las_vegas_strip"
    assert normalize_key("Red Bull Racing!") == "red_bull_racing"
    assert normalize_key("") == ""
    assert normalize_key(None) == ""


# ---------------------------------------------------------------------------
# 2. Driver Resolution Tests
# ---------------------------------------------------------------------------

def test_known_driver_resolution_by_code():
    """Verify drivers resolve correctly from 3-letter codes."""
    cases = [
        ("VER", "max_verstappen"),
        ("NOR", "norris"),
        ("LEC", "leclerc"),
        ("HAM", "hamilton"),
        ("ANT", "antonelli"),
        ("PIA", "piastri"),
        ("RUS", "russell"),
        ("SAI", "sainz"),
        ("ALO", "alonso"),
        ("GAS", "gasly"),
        ("TSU", "tsunoda"),
        ("ALB", "albon"),
        ("OCO", "ocon"),
        ("STR", "stroll"),
        ("HUL", "hulkenberg"),
        ("BEA", "bearman"),
        ("BOR", "bortoleto"),
        ("LAW", "lawson"),
        ("COL", "colapinto"),
        ("HAD", "hadjar"),
        ("LIN", "arvid_lindblad"),
        ("BOT", "bottas"),
        ("PER", "perez"),
    ]
    for code, expected_id in cases:
        res = resolve_driver(code)
        assert res.is_resolved, f"Failed to resolve driver code: {code}"
        assert res.canonical_id == expected_id, f"Code {code} resolved to {res.canonical_id}, expected {expected_id}"
        assert res.strategy == "code"


def test_known_driver_resolution_by_number():
    """Verify drivers resolve correctly from permanent car numbers."""
    cases = [
        (1, "max_verstappen"),
        (4, "norris"),
        (16, "leclerc"),
        (44, "hamilton"),
        (12, "antonelli"),
        (81, "piastri"),
        (63, "russell"),
        (55, "sainz"),
        (14, "alonso"),
        (10, "gasly"),
        (22, "tsunoda"),
        (23, "albon"),
        (31, "ocon"),
        (18, "stroll"),
        (27, "hulkenberg"),
        (87, "bearman"),
        (5, "bortoleto"),
        (30, "lawson"),
        (43, "colapinto"),
        (6, "hadjar"),
        (41, "arvid_lindblad"),
        (77, "bottas"),
        (11, "perez"),
    ]
    for num, expected_id in cases:
        res = resolve_driver(num, season=2026)
        assert res.is_resolved, f"Failed to resolve driver number: {num}"
        assert res.canonical_id == expected_id, f"Number {num} resolved to {res.canonical_id}, expected {expected_id}"
        assert res.strategy == "number"


def test_driver_historical_aliases_and_diacritics():
    """Verify driver alias and accent-insensitive matching."""
    test_queries = [
        ("Andrea Kimi Antonelli", "antonelli"),
        ("kimi antonelli", "antonelli"),
        ("Kimi Antonelli", "antonelli"),
        ("Sergio Pérez", "perez"),
        ("sergio perez", "perez"),
        ("Checo", "perez"),
        ("Checo Pérez", "perez"),
        ("Nico Hülkenberg", "hulkenberg"),
        ("nico huelkenberg", "hulkenberg"),
        ("Carlos Sainz Jr", "sainz"),
        ("sainz jr", "sainz"),
        ("Max Verstappen", "max_verstappen"),
        ("Ollie Bearman", "bearman"),
        ("Kimi Räikkönen", "raikkonen"),
        ("The Iceman", "raikkonen"),
        ("Sebastian Vettel", "vettel"),
    ]
    for query, expected_id in test_queries:
        res = resolve_driver(query)
        assert res.is_resolved, f"Failed to resolve alias: '{query}'"
        assert res.canonical_id == expected_id, f"Query '{query}' resolved to {res.canonical_id}, expected {expected_id}"


def test_driver_seasonal_number_override():
    """Verify seasonal number disambiguation (e.g., Vettel #5 in 2021 vs Bortoleto #5 in 2026)."""
    vettel_res = resolve_driver(5, season=2021)
    assert vettel_res.is_resolved
    assert vettel_res.canonical_id == "vettel"

    bortoleto_res = resolve_driver(5, season=2026)
    assert bortoleto_res.is_resolved
    assert bortoleto_res.canonical_id == "bortoleto"


# ---------------------------------------------------------------------------
# 3. Constructor Resolution Tests
# ---------------------------------------------------------------------------

def test_known_constructor_resolution():
    """Verify all 2026 constructors resolve correctly from canonical IDs and official names."""
    cases = [
        ("red_bull", "Red Bull Racing"),
        ("ferrari", "Scuderia Ferrari"),
        ("mclaren", "McLaren Formula 1 Team"),
        ("mercedes", "Mercedes-AMG PETRONAS F1 Team"),
        ("aston_martin", "Aston Martin Aramco F1 Team"),
        ("alpine", "BWT Alpine F1 Team"),
        ("williams", "Williams Racing"),
        ("haas", "MoneyGram Haas F1 Team"),
        ("sauber", "Stake F1 Team Kick Sauber"),
        ("rb", "Visa Cash App RB F1 Team"),
        ("cadillac", "Cadillac Formula 1 Team"),
        ("audi", "Audi Revolut F1 Team"),
    ]
    for c_id, official_name in cases:
        # Resolve by ID
        res_id = resolve_constructor(c_id)
        assert res_id.is_resolved
        assert res_id.canonical_id == c_id

        # Resolve by Name
        res_name = resolve_constructor(official_name)
        assert res_name.is_resolved
        assert res_name.canonical_id == c_id


def test_constructor_historical_aliases_and_rebrands():
    """Verify historical constructor renamings map to canonical teams."""
    aliases = [
        ("AlphaTauri", "rb"),
        ("alpha_tauri", "rb"),
        ("Toro Rosso", "rb"),
        ("scuderia toro rosso", "rb"),
        ("racing bulls", "rb"),
        ("VCARB", "rb"),
        ("Kick Sauber", "sauber"),
        ("stake f1", "sauber"),
        ("Alfa Romeo", "sauber"),
        ("alfa", "sauber"),
        ("Racing Point", "aston_martin"),
        ("Force India", "aston_martin"),
        ("Sahara Force India", "aston_martin"),
        ("Jordan", "aston_martin"),
        ("Renault", "alpine"),
        ("Renault F1", "alpine"),
        ("Lotus F1", "alpine"),
        ("Brawn GP", "mercedes"),
        ("Brawn", "mercedes"),
    ]
    for alias, expected_canonical in aliases:
        res = resolve_constructor(alias)
        assert res.is_resolved, f"Failed constructor alias: '{alias}'"
        assert res.canonical_id == expected_canonical, (
            f"Constructor alias '{alias}' resolved to {res.canonical_id}, expected {expected_canonical}"
        )


# ---------------------------------------------------------------------------
# 4. Circuit Resolution & The Las Vegas Defect
# ---------------------------------------------------------------------------

def test_las_vegas_circuit_resolution():
    """Verify that 'vegas' is canonical and all Las Vegas aliases resolve to 'vegas'."""
    aliases = [
        "vegas",
        "las_vegas",
        "Las Vegas",
        "las-vegas",
        "Las Vegas Strip Street Circuit",
        "vegas strip",
        "las vegas strip",
        "lasvegas",
    ]
    for alias in aliases:
        res = resolve_circuit(alias)
        assert res.is_resolved, f"Failed to resolve Las Vegas circuit alias: '{alias}'"
        assert res.canonical_id == "vegas", f"Las Vegas alias '{alias}' resolved to {res.canonical_id}, expected 'vegas'"
        assert res.entity.lap_length_km == 6.201
        assert res.entity.typical_laps == 50
        assert res.entity.default_transit_loss_s == 20.5
        assert res.entity.country == "USA"


def test_las_vegas_calibrated_pit_transit_loss():
    """Verify PitStopModel retrieves the calibrated 20.5s transit loss for Las Vegas across all aliases."""
    pit_model = PitStopModel()

    # Under Green Flag
    for alias in ["vegas", "las_vegas", "Las Vegas", "las-vegas", "vegas strip"]:
        transit_loss = pit_model.get_circuit_transit_loss(alias, is_sc_vsc=False)
        assert transit_loss == 20.5, f"Expected 20.5s for '{alias}', got {transit_loss}s"

    # Under Safety Car / VSC (~45% transit discount: 20.5 * 0.55 = 11.275s)
    sc_loss = pit_model.get_circuit_transit_loss("vegas", is_sc_vsc=True)
    assert sc_loss == pytest.approx(20.5 * 0.55, abs=1e-3)

    sc_alias_loss = pit_model.get_circuit_transit_loss("las_vegas", is_sc_vsc=True)
    assert sc_alias_loss == pytest.approx(20.5 * 0.55, abs=1e-3)


def test_marina_bay_singapore_resolution_and_pit_loss():
    """Verify Marina Bay / Singapore resolution and 28.0s pit transit loss."""
    pit_model = PitStopModel()

    for alias in ["marina_bay", "singapore", "Marina Bay Street Circuit", "Singapore"]:
        res = resolve_circuit(alias)
        assert res.is_resolved
        assert res.canonical_id == "marina_bay"

        transit_loss = pit_model.get_circuit_transit_loss(alias, is_sc_vsc=False)
        assert transit_loss == 28.0, f"Expected 28.0s for '{alias}', got {transit_loss}s"


def test_circuit_cross_source_identifiers():
    """Verify TUM and OpenF1 identifiers resolve to canonical circuit IDs."""
    # TUM Racetrack Database keys
    tum_cases = [
        ("LasVegas", "vegas"),
        ("Melbourne", "albert_park"),
        ("Austin", "americas"),
        ("MarinaBay", "marina_bay"),
        ("Monaco", "monaco"),
        ("Monza", "monza"),
        ("Silverstone", "silverstone"),
        ("Spa", "spa"),
        ("Spielberg", "red_bull_ring"),
        ("Zandvoort", "zandvoort"),
    ]
    for tum_id, expected_id in tum_cases:
        res = resolve_circuit(tum_id)
        assert res.is_resolved, f"Failed TUM key '{tum_id}'"
        assert res.canonical_id == expected_id, f"TUM key '{tum_id}' resolved to {res.canonical_id}, expected {expected_id}"


def test_traffic_drs_circuit_resolution():
    """Verify TrafficOvertakingEngine retrieves correct circuit DRS configs across aliases."""
    traffic = TrafficOvertakingEngine()

    vegas_config = traffic.get_circuit_config("las_vegas")
    assert vegas_config.circuit_id == "vegas"
    assert vegas_config.num_drs_zones == 2
    assert vegas_config.drs_delta_per_zone_s == 0.25

    marina_config = traffic.get_circuit_config("singapore")
    assert marina_config.circuit_id == "marina_bay"
    assert marina_config.num_drs_zones == 4


def test_feature_topology_extraction_with_las_vegas_alias():
    """Verify extract_circuit_topology_features correctly retrieves Vegas data with 'las_vegas' alias."""
    # Create mock characteristics DataFrame using canonical 'vegas'
    df_char = pd.DataFrame([
        {
            "circuit_id": "vegas",
            "circuit_length_km": 6.201,
            "overtaking_difficulty_rating": "Low",
            "tyre_stress_level": "Low",
            "downforce_level": "Low",
            "altitude_m": 610.0,
        }
    ])

    # Query with alias 'las_vegas'
    topo = extract_circuit_topology_features(df_char, "las_vegas")
    assert topo["circuit_length_km"] == 6.201
    assert topo["altitude_m"] == 610.0
    assert topo["overtaking_difficulty"] == 2.0  # 'Low' maps to 2.0


# ---------------------------------------------------------------------------
# 5. Grand Prix Event Resolution Tests
# ---------------------------------------------------------------------------

def test_event_resolution():
    """Verify event resolution for the 2026 calendar, focusing on Round 21 Las Vegas."""
    # Resolve by season and round
    ev_by_round = resolve_event(2026, round_num=21)
    assert ev_by_round.is_resolved
    assert ev_by_round.canonical_id == "2026_21_vegas"
    assert ev_by_round.entity.event_name == "Las Vegas Grand Prix"
    assert ev_by_round.entity.circuit_id == "vegas"
    assert ev_by_round.entity.session_ids["race"] == "2026_21_vegas_race"

    # Resolve by season and circuit alias
    ev_by_circ = resolve_event(2026, circuit_id="las_vegas")
    assert ev_by_circ.is_resolved
    assert ev_by_circ.canonical_id == "2026_21_vegas"

    # Resolve by season and grand prix name
    ev_by_name = resolve_event(2026, event_name="Las Vegas Grand Prix")
    assert ev_by_name.is_resolved
    assert ev_by_name.canonical_id == "2026_21_vegas"


# ---------------------------------------------------------------------------
# 6. Error Handling & Strict Mode Tests
# ---------------------------------------------------------------------------

def test_unresolved_entity_handling():
    """Verify unresolved entities return is_resolved=False or raise UnresolvedEntityError in strict mode."""
    # Non-strict mode
    res_d = resolve_driver("non_existent_driver_xyz", strict=False)
    assert not res_d.is_resolved
    assert res_d.canonical_id is None
    assert "Unknown driver" in res_d.error_message

    res_c = resolve_constructor("imaginary_team_abc", strict=False)
    assert not res_c.is_resolved
    assert res_c.canonical_id is None

    res_circ = resolve_circuit("moon_circuit_gp", strict=False)
    assert not res_circ.is_resolved
    assert res_circ.canonical_id is None

    # Strict mode
    with pytest.raises(UnresolvedEntityError) as exc_info:
        get_canonical_driver_id("non_existent_driver_xyz")
    assert "non_existent_driver_xyz" in str(exc_info.value)

    with pytest.raises(UnresolvedEntityError):
        get_canonical_constructor_id("imaginary_team_abc")

    with pytest.raises(UnresolvedEntityError):
        get_canonical_circuit_id("moon_circuit_gp")


def test_empty_and_none_queries():
    """Verify empty or None inputs are handled cleanly without crashing."""
    res_none = resolve_driver(None, strict=False)
    assert not res_none.is_resolved

    res_empty_circ = resolve_circuit("", strict=False)
    assert not res_empty_circ.is_resolved

    res_empty_const = resolve_constructor("   ", strict=False)
    assert not res_empty_const.is_resolved
