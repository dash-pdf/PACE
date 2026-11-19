"""Tests for PACE schema definitions."""

import pyarrow as pa
from src.data.schemas import (
    DRIVERS_SCHEMA,
    CONSTRUCTORS_SCHEMA,
    CIRCUITS_SCHEMA,
    EVENTS_SCHEMA,
    RACE_RESULTS_SCHEMA,
    QUALIFYING_RESULTS_SCHEMA,
    SPRINT_RESULTS_SCHEMA,
    LAPS_SCHEMA,
    PIT_STOPS_SCHEMA,
    PRACTICE_LAPS_SCHEMA,
)


def test_schema_field_names():
    """Verify primary key fields exist in each schema."""
    assert "driver_id" in DRIVERS_SCHEMA.names
    assert "constructor_id" in CONSTRUCTORS_SCHEMA.names
    assert "circuit_id" in CIRCUITS_SCHEMA.names
    assert "season" in EVENTS_SCHEMA.names and "round" in EVENTS_SCHEMA.names
    assert "finish_position" in RACE_RESULTS_SCHEMA.names
    assert "qualifying_position" in QUALIFYING_RESULTS_SCHEMA.names
    assert "lap_number" in LAPS_SCHEMA.names
    assert "stop_number" in PIT_STOPS_SCHEMA.names
    assert "session_type" in PRACTICE_LAPS_SCHEMA.names


def test_race_results_schema_types():
    """Verify data types for critical race results fields."""
    assert RACE_RESULTS_SCHEMA.field("finish_position").type == pa.int32()
    assert RACE_RESULTS_SCHEMA.field("points").type == pa.float64()
    assert RACE_RESULTS_SCHEMA.field("is_classified").type == pa.bool_()
    assert RACE_RESULTS_SCHEMA.field("is_dnf").type == pa.bool_()
