"""Data schemas and type specifications for the PACE historical data infrastructure.

Defines schemas for drivers, constructors, circuits, events, sessions, qualifying,
race results, sprint results, lap times, pit stops, and practice sessions.
"""

from typing import Dict, Any
import pyarrow as pa


# Driver Schema
DRIVERS_SCHEMA = pa.schema([
    ("driver_id", pa.string()),          # Canonical ID (e.g., 'max_verstappen')
    ("driver_code", pa.string()),        # 3-letter code (e.g., 'VER')
    ("driver_number", pa.int32()),       # Permanent car number (e.g., 1)
    ("given_name", pa.string()),         # First name
    ("family_name", pa.string()),        # Last name
    ("full_name", pa.string()),          # Combined display name
    ("date_of_birth", pa.string()),      # YYYY-MM-DD
    ("nationality", pa.string()),        # Driver nationality
    ("driver_url", pa.string()),         # Biographical URL reference
    ("source_api", pa.string()),         # Data origin
    ("retrieved_at", pa.string()),       # ISO 8601 timestamp
])

# Constructor Schema
CONSTRUCTORS_SCHEMA = pa.schema([
    ("constructor_id", pa.string()),    # Canonical ID (e.g., 'red_bull')
    ("constructor_name", pa.string()),  # Display name (e.g., 'Red Bull')
    ("nationality", pa.string()),       # Country of license
    ("constructor_url", pa.string()),   # Reference URL
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Circuit Schema
CIRCUITS_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),        # Canonical ID (e.g., 'bahrain')
    ("circuit_name", pa.string()),      # Official circuit title
    ("locality", pa.string()),          # City/locality
    ("country", pa.string()),           # Country
    ("latitude", pa.float64()),         # Latitude coordinate
    ("longitude", pa.float64()),        # Longitude coordinate
    ("circuit_url", pa.string()),       # Reference URL
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Event Schedule Schema
EVENTS_SCHEMA = pa.schema([
    ("season", pa.int32()),             # Championship season (e.g., 2023)
    ("round", pa.int32()),              # Calendar round number
    ("event_name", pa.string()),        # Grand Prix name
    ("circuit_id", pa.string()),        # Linked circuit key
    ("country", pa.string()),           # Host country
    ("event_date", pa.string()),        # Date of Sunday Grand Prix (YYYY-MM-DD)
    ("event_url", pa.string()),         # Ergast / F1 race URL
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Session Metadata Schema
SESSIONS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("session_type", pa.string()),      # FP1, FP2, FP3, Qualifying, Sprint, Race
    ("session_name", pa.string()),
    ("session_date", pa.string()),      # YYYY-MM-DD
    ("session_time_utc", pa.string()),  # HH:MM:SSZ if available
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Race Results Schema
RACE_RESULTS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("driver_id", pa.string()),
    ("constructor_id", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_code", pa.string()),
    ("grid_position", pa.int32()),      # Starting grid position
    ("finish_position", pa.int32()),    # Final classified finishing position (None if unclassified/DNS)
    ("position_text", pa.string()),     # Raw text ('1', 'R', 'D', 'W')
    ("points", pa.float64()),           # Championship points awarded
    ("laps_completed", pa.int32()),     # Completed race distance in laps
    ("status", pa.string()),            # Finishing or retirement reason
    ("is_classified", pa.bool_()),      # True if officially classified in results
    ("is_dnf", pa.bool_()),             # True if Did Not Finish
    ("is_dns", pa.bool_()),             # True if Did Not Start
    ("is_dsq", pa.bool_()),             # True if Disqualified
    ("total_time_ms", pa.float64()),    # Total race duration in ms (if available)
    ("total_time_str", pa.string()),    # Human-readable elapsed time or gap
    ("fastest_lap_rank", pa.int32()),   # Rank of driver's fastest lap
    ("fastest_lap_time", pa.string()),  # Fastest lap string MM:SS.sss
    ("fastest_lap_speed", pa.float64()),# Fastest lap average speed (kph)
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Qualifying Results Schema
QUALIFYING_RESULTS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("driver_id", pa.string()),
    ("constructor_id", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_code", pa.string()),
    ("qualifying_position", pa.int32()),
    ("q1_time_str", pa.string()),
    ("q1_time_ms", pa.float64()),
    ("q2_time_str", pa.string()),
    ("q2_time_ms", pa.float64()),
    ("q3_time_str", pa.string()),
    ("q3_time_ms", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Sprint Results Schema
SPRINT_RESULTS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("driver_id", pa.string()),
    ("constructor_id", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_code", pa.string()),
    ("grid_position", pa.int32()),
    ("finish_position", pa.int32()),
    ("position_text", pa.string()),
    ("points", pa.float64()),
    ("laps_completed", pa.int32()),
    ("status", pa.string()),
    ("is_classified", pa.bool_()),
    ("is_dnf", pa.bool_()),
    ("is_dns", pa.bool_()),
    ("is_dsq", pa.bool_()),
    ("total_time_ms", pa.float64()),
    ("total_time_str", pa.string()),
    ("fastest_lap_number", pa.int32()),
    ("fastest_lap_time", pa.string()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Lap Times Schema (Race Laps)
LAPS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("lap_number", pa.int32()),
    ("driver_id", pa.string()),
    ("position", pa.int32()),
    ("lap_time_str", pa.string()),      # e.g., '1:34.567'
    ("lap_time_ms", pa.float64()),      # Duration in milliseconds
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Pit Stops Schema
PIT_STOPS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("driver_id", pa.string()),
    ("stop_number", pa.int32()),
    ("lap_number", pa.int32()),
    ("pit_time_str", pa.string()),      # Time of day (HH:MM:SS)
    ("duration_str", pa.string()),      # Pit stop duration (e.g., '24.123')
    ("duration_ms", pa.float64()),      # Duration in milliseconds
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Practice Session Laps Schema (from OpenF1 / Timing Feeds)
PRACTICE_LAPS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("session_type", pa.string()),      # FP1, FP2, FP3
    ("driver_number", pa.int32()),
    ("lap_number", pa.int32()),
    ("lap_time_seconds", pa.float64()),
    ("sector_1_seconds", pa.float64()),
    ("sector_2_seconds", pa.float64()),
    ("sector_3_seconds", pa.float64()),
    ("is_pit_out_lap", pa.bool_()),
    ("i1_speed", pa.float64()),
    ("i2_speed", pa.float64()),
    ("st_speed", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])
