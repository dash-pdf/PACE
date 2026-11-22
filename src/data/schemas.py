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


# Driver-Circuit Historical Intelligence Schemas
# -----------------------------------------------

# Individual event-level appearance record
DRIVER_CIRCUIT_APPEARANCES_SCHEMA = pa.schema([
    ("driver_id", pa.string()),
    ("circuit_id", pa.string()),
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("event_name", pa.string()),
    ("event_date", pa.string()),
    ("regulation_era", pa.string()),     # 'ground_effect_2022_2026', 'turbo_hybrid_2014_2021', 'v8_2006_2013', 'v10_1996_2005', 'early_pre_1996'
    ("constructor_id", pa.string()),
    ("constructor_name", pa.string()),
    ("grid_position", pa.int32()),
    ("finish_position", pa.int32()),
    ("position_text", pa.string()),
    ("positions_gained_lost", pa.int32()), # grid - finish (positive = gained, negative = lost)
    ("points", pa.float64()),
    ("laps_completed", pa.int32()),
    ("status", pa.string()),
    ("is_classified", pa.bool_()),
    ("is_dnf", pa.bool_()),
    ("is_dns", pa.bool_()),
    ("is_dsq", pa.bool_()),
    ("qualifying_position", pa.int32()),
    ("q1_time_str", pa.string()),
    ("q1_time_ms", pa.float64()),
    ("q2_time_str", pa.string()),
    ("q2_time_ms", pa.float64()),
    ("q3_time_str", pa.string()),
    ("q3_time_ms", pa.float64()),
    ("fastest_lap_rank", pa.int32()),
    ("fastest_lap_time", pa.string()),
    ("fastest_lap_speed", pa.float64()),
    ("total_race_time_ms", pa.float64()),
    ("teammate_driver_id", pa.string()),
    ("teammate_grid_position", pa.int32()),
    ("teammate_finish_position", pa.int32()),
    ("beat_teammate_race", pa.bool_()),
    ("beat_teammate_qualy", pa.bool_()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Aggregated driver-circuit performance profile
DRIVER_CIRCUIT_AGGREGATED_SCHEMA = pa.schema([
    ("driver_id", pa.string()),
    ("circuit_id", pa.string()),
    ("total_starts", pa.int32()),
    ("first_season", pa.int32()),
    ("latest_season", pa.int32()),
    ("wins", pa.int32()),
    ("podiums", pa.int32()),
    ("poles", pa.int32()),
    ("top_5_finishes", pa.int32()),
    ("top_10_finishes", pa.int32()),
    ("dnf_count", pa.int32()),
    ("dnf_rate", pa.float64()),
    ("total_points", pa.float64()),
    ("avg_points_per_start", pa.float64()),
    ("avg_grid_position", pa.float64()),
    ("avg_finish_position", pa.float64()),
    ("avg_positions_gained", pa.float64()),
    ("best_finish_position", pa.int32()),
    ("best_grid_position", pa.int32()),
    ("teammate_race_h2h_wins", pa.int32()),
    ("teammate_qualy_h2h_wins", pa.int32()),
    ("teammate_race_h2h_rate", pa.float64()),
    ("teammate_qualy_h2h_rate", pa.float64()),
    ("ground_effect_starts", pa.int32()),
    ("ground_effect_avg_finish", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Circuit characteristics and technical profile
CIRCUIT_CHARACTERISTICS_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),
    ("circuit_name", pa.string()),
    ("country", pa.string()),
    ("locality", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("first_grand_prix_year", pa.int32()),
    ("total_races_held", pa.int32()),
    ("circuit_length_km", pa.float64()),
    ("typical_race_laps", pa.int32()),
    ("overtaking_difficulty_rating", pa.string()), # 'Low', 'Medium', 'High'
    ("tyre_stress_level", pa.string()),          # 'Low', 'Medium', 'High', 'Very High'
    ("downforce_level", pa.string()),            # 'Low', 'Medium', 'High'
    ("altitude_m", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Driver Standings Schema (Championship progression)
DRIVER_STANDINGS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("driver_id", pa.string()),
    ("points", pa.float64()),
    ("position", pa.int32()),
    ("position_text", pa.string()),
    ("wins", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Constructor Standings Schema (Championship progression)
CONSTRUCTOR_STANDINGS_SCHEMA = pa.schema([
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("constructor_id", pa.string()),
    ("points", pa.float64()),
    ("position", pa.int32()),
    ("position_text", pa.string()),
    ("wins", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])


# ===========================================================================
# OpenF1 Race Intelligence Schemas
# ===========================================================================

# OpenF1 Meetings (Grand Prix events / testing)
OPENF1_MEETINGS_SCHEMA = pa.schema([
    ("meeting_key", pa.int32()),
    ("meeting_name", pa.string()),
    ("meeting_official_name", pa.string()),
    ("location", pa.string()),
    ("country_key", pa.int32()),
    ("country_code", pa.string()),
    ("country_name", pa.string()),
    ("circuit_key", pa.int32()),
    ("circuit_short_name", pa.string()),
    ("circuit_id", pa.string()),         # Canonical circuit identifier
    ("year", pa.int32()),
    ("date_start", pa.string()),
    ("date_end", pa.string()),
    ("gmt_offset", pa.string()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Sessions (Practice, Qualifying, Sprint, Race)
OPENF1_SESSIONS_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("session_name", pa.string()),
    ("session_type", pa.string()),
    ("year", pa.int32()),
    ("circuit_key", pa.int32()),
    ("circuit_short_name", pa.string()),
    ("circuit_id", pa.string()),         # Canonical circuit identifier
    ("country_name", pa.string()),
    ("location", pa.string()),
    ("date_start", pa.string()),
    ("date_end", pa.string()),
    ("gmt_offset", pa.string()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Drivers
OPENF1_DRIVERS_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),          # Canonical driver identifier
    ("broadcast_name", pa.string()),
    ("full_name", pa.string()),
    ("name_acronym", pa.string()),
    ("team_name", pa.string()),
    ("constructor_id", pa.string()),     # Canonical constructor identifier
    ("team_colour", pa.string()),
    ("first_name", pa.string()),
    ("last_name", pa.string()),
    ("country_code", pa.string()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Intervals & Gaps
OPENF1_INTERVALS_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),          # Canonical driver identifier
    ("gap_to_leader", pa.float64()),
    ("interval", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Tyre Stints
OPENF1_STINTS_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("stint_number", pa.int32()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),          # Canonical driver identifier
    ("compound", pa.string()),
    ("lap_start", pa.int32()),
    ("lap_end", pa.int32()),
    ("stint_length", pa.int32()),
    ("tyre_age_at_start", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Race Control (Flags, Safety Car, Incidents)
OPENF1_RACE_CONTROL_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("lap_number", pa.int32()),
    ("category", pa.string()),
    ("flag", pa.string()),
    ("scope", pa.string()),
    ("sector", pa.int32()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),          # Canonical driver identifier (if driver-specific)
    ("message", pa.string()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Weather
OPENF1_WEATHER_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("air_temperature", pa.float64()),
    ("track_temperature", pa.float64()),
    ("humidity", pa.float64()),
    ("pressure", pa.float64()),
    ("wind_speed", pa.float64()),
    ("wind_direction", pa.int32()),
    ("rainfall", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Overtakes
OPENF1_OVERTAKES_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("overtaking_driver_number", pa.int32()),
    ("overtaking_driver_id", pa.string()), # Canonical
    ("overtaken_driver_number", pa.int32()),
    ("overtaken_driver_id", pa.string()),  # Canonical
    ("position", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Position Changes
OPENF1_POSITION_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),
    ("position", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Pit Stops
OPENF1_PIT_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),
    ("lap_number", pa.int32()),
    ("pit_duration", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Selective Car Telemetry
OPENF1_CAR_DATA_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),
    ("speed", pa.int32()),
    ("rpm", pa.int32()),
    ("n_gear", pa.int32()),
    ("throttle", pa.int32()),
    ("brake", pa.int32()),
    ("drs", pa.int32()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# OpenF1 Car Location Coordinates
OPENF1_LOCATION_SCHEMA = pa.schema([
    ("session_key", pa.int32()),
    ("meeting_key", pa.int32()),
    ("date", pa.string()),
    ("driver_number", pa.int32()),
    ("driver_id", pa.string()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("z", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])
