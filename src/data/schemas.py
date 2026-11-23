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

# ---------------------------------------------------------------------------
# TUM Racetrack Database Schemas
# ---------------------------------------------------------------------------

# Normalized Track Centerline, Boundaries and Derived Geometry
TUM_TRACK_GEOMETRY_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),            # Canonical circuit ID (e.g. 'monza', 'americas')
    ("tum_circuit_name", pa.string()),      # Original TUM filename stem (e.g. 'Monza', 'Austin')
    ("point_index", pa.int32()),            # Continuous sequence 0, 1, ... N-1
    ("s_m", pa.float64()),                  # Derived cumulative centerline distance (meters)
    ("x_m", pa.float64()),                  # Source 2D Cartesian x coordinate (meters)
    ("y_m", pa.float64()),                  # Source 2D Cartesian y coordinate (meters)
    ("w_tr_right_m", pa.float64()),         # Source track width right of centerline (meters)
    ("w_tr_left_m", pa.float64()),          # Source track width left of centerline (meters)
    ("w_total_m", pa.float64()),            # Derived total track width (meters)
    ("psi_rad", pa.float64()),              # Derived heading angle (radians)
    ("kappa_radpm", pa.float64()),          # Derived curvature (rad/m)
    ("radius_m", pa.float64()),             # Derived curvature radius = 1/|kappa| (meters)
    ("source_api", pa.string()),            # 'tum'
    ("retrieved_at", pa.string()),
])

# Minimum Curvature Raceline
TUM_RACELINE_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),            # Canonical circuit ID
    ("tum_circuit_name", pa.string()),
    ("point_index", pa.int32()),
    ("s_m", pa.float64()),                  # Derived cumulative raceline distance (meters)
    ("x_m", pa.float64()),                  # Source 2D Cartesian x coordinate (meters)
    ("y_m", pa.float64()),                  # Source 2D Cartesian y coordinate (meters)
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# Overall Circuit Geometry and Classification Summary
TUM_CIRCUIT_SUMMARY_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),            # Canonical circuit ID (or TUM name if unmapped)
    ("tum_circuit_name", pa.string()),      # Raw TUM identifier
    ("is_canonical_f1", pa.bool_()),        # Whether circuit maps to canonical F1 championship calendar
    ("num_centerline_points", pa.int32()),
    ("num_raceline_points", pa.int32()),
    ("track_length_m", pa.float64()),       # Derived closed centerline length (meters)
    ("raceline_length_m", pa.float64()),    # Derived closed raceline length (meters)
    ("closure_error_m", pa.float64()),      # Euclidean gap from final point to initial point (meters)
    ("min_width_m", pa.float64()),          # Minimum track width (meters)
    ("max_width_m", pa.float64()),          # Maximum track width (meters)
    ("mean_width_m", pa.float64()),         # Mean track width (meters)
    ("num_corners_detected", pa.int32()),   # Inferred corner count
    ("num_low_speed_corners", pa.int32()),  # Inferred (R_apex < 75m)
    ("num_medium_speed_corners", pa.int32()), # Inferred (75m <= R_apex <= 150m)
    ("num_high_speed_corners", pa.int32()), # Inferred (R_apex > 150m)
    ("longest_straight_m", pa.float64()),   # Inferred longest straight segment (meters)
    ("low_speed_pct", pa.float64()),        # Inferred percentage of lap in low speed corners
    ("medium_speed_pct", pa.float64()),     # Inferred percentage of lap in medium speed corners
    ("high_speed_pct", pa.float64()),       # Inferred percentage of lap in high speed corners
    ("straight_pct", pa.float64()),         # Inferred percentage of lap in straights
    ("source_commit", pa.string()),         # Source git commit hash
    ("source_license", pa.string()),        # 'LGPL-3.0'
    ("retrieved_at", pa.string()),
])

# Inferred Corner Profiles
TUM_CORNER_PROFILE_SCHEMA = pa.schema([
    ("circuit_id", pa.string()),
    ("corner_number", pa.int32()),          # 1, 2, ...
    ("turn_direction", pa.string()),        # 'left', 'right'
    ("speed_category", pa.string()),        # 'low', 'medium', 'high'
    ("start_s_m", pa.float64()),
    ("apex_s_m", pa.float64()),
    ("end_s_m", pa.float64()),
    ("corner_length_m", pa.float64()),
    ("apex_radius_m", pa.float64()),
    ("apex_curvature_radpm", pa.float64()),
    ("min_track_width_m", pa.float64()),
    ("source_api", pa.string()),
    ("retrieved_at", pa.string()),
])

# ---------------------------------------------------------------------------
# Stage 5.6: Cross-Source Integration & Reconciliation Schemas
# ---------------------------------------------------------------------------

# Unified Grand Prix Event Schema
UNIFIED_EVENTS_SCHEMA = pa.schema([
    ("canonical_event_id", pa.string()),    # Format: '{season}_{round:02d}_{circuit_id}'
    ("season", pa.int32()),                 # Championship year
    ("round", pa.int32()),                  # Championship round
    ("official_name", pa.string()),         # Official title (e.g. 'Formula 1 Heineken Silver Las Vegas Grand Prix 2024')
    ("event_name", pa.string()),            # Display title (e.g. 'Las Vegas Grand Prix')
    ("circuit_id", pa.string()),            # Canonical circuit ID
    ("circuit_layout_id", pa.string()),     # Versioned circuit layout ID (e.g. 'vegas:2023_present')
    ("country", pa.string()),               # Host country
    ("locality", pa.string()),              # City / locality
    ("event_date", pa.string()),            # Sunday race date (YYYY-MM-DD)
    ("has_sprint", pa.bool_()),             # True if Sprint weekend
    ("source_event_keys_json", pa.string()),# JSON serialized source mapping keys
    ("source_apis", pa.string()),           # Comma-separated contributing sources
    ("retrieved_at", pa.string()),
])

# Unified Event Session Schema
UNIFIED_SESSIONS_SCHEMA = pa.schema([
    ("canonical_session_id", pa.string()),  # Format: '{season}_{round:02d}_{session_type}'
    ("canonical_event_id", pa.string()),    # Linked event ID
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("session_type", pa.string()),          # 'fp1', 'fp2', 'fp3', 'qualifying', 'sprint_qualifying', 'sprint', 'race'
    ("session_name", pa.string()),          # Full session name
    ("circuit_id", pa.string()),
    ("circuit_layout_id", pa.string()),
    ("start_time_utc", pa.string()),        # ISO-8601 UTC timestamp
    ("end_time_utc", pa.string()),          # ISO-8601 UTC timestamp
    ("original_tz", pa.string()),           # Original offset or timezone name if available
    ("original_start_time", pa.string()),   # Unaltered source timestamp string
    ("status", pa.string()),                # 'SCHEDULED', 'COMPLETED', 'CANCELLED', 'ABANDONED'
    ("source_session_keys_json", pa.string()),# JSON serialized source keys (e.g. openf1 session_key)
    ("retrieved_at", pa.string()),
])

# Cross-Source Discrepancy & Reconciliation Record Schema
CROSS_SOURCE_RECONCILIATION_SCHEMA = pa.schema([
    ("domain", pa.string()),                # 'results', 'standings', 'timing', 'stints', 'weather', 'geometry'
    ("entity_id", pa.string()),             # e.g., driver_id, constructor_id, circuit_id
    ("season", pa.int32()),
    ("round", pa.int32()),
    ("session_type", pa.string()),
    ("field_name", pa.string()),            # e.g., 'finish_position', 'lap_time_ms', 'points'
    ("source_a", pa.string()),              # Primary/first source
    ("val_a_str", pa.string()),
    ("source_b", pa.string()),              # Comparator/secondary source
    ("val_b_str", pa.string()),
    ("discrepancy_type", pa.string()),      # 'EXACT_MATCH', 'EQUIVALENT_CONVERTED', 'DIFFERENT_DEFINITION', etc.
    ("absolute_delta", pa.float64()),       # Numerical delta if applicable
    ("tolerance_applied", pa.float64()),    # Numerical tolerance threshold
    ("selected_source", pa.string()),       # Winning authoritative source
    ("selection_reason", pa.string()),      # Documented rationale
    ("recorded_at", pa.string()),
])

# Multi-Dimensional Data Quality Scorecard Schema
DATA_QUALITY_SCORECARD_SCHEMA = pa.schema([
    ("source_name", pa.string()),           # 'fastf1', 'kaggle', 'jolpica', 'openf1', 'tum'
    ("dataset_category", pa.string()),      # 'race_results', 'driver_standings', 'laps', 'geometry', etc.
    ("records_evaluated", pa.int64()),
    ("completeness_pct", pa.float64()),     # Non-null required fields
    ("consistency_pct", pa.float64()),      # Cross-field / relational integrity
    ("validity_pct", pa.float64()),         # Value range / domain constraint adherence
    ("reconciliation_coverage_pct", pa.float64()), # Overlap percentage with secondary source
    ("unresolved_count", pa.int64()),       # Unresolved identities or conflicts
    ("evaluated_at", pa.string()),
])

# Provenance and Lineage Audit Schema
DATA_LINEAGE_SCHEMA = pa.schema([
    ("canonical_record_id", pa.string()),   # Unique canonical record identifier
    ("domain", pa.string()),
    ("canonical_event_id", pa.string()),
    ("source_api", pa.string()),            # 'fastf1', 'kaggle.rohanrao', 'jolpica.ergast', 'openf1', 'tum'
    ("source_record_id", pa.string()),      # Raw ID in originating system
    ("ingestion_timestamp", pa.string()),
    ("transformation_pipeline", pa.string()),# Pipeline stage applied
    ("reconciliation_outcome", pa.string()), # 'RECONCILED_EXACT', 'RECONCILED_AUTHORITY', 'STANDALONE'
    ("selected_source", pa.string()),
    ("selection_reason", pa.string()),
    ("pipeline_version", pa.string()),
])
