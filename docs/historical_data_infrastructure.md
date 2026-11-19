# PACE: Historical Data Infrastructure Documentation

## 1. Overview

The Historical Data Infrastructure module (`src/data`) is responsible for acquiring, standardizing, validating, and persisting historical Formula 1 data.

The infrastructure serves two complementary objectives:
1. **General Seasonal Telemetry & Results (2023–2026):** High-resolution recent session data, practice lap timings (Sectors 1–3, speed traps), qualifying, race classifications, and pit stops.
2. **Driver–Circuit Historical Intelligence (Complete Career History):** Full career records for modern drivers and key historical teammates spanning 2001–2026 across 39 world championship circuits, capturing granular event-level appearances, qualifying times, positions gained/lost, teammate head-to-head matchups, and aggregated statistical profiles.

The infrastructure is designed with the following core engineering principles:
- **Resumability & Idempotency:** Data collection checks existing local caches before querying external APIs, avoiding redundant network traffic.
- **Strict Data Partitioning:** Raw payloads, seasonal processed Parquet datasets, driver-circuit intelligence tables, and compiled multi-season historical tables are isolated in dedicated directories.
- **Strong Typing & Schema Enforcement:** All processed and historical tables conform to standardized PyArrow schemas with explicit nullable and physical data types.
- **Data Leakage Prevention:** Race results and classifications are treated strictly as target labels and are partitioned from pre-race predictor features.
- **No Reliance on Authenticated Private Streams:** Core timing, classification, and event metadata are ingested via FastF1's open Ergast client and OpenF1 APIs, avoiding private F1 credential dependencies.

---

## 2. Data Sources & Ingestion Endpoints

| Category | Primary Source | Protocol / Client | Extracted Attributes |
| :--- | :--- | :--- | :--- |
| **Race Results** | FastF1 Ergast | `fastf1.ergast.Ergast.get_race_results()` | Grid, finish position, points, laps, status, fastest lap, race duration |
| **Qualifying** | FastF1 Ergast | `fastf1.ergast.Ergast.get_qualifying_results()` | Q1, Q2, Q3 lap times, grid classifications |
| **Sprint Results** | FastF1 Ergast | `fastf1.ergast.Ergast.get_sprint_results()` | Sprint finishing positions, points, lap times |
| **Lap Times & Pit Stops** | FastF1 Ergast | `fastf1.ergast.Ergast.get_lap_times()`, `get_pit_stops()` | Lap-by-lap positions, elapsed lap times, pit stop durations |
| **Practice Telemetry** | OpenF1 API | `https://api.openf1.org/v1/laps` | Sector 1/2/3 durations, speed trap speeds (i1, i2, st), in/out lap flags |
| **Biographical & Entities** | FastF1 Ergast | `get_driver_info()`, `get_constructor_info()`, `get_circuits()`, `get_race_schedule()` | Canonical driver/team IDs, circuit coordinates, session timelines |
| **Driver Career History** | FastF1 Ergast | `get_race_results(driver=...)`, `get_qualifying_results(driver=...)` | Full career race and qualifying starts, teammate comparisons, positions gained/lost |

---

## 3. Directory Layout

```
data/
├── raw/                              # Raw JSON payloads directly from APIs
│   ├── driver_circuit/               # Career race and qualy JSON files per driver ({driver_id}_races.json)
│   ├── 2023/
│   │   ├── drivers/
│   │   ├── constructors/
│   │   ├── circuits/
│   │   ├── events/
│   │   ├── race_results/
│   │   ├── qualifying/
│   │   ├── sprints/
│   │   ├── laps/
│   │   ├── pit_stops/
│   │   └── practice_laps/
│   ├── 2024/ ...
│   ├── 2025/ ...
│   └── 2026/ ...
├── processed/                        # Normalized, schema-validated Parquet files per season
│   ├── 2023/
│   │   ├── drivers_2023.parquet
│   │   ├── constructors_2023.parquet
│   │   ├── circuits_2023.parquet
│   │   ├── events_2023.parquet
│   │   ├── sessions_2023.parquet
│   │   ├── race_results_2023.parquet
│   │   ├── qualifying_2023.parquet
│   │   ├── sprints_2023.parquet
│   │   ├── laps_2023.parquet
│   │   ├── pit_stops_2023.parquet
│   │   └── practice_laps_2023.parquet
│   ├── 2024/ ...
│   ├── 2025/ ...
│   └── 2026/ ...
└── historical/                       # Master compiled multi-season datasets
    ├── driver_circuit/               # Driver-Circuit Historical Intelligence tables
    │   ├── driver_circuit_appearances.parquet  # Granular per-event records
    │   ├── driver_circuit_aggregated.parquet   # Aggregated performance profiles
    │   └── circuit_characteristics.parquet     # Circuit technical profiles (all 39 tracks)
    ├── historical_drivers.parquet
    ├── historical_constructors.parquet
    ├── historical_circuits.parquet
    ├── historical_events.parquet
    ├── historical_sessions.parquet
    ├── historical_race_results.parquet
    ├── historical_qualifying.parquet
    ├── historical_sprints.parquet
    ├── historical_laps.parquet
    ├── historical_pit_stops.parquet
    └── historical_practice_laps.parquet
```

---

## 4. Schemas and Entity Relationships

### General Historical Tables
- **`historical_drivers.parquet`:** `driver_id` (PK), `driver_code`, `driver_number`, `given_name`, `family_name`, `full_name`, `date_of_birth`, `nationality`, `driver_url`, `source_api`, `retrieved_at`.
- **`historical_constructors.parquet`:** `constructor_id` (PK), `constructor_name`, `nationality`, `constructor_url`, `source_api`, `retrieved_at`.
- **`historical_circuits.parquet`:** `circuit_id` (PK), `circuit_name`, `locality`, `country`, `latitude`, `longitude`, `circuit_url`, `source_api`, `retrieved_at`.
- **`historical_events.parquet`:** `(season, round)` (PK), `event_name`, `circuit_id` (FK), `country`, `event_date`, `event_url`, `source_api`, `retrieved_at`.
- **`historical_race_results.parquet`:** `(season, round, driver_id)` (PK), `constructor_id` (FK), `grid_position`, `finish_position`, `position_text`, `points`, `laps_completed`, `status`, `is_classified`, `is_dnf`, `is_dns`, `is_dsq`, `total_time_ms`, `total_time_str`, `fastest_lap_rank`, `fastest_lap_time`, `fastest_lap_speed`.
- **`historical_qualifying.parquet`:** `(season, round, driver_id)` (PK), `constructor_id` (FK), `qualifying_position`, `q1_time_str`, `q1_time_ms`, `q2_time_str`, `q2_time_ms`, `q3_time_str`, `q3_time_ms`.
- **`historical_practice_laps.parquet`:** `season`, `round`, `session_type`, `driver_number`, `lap_number`, `lap_time_seconds`, `sector_1_seconds`, `sector_2_seconds`, `sector_3_seconds`, `is_pit_out_lap`, `i1_speed`, `i2_speed`, `st_speed`.

### Driver–Circuit Intelligence Schemas
- **`driver_circuit_appearances.parquet`:** Granular per-event records preserving individual historical appearances:
  - `driver_id`, `circuit_id`, `season`, `round`, `event_name`, `event_date`
  - `regulation_era`: F1 regulation era context (`ground_effect_2022_2026`, `turbo_hybrid_2014_2021`, `v8_2006_2013`, `v10_1996_2005`, `early_pre_1996`)
  - `constructor_id`, `constructor_name`: Team driven for during that specific Grand Prix
  - `grid_position`, `finish_position`, `position_text`, `positions_gained_lost` (`grid - finish`)
  - `points`, `laps_completed`, `status`, `is_classified`, `is_dnf`, `is_dns`, `is_dsq`
  - `qualifying_position`, `q1_time_str`, `q1_time_ms`, `q2_time_str`, `q2_time_ms`, `q3_time_str`, `q3_time_ms`
  - `fastest_lap_rank`, `fastest_lap_time`, `fastest_lap_speed`, `total_race_time_ms`
  - `teammate_driver_id`, `teammate_grid_position`, `teammate_finish_position`, `beat_teammate_race`, `beat_teammate_qualy`
  - `source_api`, `retrieved_at`

- **`driver_circuit_aggregated.parquet`:** Aggregated performance profiles:
  - `driver_id`, `circuit_id`, `total_starts`, `first_season`, `latest_season`
  - `wins`, `podiums`, `poles`, `top_5_finishes`, `top_10_finishes`
  - `dnf_count`, `dnf_rate`
  - `total_points`, `avg_points_per_start`
  - `avg_grid_position`, `avg_finish_position`, `avg_positions_gained`
  - `best_finish_position`, `best_grid_position`
  - `teammate_race_h2h_wins`, `teammate_qualy_h2h_wins`, `teammate_race_h2h_rate`, `teammate_qualy_h2h_rate`
  - `ground_effect_starts`, `ground_effect_avg_finish`: Specific to current 2022+ technical regulation era

- **`circuit_characteristics.parquet`:** Engineering characteristics across all 39 historical circuits:
  - `circuit_id`, `circuit_name`, `country`, `locality`, `latitude`, `longitude`
  - `first_grand_prix_year`, `total_races_held`
  - `circuit_length_km`, `typical_race_laps`
  - `overtaking_difficulty_rating` (`Low`, `Medium`, `High`)
  - `tyre_stress_level` (`Low`, `Medium`, `High`, `Very High`)
  - `downforce_level` (`Low`, `Medium`, `High`)
  - `altitude_m`: Circuit elevation above sea level

---

## 5. Pipeline Execution Instructions

### Ingest Complete Driver-Circuit Historical Intelligence
```bash
python main.py --collect-circuit-history
```

### Ingest Seasonal Datasets (2023–2026)
```bash
python main.py --collect-data
```

### Ingest Specific Drivers
```bash
python main.py --collect-circuit-history --drivers hamilton alonso max_verstappen
```

### Run Quality Validation Only
```bash
python main.py --validate-data
```

### Check Project Status
```bash
python main.py --status
```

---

## 6. Historical Data Coverage & Regulation Eras

Formula 1 technical regulations significantly alter aerodynamic downforce, engine power, tyre degradation, and overtaking dynamics. PACE contextualizes historical performance across five recognized technical eras:

1. **Ground Effect Era (2022–2026):** Venturi tunnel underbody aerodynamics, 18-inch low-profile Pirelli tyres, E10 sustainable fuel, standardized minimum weight.
2. **Turbo-Hybrid Era (2014–2021):** 1.6L V6 turbocharged internal combustion engines with dual Motor Generator Units (MGU-K and MGU-H), advanced energy recovery systems, wide track wings introduced in 2017.
3. **V8 Era (2006–2013):** 2.4L naturally aspirated V8 engines, introduction of knockout qualifying format (Q1/Q2/Q3, 2006), Kinetic Energy Recovery System (KERS, 2009/2011), Drag Reduction System (DRS, 2011), Pirelli tyre supplier switch (2011).
4. **V10 Modern Era (1996–2005):** 3.0L V10 engines, grooved tyres, electronic timing and telemetry standardization.
5. **Early Classic Era (Pre-1996):** Pre-modern timing archives, manual lap scoring.

---

## 7. Known Limitations & Technical Decisions

1. **FastF1 Historical Scope & Ergast Bridge:** FastF1's high-frequency session car telemetry (`session.load()`) is primarily supported from the 2018 season onwards, and prompts for credentials when querying LiveTiming directly. PACE bridges this by leveraging FastF1's Ergast interface (`fastf1.ergast.Ergast`) and OpenF1 REST endpoints, achieving 100% open, automated, unauthenticated ingestion spanning 1950–2026.
2. **Rate Limiting & Resumability:** The Ergast backend (`api.jolpi.ca`) enforces request throttling (~4 req/sec). PACE implements automatic retry with exponential backoff (`2s`, `4s`, `8s`, `16s`, `32s`) and persistent raw disk caching (`data/raw/driver_circuit/{driver_id}_races.json`), ensuring resumable, zero-waste data acquisition.
3. **Telemetry Granularity:** High-frequency 20Hz car sensor telemetry (throttle, brake, RPM) is deliberately excluded from bulk historical lap storage to avoid excessive disk bloat (>50GB). Detailed telemetry interfaces are preserved for targeted, on-demand head-to-head analysis in later phases.
4. **Ongoing / Future Seasons (2026):** For scheduled rounds without completed sessions, result records are omitted without failing validation checks or stopping the pipeline.
