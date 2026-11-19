# PACE: Historical Data Infrastructure Documentation

## 1. Overview

The Historical Data Infrastructure module (`src/data`) is responsible for acquiring, standardizing, validating, and persisting historical Formula 1 data spanning the 2023–2026 championship seasons.

The infrastructure is designed with the following core engineering principles:
- **Resumability & Idempotency:** Data collection checks existing local caches before querying external APIs, avoiding redundant network traffic.
- **Strict Data Partitioning:** Raw payloads, seasonal processed Parquet datasets, and compiled multi-season historical tables are isolated in dedicated directories.
- **Strong Typing & Schema Enforcement:** All processed and historical tables conform to standardized PyArrow schemas.
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

---

## 3. Directory Layout

```
data/
├── raw/                      # Raw JSON payloads directly from APIs
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
├── processed/                # Normalized, schema-validated Parquet files per season
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
└── historical/               # Master compiled multi-season datasets
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

### Entity Schemas
- **`historical_drivers.parquet`:** `driver_id` (PK), `driver_code`, `driver_number`, `given_name`, `family_name`, `full_name`, `date_of_birth`, `nationality`, `driver_url`, `source_api`, `retrieved_at`.
- **`historical_constructors.parquet`:** `constructor_id` (PK), `constructor_name`, `nationality`, `constructor_url`, `source_api`, `retrieved_at`.
- **`historical_circuits.parquet`:** `circuit_id` (PK), `circuit_name`, `locality`, `country`, `latitude`, `longitude`, `circuit_url`, `source_api`, `retrieved_at`.
- **`historical_events.parquet`:** `(season, round)` (PK), `event_name`, `circuit_id` (FK), `country`, `event_date`, `event_url`, `source_api`, `retrieved_at`.

### Result & Session Schemas
- **`historical_race_results.parquet`:** `(season, round, driver_id)` (PK), `constructor_id` (FK), `grid_position`, `finish_position`, `position_text`, `points`, `laps_completed`, `status`, `is_classified`, `is_dnf`, `is_dns`, `is_dsq`, `total_time_ms`, `total_time_str`, `fastest_lap_rank`, `fastest_lap_time`, `fastest_lap_speed`.
- **`historical_qualifying.parquet`:** `(season, round, driver_id)` (PK), `constructor_id` (FK), `qualifying_position`, `q1_time_str`, `q1_time_ms`, `q2_time_str`, `q2_time_ms`, `q3_time_str`, `q3_time_ms`.
- **`historical_practice_laps.parquet`:** `season`, `round`, `session_type`, `driver_number`, `lap_number`, `lap_time_seconds`, `sector_1_seconds`, `sector_2_seconds`, `sector_3_seconds`, `is_pit_out_lap`, `i1_speed`, `i2_speed`, `st_speed`.

---

## 5. Pipeline Execution Instructions

### Run Ingestion
To collect all historical data across 2023–2026:
```bash
python main.py --collect-data
```

### Ingest Specific Seasons
```bash
python main.py --collect-data --seasons 2023 2024
```

### Force Refresh Caches
```bash
python main.py --collect-data --force
```

### Run Quality Validation Only
```bash
python main.py --validate-data
```

---

## 6. Known Limitations & Technical Decisions

1. **Live Timing Credentials Exemption:** FastF1's `livetiming.formula1.com` endpoint prompts for Formula 1 subscriber login credentials. To keep the pipeline open, automated, and free of sensitive plain-text credentials, race and session timing data are retrieved through FastF1's Ergast interface and OpenF1 APIs.
2. **Deep Telemetry Scope:** High-frequency (20Hz) car telemetry (throttle, brake, RPM) is deliberately excluded from bulk lap-level ingestion to prevent excessive disk utilization. The pipeline is architected so telemetry can be fetched on demand for specific driver-lap comparisons in later modeling phases.
3. **Current Season Scope (2026):** For future or ongoing seasons (e.g. 2026), scheduled rounds without completed sessions return empty result records and are gracefully handled without pipeline interruption.
