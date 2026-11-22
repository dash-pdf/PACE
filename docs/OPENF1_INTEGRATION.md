# OpenF1 Race Intelligence Integration

## 1. Architectural Mission & Scope

PACE integrates the OpenF1 REST API (`https://api.openf1.org/v1/`) as its dedicated **Race Intelligence** data provider. In accordance with the PACE multi-source intelligence architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PACE DATA INFRASTRUCTURE                         │
├──────────────────┬───────────────────┬──────────────────┬───────────────┤
│     FastF1       │   Kaggle Archive  │   Jolpica F1     │    OpenF1     │
│ (2023–Present)   │    (1950–2024)    │  (Standings)     │(Race Intel)   │
├──────────────────┼───────────────────┼──────────────────┼───────────────┤
│ Authoritative    │ Historical        │ Standings,       │ Intervals,    │
│ timing feed,     │ relational back-  │ mathematical     │ stints, race  │
│ telemetry laps,  │ bone, long-term   │ eliminations,    │ control msgs, │
│ transponders     │ driver career     │ era scoring      │ overtakes,    │
│ & micro-sectors  │ baselines         │ rules            │ weather, wake │
└──────────────────┴───────────────────┴──────────────────┴───────────────┘
```

OpenF1 provides high-frequency telemetry, live intervals, tyre compound stints, weather telemetry, on-track overtakes, and official FIA race control messages. It serves as an independent corroborative source for FastF1 and generates critical features for the PACE simulation and prediction models:

- **Traffic & Wake Modelling:** Dynamic gap-to-leader and interval tracking to calculate empirical traffic exposure percentages ($\le 1.0\text{s}$ DRS wake exposure).
- **Incident & Reliability Calibration:** Exact timestamps, start/end lap bounds, and durations for Safety Cars, Virtual Safety Cars, and Red Flags.
- **Tyre Strategy Validation:** Live stint tracking, compound progression, and pit stops conducted under neutralisation.
- **Environmental Dynamics:** Continuous track and ambient temperature gradient tracking and rainfall detection.

PACE remains a **CLI-first engineering analytics and simulation system**. OpenF1 operates without mandatory user credentials or live web dashboards, fully functional in historical mode for seasons 2023 onwards.

---

## 2. API Endpoints & Capabilities

| Endpoint | OpenF1 Path | Key Parameters | PACE Schema | Primary Use in PACE |
|:---|:---|:---|:---|:---|
| Meetings | `/meetings` | `year`, `meeting_key` | `OPENF1_MEETINGS_SCHEMA` | Meeting calendar & canonical circuit resolution |
| Sessions | `/sessions` | `meeting_key`, `year`, `session_type` | `OPENF1_SESSIONS_SCHEMA` | Session discovery (Practice, Qualifying, Sprint, Race) |
| Drivers | `/drivers` | `session_key`, `driver_number` | `OPENF1_DRIVERS_SCHEMA` | Session roster, number-to-driver canonical mapping |
| Intervals | `/intervals` | `session_key`, `driver_number` | `OPENF1_INTERVALS_SCHEMA` | High-frequency gaps to leader and car ahead |
| Stints | `/stints` | `session_key`, `driver_number` | `OPENF1_STINTS_SCHEMA` | Compound usage sequences, tyre age, stint lengths |
| Race Control | `/race_control` | `session_key`, `flag`, `driver_number`| `OPENF1_RACE_CONTROL_SCHEMA` | Safety Car, VSC, Red Flag, penalties, incidents |
| Weather | `/weather` | `session_key` | `OPENF1_WEATHER_SCHEMA` | Track/air temperature, humidity, wind, rainfall |
| Overtakes | `/overtakes` | `session_key` | `OPENF1_OVERTAKES_SCHEMA` | On-track position changes and overtake counts |
| Position | `/position` | `session_key`, `driver_number` | `OPENF1_POSITION_SCHEMA` | Lap-by-lap running order tracking |
| Pit Stops | `/pit` | `session_key`, `driver_number` | `OPENF1_PIT_SCHEMA` | Pit stop entry, duration, and stationary time |
| Car Data | `/car_data` | `session_key`, `driver_number`, `speed>=` | `OPENF1_CAR_DATA_SCHEMA` | High-frequency engine RPM, speed, gear, throttle, DRS |
| Location | `/location` | `session_key`, `driver_number`, `date>=` | `OPENF1_LOCATION_SCHEMA` | 3D car coordinates $(X, Y, Z)$ on circuit |

---

## 3. Rate Limiting, Resilience & Disk Caching

### 3.1 Rate Limiting Architecture
OpenF1 enforces a free-tier rate limit of:
- **Burst cap:** Maximum 3 requests per second.
- **Rolling cap:** Maximum 30 requests per minute.

`OpenF1RateLimiter` implements a sliding 60-second window tracking timestamps of past requests. If 30 requests have been issued within the last 60 seconds, it sleeps until the oldest request leaves the window. Between individual requests, it enforces a minimum delay (default: $0.35\text{ s}$).

### 3.2 HTTP 429 & Retry Handling
If an HTTP 429 response is received:
1. The client inspects the `Retry-After` header. If present, it sleeps for the specified seconds.
2. If absent, it performs exponential backoff: $t_{\text{sleep}} = \text{base} \times 2^{\text{attempt}-1} + \text{jitter}$.
3. Max retries default to 4 attempts. Persistent errors raise `OpenF1RateLimitError`.

### 3.3 HTTP 404 Normalization
When querying sessions or drivers with no matching records (e.g., an endpoint returning `{"detail":"No results found."}`), OpenF1 returns HTTP 404. `OpenF1Client` intercepts this status and returns an empty list `[]` cleanly rather than throwing an unhandled exception.

### 3.4 Comparison Operator URL Encoding Preservation
OpenF1 route handlers require literal comparison operators (e.g., `speed>=200`, `date>=...`). Standard HTTP client URL encoding turns `>=` into `%3E%3D`, which causes OpenF1's router to fail to match filters. `OpenF1Client` uses prepared requests to unquote comparison operator keys, ensuring exact URL dispatch (`https://api.openf1.org/v1/car_data?session_key=7953&driver_number=1&speed>=200`).

### 3.5 Disk Caching
Responses are cached locally in `data/raw/openf1/cache/`:
- **Deterministic filenames:** `openf1_{endpoint}_{sha256[:16]}.json` computed from normalized query parameters.
- **Atomic persistence:** Files are written to `.tmp` and renamed atomically to prevent partial writes.
- **Self-healing:** Corrupted or invalid JSON cache files are automatically purged and transparently re-fetched from the API.

---

## 4. Canonical Identity Resolution

OpenF1 numbers are session-specific and team names often reflect ephemeral title sponsors. `OpenF1IngestionEngine` resolves every entity against `CanonicalEntityRegistry`:

1. **Session Driver Map:** Ingestion queries `/drivers` first to map session driver numbers to broadcast names and family names.
2. **Driver Resolution:** Driver numbers and names resolve to canonical IDs (e.g., Car 1 in 2023 $\to$ `"max_verstappen"`).
3. **Constructor Resolution:** Team names resolve to canonical constructor IDs (e.g., `"Oracle Red Bull Racing"` $\to$ `"red_bull"`, `"Aston Martin Aramco Cognizant F1 Team"` $\to$ `"aston_martin"`).
4. **Circuit Resolution:** Circuit keys resolve to canonical IDs (e.g., `"Sakhir"` $\to$ `"bahrain"`, `"Las Vegas"` $\to$ `"vegas"`).

Unresolved entities are collected in an audit ledger (`engine.unresolved`) and logged without halting execution.

---

## 5. High-Frequency Telemetry Sampling Strategy

A full Grand Prix generates over 35,000 raw telemetry samples per driver ($>700,000$ points across 20 cars). Ingesting raw high-frequency telemetry indiscriminately causes storage bloat and slow Parquet scans.

`OpenF1IngestionEngine.ingest_selective_car_data` applies targeted ingestion:
- **Driver Filtering:** Telemetry is only ingested for explicitly requested drivers (e.g., `--drivers 1 16 4`).
- **Speed Thresholding:** Queries OpenF1 with `speed>=200` to capture high-speed straight-line performance and speed trap apexes.
- **Subsampling Step:** Applies decimation (e.g., `sample_step=5`) to retain representative dynamics while reducing storage by $80\%$.

---

## 6. Race Intelligence Analytics Engine

`src/data/openf1_analytics.py` provides `OpenF1RaceIntelligence` for deriving structured features from raw OpenF1 Parquet datasets:

### 6.1 Reconstructed Neutralisation Periods
Reconstructs continuous Safety Car, VSC, and Red Flag periods from discrete race control messages:
- **Event Classification:** Distinguishes physical Safety Car from Virtual Safety Car (VSC) and Red Flags.
- **Boundary Detection:** Pairs `DEPLOYED` messages with `IN THIS LAP`, `VSC ENDING`, or `TRACK CLEAR` / `GREEN` flags.
- **Metrics Computed:** Exact start/end timestamps, start/end lap numbers, total duration in seconds, and number of laps affected.

### 6.2 Traffic & Wake Exposure Metric
Calculates the proportion of time a car ran directly in dirty air:
$$\text{Traffic Exposure} = \frac{N(\text{gap\_ahead} \le 1.0\text{ s})}{N(\text{total valid intervals})} \times 100\%$$
Identifies drivers compromised by wake turbulence and DRS trains versus those running in clear air.

### 6.3 Environmental Conditions & Weather Profiles
Aggregates session weather observations:
- Mean, minimum, maximum track and air temperatures, plus total track temperature gradient ($\Delta T_{\text{track}}$).
- Mean relative humidity and wind speed.
- Rainfall detection and wet onset timestamps.

### 6.4 Driver Overtaking & Progression Intelligence
Synthesizes driver-level performance:
- Total laps completed, stint count, and compounds used.
- Overtakes made, times overtaken, and net overtakes.
- Pit stops executed under Safety Car or VSC conditions.
- Standard deviation of running position (position volatility).

---

## 7. Cross-Source Reconciliation & Precedence Matrix

`OpenF1Reconciler` compares overlapping datasets between OpenF1 and FastF1.

### Official Source Precedence Matrix
| Dimension | Authoritative Source | Precedence Justification |
|:---|:---|:---|
| **Lap Duration & Sector Times** | **FastF1** | Authoritative. Direct FOM transponder timing loops provide microsecond-accurate lap and sector boundaries. |
| **Speed Traps & Corner Speeds** | **FastF1** | Authoritative. Official FOM calibrated optical speed trap transponder lines. |
| **Gaps & Intervals Stream** | **OpenF1** | Authoritative. Continuous high-frequency live gap-to-leader and interval feed. |
| **Race Control Messages** | **OpenF1** | Authoritative. Exact verbatim FIA race director message text and ISO timestamps. |
| **Overtaking Maneuvers** | **OpenF1** | Corroborative. Independent track overtake sensor detection cross-verified with lap positions. |
| **Tyre Stints & Compounds** | **FastF1** | Authoritative for pit allocation; OpenF1 provides corroboration on lap transitions and tyre age. |

---

## 8. Command-Line Interface (CLI) Reference

The PACE CLI provides complete OpenF1 management commands:

```bash
# 1. Check OpenF1 API connectivity and local cache status
python main.py --openf1-status

# 2. Query championship meetings for a given season
python main.py --openf1-meetings --target-season 2023

# 3. Query sessions for a specific meeting
python main.py --openf1-sessions --meeting-key 1141

# 4. Ingest and persist all race intelligence datasets for a session
python main.py --openf1-ingest --target-season 2023 --session-key 7953

# 5. Generate and print full race intelligence report
python main.py --openf1-intelligence --target-season 2023 --session-key 7953

# 6. Selectively ingest high-frequency telemetry for car 1 (Verstappen)
python main.py --openf1-telemetry --target-season 2023 --session-key 7953 --drivers 1

# 7. Run cross-source reconciliation against FastF1
python main.py --openf1-reconcile --target-season 2023 --session-key 7953
```
