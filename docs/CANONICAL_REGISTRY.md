# PACE 1.0 — Canonical Entity Registry Documentation

## Overview

The Canonical Entity Registry (`src/data/canonical_registry.py`) is the authoritative identity resolution engine and single source of truth across all data streams and simulation modules in PACE. It resolves drivers, constructors, circuits, and Grand Prix events into immutable, verified identities.

```
Incoming Queries (FastF1, Ergast, Jolpica, OpenF1, TUM, Kaggle, User CLI)
                                 │
                                 ▼
                     normalize_key(query)
         (NFKD Unicode Decomposition, Lowercase, Separator Normalization)
                                 │
                                 ▼
                   CanonicalEntityRegistry Resolver
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
CanonicalDriver        CanonicalConstructor         CanonicalCircuit
(e.g., 'max_verstappen') (e.g., 'red_bull')           (e.g., 'vegas')
```

---

## 1. Resolution Methodology & Precedence

Resolution operates deterministically through the following precedence hierarchy:
1. **Exact Canonical ID Match:** If query matches a registered `canonical_id`, resolve immediately.
2. **Standardized Code Match:** For drivers, 3-letter uppercase abbreviations (`'VER'`, `'HAM'`, `'LEC'`, `'ANT'`).
3. **Permanent Number Resolution:** For drivers, car numbers with optional season disambiguation (e.g., `#5` in 2021 -> `'vettel'`, `#5` in 2026 -> `'bortoleto'`).
4. **External Source Cross-Index Match:**
   - TUM Racetrack Database keys (e.g., `'LasVegas'`, `'Melbourne'`, `'Austin'`, `'MarinaBay'`).
   - OpenF1 circuit keys / short names (e.g., `'vegas'`, `'albert_park'`).
   - Ergast / Jolpica IDs.
5. **Normalized Slug / Alias Match:** Strip unicode diacritics (`'Pérez'` -> `'perez'`, `'Hülkenberg'` -> `'hulkenberg'`), normalize punctuation (`'Las-Vegas Strip'` -> `'las_vegas_strip'`), and match registered historical aliases and renamings.
6. **Unresolved Handling:** Returns `ResolutionResult(is_resolved=False)` in default mode, or raises `UnresolvedEntityError` when `strict=True`.

---

## 2. The Las Vegas Defect Resolution

### Background & Investigation
- In the initial codebase, `src/simulation/pit_model.py` specified:
  ```python
  DEFAULT_CIRCUIT_TRANSIT_LOSS = {
      ...
      "las_vegas": 20.5,
  }
  ```
- However, all underlying historical data tables (`historical_events.parquet`, `historical_circuits.parquet`, `circuit_characteristics.parquet`), FastF1 schedules, and `traffic.py` utilized the canonical key **`"vegas"`**.
- Consequently, when the simulator or pit model was invoked with `circuit_id="vegas"`, the dictionary lookup failed and silently reverted to the default transit time ($21.0\text{ s}$) rather than the calibrated Vegas transit time ($20.5\text{ s}$).

### Resolution Implementation
1. **Authoritative Key Standard:** **`"vegas"`** is established as the single canonical identifier.
2. **Registered Aliases:** `"las_vegas"`, `"las vegas"`, `"las-vegas"`, `"vegas strip"`, `"las vegas strip street circuit"`, and `"LasVegas"` (TUM).
3. **Pit Transit Model Integration:** `PitStopModel.get_circuit_transit_loss` routes all circuit inputs through `resolve_circuit(circuit_id)`. Both `"vegas"` and `"las_vegas"` return exactly **$20.5\text{ s}$** (or $20.5 \times 0.55 = 11.275\text{ s}$ under Safety Car / VSC).
4. **Traffic & DRS Integration:** `TrafficOvertakingEngine.get_circuit_config` routes circuit lookups through `resolve_circuit`, retrieving the 2-zone Vegas DRS profile regardless of alias used.
5. **Feature Engineering Integration:** `extract_circuit_topology_features` uses `resolve_circuit`, properly mapping Vegas altitude ($610\text{ m}$) and length ($6.201\text{ km}$).

---

## 3. Entity Reference Tables

### 3.1 Constructors & Historical Renamings
| Canonical ID | Official Name | Active 2026 | Key Registered Aliases |
| :--- | :--- | :---: | :--- |
| `red_bull` | Red Bull Racing | Yes | `red bull`, `rbr`, `red bull f1` |
| `ferrari` | Scuderia Ferrari | Yes | `scuderia ferrari`, `ferrari f1`, `scuderia` |
| `mclaren` | McLaren Formula 1 Team | Yes | `mclaren`, `mclaren f1`, `mclaren racing` |
| `mercedes` | Mercedes-AMG PETRONAS F1 Team | Yes | `mercedes amg`, `brawn gp`, `brawn` |
| `aston_martin` | Aston Martin Aramco F1 Team | Yes | `racing point`, `force india`, `sahara force india`, `jordan`, `spyker`, `midland` |
| `alpine` | BWT Alpine F1 Team | Yes | `renault`, `renault f1`, `lotus f1`, `lotus` |
| `williams` | Williams Racing | Yes | `williams racing`, `williams f1` |
| `haas` | MoneyGram Haas F1 Team | Yes | `haas f1`, `haas f1 team` |
| `sauber` | Stake F1 Team Kick Sauber | Yes | `kick sauber`, `stake f1`, `alfa romeo`, `alfa`, `bmw sauber` |
| `rb` | Visa Cash App RB F1 Team | Yes | `racing bulls`, `vcarb`, `alphatauri`, `toro rosso`, `scuderia toro rosso`, `minardi` |
| `cadillac` | Cadillac Formula 1 Team | Yes | `cadillac f1`, `andretti cadillac` |
| `audi` | Audi Revolut F1 Team | Yes | `audi f1` |

### 3.2 Circuits & Pit Transit Losses
| Canonical ID | Circuit Name | Country | Length (km) | Laps | Transit Loss (s) | TUM Key |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`vegas`** | **Las Vegas Strip Street Circuit** | **USA** | **6.201** | **50** | **20.5** | **`LasVegas`** |
| `albert_park` | Albert Park Grand Prix Circuit | Australia | 5.278 | 58 | 20.5 | `Melbourne` |
| `americas` | Circuit of the Americas | USA | 5.513 | 56 | 21.8 | `Austin` |
| `bahrain` | Bahrain International Circuit | Bahrain | 5.412 | 57 | 24.5 | `Bahrain` |
| `baku` | Baku City Circuit | Azerbaijan | 6.003 | 51 | 21.0 | `Baku` |
| `catalunya` | Circuit de Barcelona-Catalunya | Spain | 4.657 | 66 | 22.5 | `Barcelona` |
| `hungaroring` | Hungaroring | Hungary | 4.381 | 70 | 21.5 | `Budapest` |
| `imola` | Autodromo Enzo e Dino Ferrari | Italy | 4.909 | 63 | 24.0 | `Imola` |
| `interlagos` | Autódromo José Carlos Pace | Brazil | 4.309 | 71 | 21.0 | `Interlagos` |
| `jeddah` | Jeddah Corniche Circuit | Saudi Arabia | 6.174 | 50 | 20.0 | `Jeddah` |
| `losail` | Lusail International Circuit | Qatar | 5.419 | 57 | 23.0 | `Lusail` |
| `marina_bay` | Marina Bay Street Circuit | Singapore | 4.940 | 62 | 28.0 | `MarinaBay` |
| `miami` | Miami International Autodrome | USA | 5.412 | 57 | 20.2 | `Miami` |
| `monaco` | Circuit de Monaco | Monaco | 3.337 | 78 | 22.0 | `Monaco` |
| `monza` | Autodromo Nazionale di Monza | Italy | 5.793 | 53 | 24.0 | `Monza` |
| `red_bull_ring`| Red Bull Ring | Austria | 4.318 | 71 | 20.0 | `Spielberg` |
| `rodriguez` | Autódromo Hermanos Rodríguez | Mexico | 4.304 | 71 | 22.5 | `MexicoCity` |
| `sepang` | Sepang International Circuit | Malaysia | 5.543 | 56 | 22.0 | `Sepang` |
| `shanghai` | Shanghai International Circuit | China | 5.451 | 56 | 22.8 | `Shanghai` |
| `silverstone` | Silverstone Circuit | UK | 5.891 | 52 | 20.8 | `Silverstone` |
| `spa` | Circuit de Spa-Francorchamps | Belgium | 7.004 | 44 | 22.0 | `Spa` |
| `suzuka` | Suzuka Circuit | Japan | 5.807 | 53 | 22.2 | `Suzuka` |
| `villeneuve` | Circuit Gilles Villeneuve | Canada | 4.361 | 70 | 18.5 | `Montreal` |
| `yas_marina` | Yas Marina Circuit | UAE | 5.281 | 58 | 22.0 | `YasMarina` |
| `zandvoort` | Circuit Park Zandvoort | Netherlands | 4.259 | 72 | 19.5 | `Zandvoort` |

---

## 4. Integration Guide for Future Data Sources

All future integrations MUST use the canonical registry interface to ensure data consistency:

### 4.1 Kaggle Historical Importer (Stage 5.2)
```python
from src.data.canonical_registry import resolve_driver, resolve_constructor, resolve_circuit

def import_kaggle_row(raw_row):
    driver_res = resolve_driver(raw_row["driverRef"], strict=True)
    constructor_res = resolve_constructor(raw_row["constructorRef"], strict=True)
    circuit_res = resolve_circuit(raw_row["circuitRef"], strict=True)
    
    return {
        "driver_id": driver_res.canonical_id,
        "constructor_id": constructor_res.canonical_id,
        "circuit_id": circuit_res.canonical_id,
        ...
    }
```

### 4.2 Jolpica REST API Client (Stage 5.3)
```python
from src.data.canonical_registry import resolve_driver, resolve_constructor

def parse_jolpica_standings(standings_data):
    for entry in standings_data:
        driver = resolve_driver(entry["Driver"]["driverId"], strict=True).entity
        constructor = resolve_constructor(entry["Constructors"][0]["constructorId"], strict=True).entity
```

### 4.3 OpenF1 Live Ingestion Client (Stage 5.4)
```python
from src.data.canonical_registry import resolve_driver, resolve_circuit

def parse_openf1_lap(lap_data, season=2026):
    driver_res = resolve_driver(lap_data["driver_number"], season=season, strict=True)
    circuit_res = resolve_circuit(lap_data["circuit_short_name"], strict=True)
```

### 4.4 TUM Track Geometry Importer (Stage 5.5)
```python
from src.data.canonical_registry import resolve_circuit

def link_tum_geojson(track_name):
    circuit = resolve_circuit(track_name, strict=True).entity
    print(f"Linking TUM track geometry to canonical circuit: {circuit.canonical_id}")
```
