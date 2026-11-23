# PACE Stage 5.5 — TUM Racetrack Database Integration: Stage Completion Report

## Executive Summary

Stage 5.5 has been implemented, validated, and verified for **PACE (Predictive Analytics for Competitive Evaluation)**.

The official **TUM Racetrack Database** (`https://github.com/TUMFTM/racetrack-database`) has been audited, ingested, and integrated as PACE's foundational **Circuit Geometry Intelligence Layer**. The implementation delivers a robust, CLI-first data and feature pipeline that parses raw centerline and boundary CSVs, computes derived differential geometry (continuous arc length, heading, periodic central difference curvature, radius, road width), infers corner classifications and curvature spectra, and persists strongly-typed PyArrow Parquet datasets.

All acceptance criteria have been achieved:
- **Mandatory Initial Audit Completed:** Source revision, license, inventory, and genuine versus unavailable attributes were fully audited prior to pipeline construction.
- **Test Suite Verification:** 16 of 16 tests in `tests/test_tum_integration.py` pass. Full project regression suite reports **154 passing tests with 0 failures**.
- **Canonical Circuit Reconciliation:** All 20 modern/historic Formula 1 circuits present in the TUM database were resolved to PACE canonical circuit identifiers ($100\%$ match rate).
- **Physical Layout Accuracy:** 17 of 20 matched F1 circuits demonstrate sub-$15\text{ m}$ length delta ($<0.3\%$) relative to official FIA/PACE specifications. Historical layout reconfigurations for Yas Marina ($+265.6\text{ m}$) and Zandvoort ($+57.5\text{ m}$) were quantitatively identified and documented.
- **Architectural Boundary Adherence:** Zero web application infrastructure introduced; no synthetic geometry fabricated for missing circuits; prediction and simulation weights preserved without premature modification.

---

## 1. Source Provenance & License

| Property | Value / Specification |
|:---|:---|
| **Upstream Repository** | `https://github.com/TUMFTM/racetrack-database` |
| **Institutional Lead** | Institute of Automotive Technology (FTM), Technical University of Munich (TUM) |
| **Primary Authors** | Alexander Heilmeier (`alexander.heilmeier@tum.de`), with width extraction by Andressa de Paula Suiti |
| **Git Commit Revision** | `e59595d1f3573b30d1ded6a08984935b957688e0` (grafted `master`, 2021-09-18T16:28:20+02:00) |
| **Official License** | **GNU Lesser General Public License v3 (LGPL-3.0)** (`LICENSE` file in repo root) |
| **Primary Data Origins** | OpenStreetMap (OSM) GPS centerline traces with smoothing splines; satellite imagery boundary processing; minimum curvature trajectory optimization (`global_racetrajectory_optimization`) |

> [!NOTE]
> *License Correction Note:* Early PACE roadmap sketches assumed CC BY 4.0 based on general academic conventions. The audit of the official repository established that the database is distributed under **LGPL-3.0**. Full LGPL-3.0 license terms and notices are maintained in `data/raw/tum/repo/LICENSE`.

---

## 2. Dataset Inventory

The official TUM database provides 25 circuits:

| Directory | File Count | Content & Format | File Size Range | Total Disk Size |
|:---|:---|:---|:---|:---|
| `tracks/` | 25 CSV files | Centerline & boundaries: `# x_m,y_m,w_tr_right_m,w_tr_left_m` | 15.6 KB – 48.3 KB | ~840 KB |
| `racelines/` | 25 CSV files | Minimum curvature line: `# x_m,y_m` | 7.8 KB – 24.2 KB | ~420 KB |
| `racelines/` (plots) | 50 PNG files | Diagnostic plots: `*_raceline.png` & `*_curvature.png` | 250 KB – 1.8 MB | ~35 MB |

### Complete Track File Inventory
```
Austin.csv         BrandsHatch.csv    Budapest.csv       Catalunya.csv      Hockenheim.csv
IMS.csv            Melbourne.csv      MexicoCity.csv     Montreal.csv       Monza.csv
MoscowRaceway.csv  Norisring.csv      Nuerburgring.csv   Oschersleben.csv   Sakhir.csv
SaoPaulo.csv       Sepang.csv         Shanghai.csv       Silverstone.csv    Sochi.csv
Spa.csv            Spielberg.csv      Suzuka.csv         YasMarina.csv      Zandvoort.csv
```

---

## 3. Circuit Coverage & Calendar Gaps

Of the 25 circuits in the TUM database:
- **20 circuits ($80\%$)** match Formula 1 championship venues in the PACE canonical registry.
- **5 circuits ($20\%$)** are non-canonical tracks (DTM, historic F1 prior to 1986, or IndyCar):
  - `BrandsHatch`: Historic F1 (1964–1986) and DTM venue.
  - `IMS`: Indianapolis Motor Speedway (historic F1 2000–2007, IndyCar).
  - `MoscowRaceway`: DTM circuit; never hosted an F1 championship GP.
  - `Norisring`: DTM Nuremberg street circuit; never hosted an F1 championship GP.
  - `Oschersleben`: Motorsport Arena Oschersleben (DTM/GT); never hosted an F1 championship GP.

### Missing Canonical F1 Championship Circuits (12 tracks)
The TUM database snapshot (September 2021) does **not** contain every modern Formula 1 circuit:
```
baku           (Baku City Circuit, Azerbaijan)
imola          (Autodromo Enzo e Dino Ferrari, Italy)
jeddah         (Jeddah Corniche Circuit, Saudi Arabia)
losail         (Lusail International Circuit, Qatar)
madring        (Madring, Spain)
magny_cours    (Circuit de Nevers Magny-Cours, France)
marina_bay     (Marina Bay Street Circuit, Singapore)
miami          (Miami International Autodrome, USA)
monaco         (Circuit de Monaco, Monaco)
mugello        (Autodromo Internazionale del Mugello, Italy)
portimao       (Autódromo Internacional do Algarve, Portugal)
vegas          (Las Vegas Strip Street Circuit, USA)
```

> [!IMPORTANT]
> **No Geometry Fabrication:** In strict adherence to Section 2 of the prompt instructions, PACE **does not fabricate geometry or substitute unverified external sources** for missing circuits. Missing tracks are explicitly marked as unavailable in the geometry layer, and downstream models fall back to canonical circuit characteristic metrics (`lap_length_km`, `default_transit_loss_s`, `CIRCUIT_METRICS`).

---

## 4. Available vs Missing Geometry Attributes

To preserve scientific rigor, PACE explicitly partitions all geometric attributes into three verified tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GEOMETRIC ATTRIBUTE TIERS                       │
├────────────────────────────────┬───────────────────────────────────────┤
│ Tier                           │ Attributes & Definitions              │
├────────────────────────────────┼───────────────────────────────────────┤
│ 1. Source Measurements         │ x_m, y_m: 2D local Cartesian coords   │
│    (Directly provided by TUM)  │ w_tr_right_m: width to right boundary │
│                                │ w_tr_left_m: width to left boundary   │
│                                │ raceline x_m, y_m: optimal line       │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. Derived Measurements        │ s_m: cumulative centerline distance   │
│    (Exact mathematical transforms│ track_length_m: closed loop distance │
│     from source coordinates)   │ closure_error_m: end-to-start gap     │
│                                │ w_total_m: w_right + w_left           │
│                                │ psi_rad: unwrapped heading angle      │
│                                │ kappa_radpm: periodic central diff dψ/ds│
│                                │ radius_m: curvature radius 1 / |κ|    │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Inferred Values             │ Corner identification (R <= 300m)     │
│    (Require empirical criteria │ Speed classification (low/med/high)   │
│     and engineering models)    │ Straight segmentation (R > 300m)      │
│                                │ Curvature spectrum (% low/med/high/st)│
│                                │ Longest straight (meters)             │
│                                │ Braking severity proxy (energy drop)  │
├────────────────────────────────┼───────────────────────────────────────┤
│ 4. Genuinely Absent Attributes │ Elevation (z_m) & vertical gradients  │
│    (Unavailable in TUM DB)     │ Track banking and camber angles       │
│                                │ Sector & corner numbered annotations  │
│                                │ WGS84 coordinates (lat/lon anchors)   │
│                                │ Velocity / lap time telemetry         │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 5. Architecture & Implementation

```
PACE Repository Architecture — Stage 5.5
├── data/
│   ├── raw/tum/repo/             <- Cloned/cached TUM repository (LGPL-3.0)
│   │   ├── tracks/               <- 25 raw centerline CSVs
│   │   └── racelines/            <- 25 raw raceline CSVs and PNGs
│   └── historical/tum/           <- Partitioned PyArrow Parquet files
│       ├── tum_circuits_summary.parquet
│       ├── {circuit_id}_track_geometry.parquet
│       ├── {circuit_id}_raceline.parquet
│       └── {circuit_id}_corners.parquet
├── src/
│   ├── data/
│   │   ├── tum_client.py         <- TUM client, repo manager, metadata
│   │   ├── tum_ingestion.py      <- Ingestion engine, schema converter
│   │   ├── tum_reconciler.py     <- Canonical matching & delta auditor
│   │   ├── canonical_registry.py <- Canonical entity lookup integration
│   │   ├── schemas.py            <- 4 PyArrow schemas for TUM tables
│   │   └── storage.py            <- Storage path resolution helpers
│   └── features/
│       └── circuit_geometry.py   <- Differential geometry, corner detection
├── tests/
│   └── test_tum_integration.py   <- 16 targeted unit & integration tests
└── docs/
    └── phase5_5_tum_integration_report.md
```

### 5.1 PyArrow Data Schemas (`src/data/schemas.py`)
1. `TUM_TRACK_GEOMETRY_SCHEMA`:
   - `circuit_id` (string), `tum_circuit_name` (string), `point_index` (int32)
   - `s_m` (float64), `x_m` (float64), `y_m` (float64)
   - `w_tr_right_m` (float64), `w_tr_left_m` (float64), `w_total_m` (float64)
   - `psi_rad` (float64), `kappa_radpm` (float64), `radius_m` (float64)
   - `source_api` (string), `retrieved_at` (string)
2. `TUM_RACELINE_SCHEMA`:
   - `circuit_id` (string), `tum_circuit_name` (string), `point_index` (int32)
   - `s_m` (float64), `x_m` (float64), `y_m` (float64)
   - `source_api` (string), `retrieved_at` (string)
3. `TUM_CIRCUIT_SUMMARY_SCHEMA`:
   - Master record per circuit containing point counts, lengths, widths, closure error, corner counts by speed category, straight breakdown, curvature spectrum, and provenance.
4. `TUM_CORNER_PROFILE_SCHEMA`:
   - Corner table recording corner index, direction (`left`/`right`), speed category (`low`/`medium`/`high`), start/apex/end arc length, length, apex radius, and minimum road width.

---

## 6. Normalization & Derived-Feature Methodology

### 6.1 Step Arc Length & Track Closure
For discrete coordinates $(x_i, y_i)_{i=0}^{N-1}$:
$$\Delta x_i = x_{i+1} - x_i, \quad \Delta y_i = y_{i+1} - y_i \quad (\text{with } x_N = x_0, y_N = y_0)$$
$$\Delta s_i = \sqrt{\Delta x_i^2 + \Delta y_i^2}, \quad s_i = \sum_{j=0}^{i-1} \Delta s_j, \quad L_{\text{track}} = \sum_{i=0}^{N-1} \Delta s_i$$
The closure error $\Delta_{\text{closure}}$ is the Euclidean distance between point $N-1$ and point $0$:
$$\Delta_{\text{closure}} = \sqrt{(x_{N-1} - x_0)^2 + (y_{N-1} - y_0)^2}$$
Across all 25 tracks, $\Delta_{\text{closure}} \approx 5.00\text{ m}$, exactly matching the nominal sampling resolution $\Delta s = 5.0\text{ m}$.

### 6.2 Continuous Heading Angle
$$\psi_i = \text{atan2}(\Delta y_i, \Delta x_i)$$
Periodic circular wrap-around is handled via circular difference modulo $[-\pi, \pi]$:
$$\Delta \psi_i = ((\psi_{i+1} - \psi_{i-1} + \pi) \pmod{2\pi}) - \pi$$

### 6.3 Curvature & Curvature Radius
Curvature $\kappa(s)$ is estimated via 3-point periodic central differences:
$$\kappa_i = \frac{\Delta \psi_i}{\Delta s_i + \Delta s_{i-1}} \quad (\text{rad/m})$$
Radius of curvature $R_i$:
$$R_i = \begin{cases} \frac{1}{|\kappa_i|} & \text{if } |\kappa_i| > 10^{-6} \\ 10^6\text{ m} & \text{otherwise} \end{cases}$$

### 6.4 Corner Detection & Speed Categorization
- **Corner Inclusion Criteria:** Contiguous segments where $R_i \le 300.0\text{ m}$ of length $\ge 15.0\text{ m}$ (at least 3 sample points).
- **Apex Property Identification:** Apex is the localized curvature extremum within the corner:
  $$\text{Apex} = \arg\max_{i \in \text{Corner}} |\kappa_i|$$
- **Turn Direction:** Positive curvature ($\kappa > 0$) corresponds to left turns; negative curvature ($\kappa < 0$) corresponds to right turns.
- **Speed Categories:**
  - **Low-Speed:** $R_{\text{apex}} < 75.0\text{ m}$ (hairpins, chicanes, slow 90° corners)
  - **Medium-Speed:** $75.0\text{ m} \le R_{\text{apex}} \le 150.0\text{ m}$ (technical sweeping turns)
  - **High-Speed:** $R_{\text{apex}} > 150.0\text{ m}$ (fast bends, long sweeps)
- **Straights:** Contiguous segments with $R_i > 300.0\text{ m}$ spanning $\ge 80.0\text{ m}$.

---

## 7. Canonical Circuit Matching Results

```
┌────────────────┬────────────────┬──────────────────────────────┬────────────┬─────────────┐
│ TUM Stem       │ Canonical ID   │ Official Track Name          │ Match Type │ Confidence  │
├────────────────┼────────────────┼──────────────────────────────┼────────────┼─────────────┤
│ Austin         │ americas       │ Circuit of the Americas      │ tum_id     │ High (100%) │
│ Budapest       │ hungaroring    │ Hungaroring                  │ tum_id     │ High (100%) │
│ Catalunya      │ catalunya      │ Circuit de Barcelona-Catalun.│ tum_id     │ High (100%) │
│ Hockenheim     │ hockenheimring │ Hockenheimring               │ tum_id     │ High (100%) │
│ Melbourne      │ albert_park    │ Albert Park GP Circuit       │ tum_id     │ High (100%) │
│ MexicoCity     │ rodriguez      │ Autódromo Hermanos Rodríguez │ tum_id     │ High (100%) │
│ Montreal       │ villeneuve     │ Circuit Gilles Villeneuve    │ tum_id     │ High (100%) │
│ Monza          │ monza          │ Autodromo Nazionale di Monza │ tum_id     │ High (100%) │
│ Nuerburgring   │ nurburgring    │ Nürburgring                  │ tum_id     │ High (100%) │
│ Sakhir         │ bahrain        │ Bahrain International Circuit│ tum_id     │ High (100%) │
│ SaoPaulo       │ interlagos     │ Autódromo José Carlos Pace   │ tum_id     │ High (100%) │
│ Sepang         │ sepang         │ Sepang International Circuit │ tum_id     │ High (100%) │
│ Shanghai       │ shanghai       │ Shanghai International Circ. │ tum_id     │ High (100%) │
│ Silverstone    │ silverstone    │ Silverstone Circuit          │ tum_id     │ High (100%) │
│ Sochi          │ sochi          │ Sochi Autodrom               │ tum_id     │ High (100%) │
│ Spa            │ spa            │ Circuit de Spa-Francorchamps │ tum_id     │ High (100%) │
│ Spielberg      │ red_bull_ring  │ Red Bull Ring                │ tum_id     │ High (100%) │
│ Suzuka         │ suzuka         │ Suzuka Circuit               │ tum_id     │ High (100%) │
│ YasMarina      │ yas_marina     │ Yas Marina Circuit           │ tum_id     │ High (100%) │
│ Zandvoort      │ zandvoort      │ Circuit Park Zandvoort       │ tum_id     │ High (100%) │
├────────────────┼────────────────┼──────────────────────────────┼────────────┼─────────────┤
│ BrandsHatch    │ brands_hatch   │ Brands Hatch (DTM / Historic)│ fallback   │ Non-Canon.  │
│ IMS            │ indianapolis   │ Indianapolis Motor Speedway  │ fallback   │ Non-Canon.  │
│ MoscowRaceway  │ moscow_raceway │ Moscow Raceway (DTM)         │ fallback   │ Non-Canon.  │
│ Norisring      │ norisring      │ Norisring Nuremberg (DTM)    │ fallback   │ Non-Canon.  │
│ Oschersleben   │ oschersleben   │ Motorsport Arena Oschersleben│ fallback   │ Non-Canon.  │
└────────────────┴────────────────┴──────────────────────────────┴────────────┴─────────────┘
```

---

## 8. Data-Quality & Reconciliation Findings

### 8.1 Track Length Comparison & Discrepancy Analysis
Reconciliation between TUM centerline closed lengths and official FIA/PACE canonical track lengths:

| Canonical ID | TUM Name | TUM Length (m) | PACE Length (m) | Delta (m) | Delta (%) | Status | Evaluation Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|
| `americas` | Austin | 5,507.54 | 5,513.00 | -5.46 | -0.10% | **EXCELLENT** | Typical apex clipping variance |
| `hungaroring` | Budapest | 4,376.86 | 4,381.00 | -4.14 | -0.09% | **EXCELLENT** | Sub-5m precision |
| `catalunya` | Catalunya | 4,649.84 | 4,657.00 | -7.16 | -0.15% | **EXCELLENT** | Pre-2023 chicane removal layout |
| `hockenheimring` | Hockenheim | 4,569.20 | 4,574.00 | -4.80 | -0.10% | **EXCELLENT** | Excellent centerline alignment |
| `albert_park` | Melbourne | 5,298.74 | 5,278.00 | +20.74 | +0.39% | **GOOD** | Pre-2022 reprofiling delta (+20.7m) |
| `rodriguez` | MexicoCity | 4,297.20 | 4,304.00 | -6.80 | -0.16% | **EXCELLENT** | Sub-7m variance |
| `villeneuve` | Montreal | 4,357.51 | 4,361.00 | -3.49 | -0.08% | **EXCELLENT** | High precision |
| `monza` | Monza | 5,790.20 | 5,793.00 | -2.80 | -0.05% | **EXCELLENT** | Exact alignment with FIA timing |
| `nurburgring` | Nuerburgring | 5,144.11 | 5,148.00 | -3.89 | -0.08% | **EXCELLENT** | High precision |
| `bahrain` | Sakhir | 5,405.75 | 5,412.00 | -6.25 | -0.12% | **EXCELLENT** | Sub-7m variance |
| `interlagos` | SaoPaulo | 4,304.62 | 4,309.00 | -4.38 | -0.10% | **EXCELLENT** | Sub-5m variance |
| `sepang` | Sepang | 5,537.35 | 5,543.00 | -5.65 | -0.10% | **EXCELLENT** | Sub-6m variance |
| `shanghai` | Shanghai | 5,445.25 | 5,451.00 | -5.75 | -0.11% | **EXCELLENT** | Sub-6m variance |
| `silverstone` | Silverstone | 5,886.80 | 5,891.00 | -4.20 | -0.07% | **EXCELLENT** | High precision |
| `sochi` | Sochi | 5,841.09 | 5,848.00 | -6.91 | -0.12% | **EXCELLENT** | Sub-7m variance |
| `spa` | Spa | 7,000.05 | 7,004.00 | -3.95 | -0.06% | **EXCELLENT** | Sub-4m precision on 7km track |
| `red_bull_ring`| Spielberg | 4,315.45 | 4,318.00 | -2.55 | -0.06% | **EXCELLENT** | Sub-3m variance |
| `suzuka` | Suzuka | 5,802.88 | 5,807.00 | -4.12 | -0.07% | **EXCELLENT** | Sub-5m variance |
| `yas_marina` | YasMarina | 5,546.57 | 5,281.00 | +265.57 | +5.03% | **RECONFIGURED** | **Pre-2021 layout** with chicane |
| `zandvoort` | Zandvoort | 4,316.48 | 4,259.00 | +57.48 | +1.35% | **RECONFIGURED** | **Pre-2020 layout** prior to banking |

### 8.2 Key Reconciliation Conclusions
1. **High Geometric Fidelity:** 17 out of 20 circuits ($85\%$) exhibit sub-$15\text{ m}$ length delta ($\le 0.15\%$). This quantitatively demonstrates that TUM centerline smoothing preserves true circuit distance with remarkable accuracy.
2. **Identification of Layout Reconfigurations:**
   - **Yas Marina (`yas_marina`):** TUM measures $5,546.6\text{ m}$, matching the pre-2021 layout which featured the slow chicane before the north hairpin and tight chicanes around the hotel. For modern post-2021 simulation, the canonical length ($5,281.0\text{ m}$) remains authoritative.
   - **Zandvoort (`zandvoort`):** TUM measures $4,316.5\text{ m}$, representing the pre-2020 historic layout before Turn 3 and Arie Luyendykbocht banked reprofiling ($4,259.0\text{ m}$).
   - **Melbourne (`albert_park`):** TUM measures $5,298.7\text{ m}$ (+20.7m delta), reflecting the pre-2022 layout prior to the removal of the Turns 9/10 chicane.

---

## 9. Command-Line Interface (CLI) Usage Examples

All six required CLI commands are fully functional within `main.py`:

```bash
# 1. Inspect repository status, commit hash, author, and license
python main.py --tum-status

# 2. Display inventory of all 25 raw circuit CSVs
python main.py --tum-inventory

# 3. Ingest, process, and persist all circuit geometries to Parquet
python main.py --tum-ingest

# 4. Display circuit coverage matrix and calendar gaps
python main.py --tum-circuits

# 5. Display detailed differential geometry & corner profiles for a circuit
python main.py --tum-geometry --circuit monza
python main.py --tum-geometry --circuit americas
python main.py --tum-geometry --circuit spa

# 6. Run comprehensive cross-source reconciliation against canonical registry
python main.py --tum-reconcile
```

---

## 10. Test Results

### 10.1 Targeted TUM Integration Suite (`tests/test_tum_integration.py`)
```
============================= test session starts ==============================
collected 16 items

tests/test_tum_integration.py::test_tum_client_metadata PASSED           [  6%]
tests/test_tum_integration.py::test_tum_client_list_tracks PASSED        [ 12%]
tests/test_tum_integration.py::test_tum_client_load_raw_track PASSED     [ 18%]
tests/test_tum_integration.py::test_tum_client_load_raw_raceline PASSED  [ 25%]
tests/test_tum_integration.py::test_tum_client_track_not_found PASSED    [ 31%]
tests/test_tum_integration.py::test_tum_client_offline_mode_guard PASSED [ 37%]
tests/test_tum_integration.py::test_geometry_processor_step_spacing_and_closure PASSED [ 43%]
tests/test_tum_integration.py::test_geometry_processor_curvature_and_heading PASSED [ 50%]
tests/test_tum_integration.py::test_geometry_processor_corner_detection PASSED [ 56%]
tests/test_tum_integration.py::test_geometry_processor_curvature_spectrum PASSED [ 62%]
tests/test_tum_integration.py::test_geometry_processor_raceline PASSED   [ 68%]
tests/test_tum_integration.py::test_ingest_single_circuit_schemas PASSED [ 75%]
tests/test_tum_integration.py::test_ingest_all_and_summary_schema PASSED [ 81%]
tests/test_tum_integration.py::test_tum_canonical_matching_coverage PASSED [ 87%]
tests/test_tum_integration.py::test_tum_reconciliation_accuracy_and_reconfigurations PASSED [ 93%]
tests/test_tum_integration.py::test_circuit_geometry_loader_methods PASSED [100%]

============================== 16 passed in 1.56s ==============================
```

### 10.2 Full System Regression Suite
```
======================= 154 passed, 1 warning in 55.15s ========================
```
Zero regressions across all existing canonical entity, Kaggle historical, Jolpica championship, OpenF1 intelligence, feature engineering, and Monte Carlo simulation components.

---

## 11. Known Limitations & Operating Boundaries

1. **Two-Dimensional Planar Coordinates:** The TUM database coordinates are strictly 2D $(x, y)$. No vertical elevation ($z$) or grade/camber measurements are present. Downstream models must not treat TUM data as 3D topography.
2. **Calendar Age & Layout Divergence:** The repository commit is dated September 2021. Circuits altered after 2021 (Yas Marina, Melbourne Albert Park, Barcelona chicane removal) represent their 2021 configurations.
3. **Calendar Gaps:** Twelve modern F1 championship circuits are not included in TUM (e.g. Las Vegas, Monaco, Baku, Miami). Downstream models use canonical circuit metadata (`CIRCUIT_METRICS`) for these tracks.
4. **No Direct Corner Numbering:** Corner numbers are inferred from localized curvature extrema, not official FIA track map turn numbers.
5. **Offline Operation:** The pipeline functions completely offline once `data/raw/tum/repo` is present. No runtime external network calls are required.

---

## 12. Recommended Next Steps

1. **Phase 6 Performance Engineering:** Integrate derived track curvature profiles with FastF1 corner apex speed telemetry in `src/analytics/telemetry.py` to compare driver apex minimum speeds against theoretical curvature limits.
2. **Simulation Braking Zones:** Utilize the detected straight-to-corner transitions to calibrate circuit-specific braking hazard zones in `IncidentEngine`.
3. **Las Vegas Grand Prix Calibration:** For Stage 6.3 / 8.3, calibrate the Las Vegas Strip Circuit using official FOM circuit geometry specifications (length: $6.201\text{ km}$, 17 turns, $1.9\text{ km}$ flat-out Strip straight) rather than fabricating unofficial TUM data.
