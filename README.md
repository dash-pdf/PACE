# PACE: Predictive Analytics for Competitive Evaluation

An engineering-driven Formula 1 race prediction system that predicts the finishing positions of all participating drivers using historical performance, practice sessions, qualifying data, statistical modelling, machine learning, and full race simulation.

---

## 1. Project Introduction

**PACE** (Predictive Analytics for Competitive Evaluation) is designed to move beyond simplistic statistical averages or purely subjective commentary by delivering an empirical, quantitative framework for Formula 1 race outcome forecasting. By fusing high-resolution telemetry, timing loops, environmental variables, and historical driver/constructor benchmarks, PACE builds a layered predictive architecture from Friday practice through Sunday chequered flag.

---

## 2. Project Objectives

- **Finishing Position Forecasting:** Predict the complete classification (P1 to P20 / DNF probabilities) for all participating drivers.
- **Multi-Factor Feature Synthesis:** Integrate driver form, constructor development trajectory, car characteristics, tyre degradation, and circuit-specific historical intelligence.
- **Session Evolution Tracking:** Progressively refine prediction distributions as real-world data accumulates from FP1, FP2, FP3, and Qualifying sessions.
- **Stochastic Race Simulation:** Model dynamic race scenarios (safety cars, pit strategies, overtakes, weather transitions, and mechanical failures) via Monte Carlo simulations.
- **Rigorous Evaluation:** Benchmark model performance against baseline heuristics (e.g., qualifying grid order) using rank correlation, mean absolute error, and top-3 accuracy.

---

## 3. Main Analytical Components

PACE is structured around seven analytical engines:

1. **Driver Performance Analysis:** Quantifies driver baseline pace, consistency, wet-weather aptitude, wheel-to-wheel racecraft, and recent momentum.
2. **Constructor & Car Performance Analysis:** Evaluates aero efficiency, high/medium/low-speed cornering speed, straight-line top speed, reliability indices, and upgrade trends.
3. **Driver–Circuit Intelligence:** Models track compatibility, historical overtaking difficulty, degradation profiles, and driver-specific performance history at each circuit.
4. **Practice & Qualifying Telemetry Analysis:** Extracts true race pace from long runs (fuel-adjusted) and single-lap qualifying performance.
5. **Hybrid Statistical & ML Prediction Models:** Combines gradient boosting, ensemble regressors/classifiers, and probabilistic ranking models.
6. **Stochastic Race Simulation Engine:** Executes thousands of simulated race instances modeling tyre degradation curves, pit stop execution, strategy divergence, safety car probabilities, and collision incidents.
7. **Post-Race Performance Evaluation:** Assesses prediction calibration, identifies divergence sources, and updates continuous rating models (e.g., Glicko/Elo variations).

---

## 4. Proposed System Architecture

```
                                  [ Historical Data & APIs ]
                                   (FastF1 / Ergast / OpenF1)
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │   src/data      │
                                      │ Ingestion & ETL │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  src/features   │
                                      │ Feature Pipeline│
                                      └────────┬────────┘
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
             ┌─────────────────────┐                       ┌─────────────────────┐
             │     src/models      │                       │   src/simulation    │
             │ Machine Learning &  │                       │ Stochastic Monte    │
             │ Probabilistic Rank  │                       │ Carlo Race Engine   │
             └──────────┬──────────┘                       └──────────┬──────────┘
                        │                                             │
                        └──────────────────────┬──────────────────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  src/evaluation │
                                      │  & src/reporting│
                                      └─────────────────┘
```

---

## 5. Technology Stack

- **Language:** Python 3.12+
- **F1 Telemetry & Timing Data:** `FastF1`
- **Data Manipulation & Storage:** `pandas`, `numpy`, `scipy`, `pyarrow` (Parquet)
- **Machine Learning & Statistical Modelling:** `scikit-learn`, `xgboost`
- **Data Visualisation:** `matplotlib`, `seaborn`
- **Testing & QA:** `pytest`

---

## 6. Development Roadmap

| Phase | Milestone | Status | Description |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Repository Creation & Project Foundation** | **Completed** | Project scaffolding, licensing, Git repository setup, virtual environment, and dependency configuration. |
| **Phase 2** | **Historical Data Infrastructure** | **Completed** | Robust data ingestion, raw/processed Parquet storage, multi-season compilation (2023–2026), and quality validation gates. |
| **Phase 3** | **Driver & Constructor Performance Modelling** | Planned | Baseline performance indices, rolling driver form, car developmental trajectory, and telemetry analysis. |
| **Phase 4** | **Feature Engineering Pipeline** | Planned | Calculation of driver form ratings, car performance indices, tyre degradation curves, and circuit compatibility vectors. |
| **Phase 5** | **Baseline & Machine Learning Models** | Planned | Implementation of baseline heuristics, regression models, and gradient-boosted ranking algorithms. |
| **Phase 6** | **Stochastic Race Simulation Engine** | Planned | Monte Carlo simulation modelling tyre wear, pit stops, safety car periods, and stochastic incidents. |
| **Phase 7** | **Evaluation & Backtesting Framework** | Planned | Historical backtesting, calibration assessment, and predictive accuracy metrics. |
| **Phase 8** | **Reporting, Visualisation & Automation** | Planned | Automated pre-race prediction reports, probability distributions, and post-race accuracy evaluations. |

---

## 7. Current Development Status

- **Current Phase:** Phase 2 (Historical Data Infrastructure) — Completed
- **Implemented Functionality:**
  - Automated data ingestion for seasons 2023–2026 using FastF1 Ergast API and OpenF1 feeds.
  - Three-tier storage architecture (`data/raw/`, `data/processed/`, `data/historical/`) with Parquet compression.
  - Comprehensive schemas covering drivers, constructors, circuits, events, sessions, qualifying, race results, sprints, laps, pit stops, and practice runs.
  - Automated quality validation gates verifying referential integrity, position validity, DNF/DNS/DSQ status flags, and coordinate ranges.
  - Command-line interface with `--collect-data`, `--validate-data`, `--seasons`, and `--force` support.
  - Comprehensive test suite with 17 unit/integration tests verifying schemas, storage, and validation gates.
- **Planned Functionality:**
  - Prediction models, driver ratings, and simulation engines will be built on top of this infrastructure in Phase 3 onwards.

---

## 8. Setup & Installation

### Prerequisites
- Python 3.12 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dash-pdf/PACE.git
   cd PACE
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation & run test suite:**
   ```bash
   pytest
   python main.py --status
   ```

5. **Execute historical data collection:**
   ```bash
   python main.py --collect-data
   ```

6. **Validate dataset quality gates:**
   ```bash
   python main.py --validate-data
   ```

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
