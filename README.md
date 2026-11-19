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
| **Phase 2** | **Historical Data Infrastructure** | **Completed** | Multi-season ingestion (2023–2026), full career Driver–Circuit Historical Intelligence (2001–2026), circuit characteristics, teammate matching, Parquet storage, and quality validation gates. |
| **Phase 3** | **Driver & Constructor Performance Modelling** | Planned | Baseline performance indices, rolling driver form, car developmental trajectory, and telemetry analysis. |
| **Phase 4** | **Feature Engineering Pipeline** | Planned | Calculation of driver form ratings, car performance indices, tyre degradation curves, and circuit compatibility vectors. |
| **Phase 5** | **Baseline & Machine Learning Models** | Planned | Implementation of baseline heuristics, regression models, and gradient-boosted ranking algorithms. |
| **Phase 6** | **Stochastic Race Simulation Engine** | Planned | Monte Carlo simulation modelling tyre wear, pit stops, safety car periods, and stochastic incidents. |
| **Phase 7** | **Evaluation & Backtesting Framework** | Planned | Historical backtesting, calibration assessment, and predictive accuracy metrics. |
| **Phase 8** | **Reporting, Visualisation & Automation** | Planned | Automated pre-race prediction reports, probability distributions, and post-race accuracy evaluations. |

---

## 7. Current Development Status

- **Current Phase:** Phase 2 (Historical Data Infrastructure & Driver–Circuit Intelligence) — Completed
- **Implemented Functionality:**
  - **General Seasonal Data (2023–2026):** Ingestion using FastF1 Ergast API and OpenF1 feeds covering race results, qualifying, sprints, practice telemetry (sector times, speed traps), and pit stops.
  - **Driver–Circuit Historical Intelligence:** Full career race and qualifying records spanning 2001–2026 across 39 Grand Prix circuits for modern drivers and key historical teammates, preserving granular event-level records alongside aggregated performance profiles.
  - **Technical Circuit Characteristics:** Engineering metrics (track length, typical laps, altitude, overtaking difficulty, tyre stress, downforce level, first GP) for all 39 world championship circuits.
  - **Teammate Head-to-Head Matching:** Event-by-event teammate comparison in both race and qualifying sessions, preserving head-to-head win counts and win rates.
  - **Regulation Era Context:** Automatic era classification (`ground_effect_2022_2026`, `turbo_hybrid_2014_2021`, `v8_2006_2013`, `v10_1996_2005`, `early_pre_1996`).
  - **Resumable Three-Tier Storage:** Raw API JSON caches (`data/raw/`), seasonal Parquet tables (`data/processed/`), and compiled master historical datasets (`data/historical/`).
  - **Automated Quality Validation Gates:** 9 quality validation suites verifying referential integrity, position validity, DNF/DNS/DSQ mutual exclusivity, and statistical bounds.
  - **Command-Line Interface:** Complete CLI support via `main.py` with `--collect-circuit-history`, `--collect-data`, `--validate-data`, `--drivers`, `--seasons`, and `--force`.
  - **Comprehensive Test Suite:** 24 unit and integration tests verifying schemas, storage, and validation gates with 100% pass rate.
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
