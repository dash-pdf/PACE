# PACE Phase 3: Driver & Constructor Performance Modelling Methodology

**Predictive Analytics for Competitive Evaluation**  
**Document Version:** 1.0.0 (Stage 1 Architecture)  
**Author:** PACE Engineering Team  
**Status:** Under Review (Methodology Sign-Off)

---

## 1. Executive Summary

Formula 1 race finishing positions are governed by a complex, interdependent hierarchy of factors:
1. **Car Competitiveness (Constructor):** Aerodynamic efficiency, power unit output, chassis balance, tyre degradation, and mechanical reliability.
2. **Driver Capability:** Single-lap qualifying execution, wheel-to-wheel racecraft, tyre management, lap time consistency, and adaptability.
3. **Driver–Circuit Historical Intelligence:** Track-specific affinity, driving style compatibility with circuit layout, and historical confidence.
4. **Context & Circuit Topology:** Overtaking difficulty, downforce requirements, ambient altitude, temperature, tyre stress, and safety car probability.
5. **Session Evolution & Grid Position:** Knockout qualifying starting position, clean vs dirty side of the grid, and Friday practice long-run race simulations.

Phase 3 establishes the mathematical and statistical foundation for PACE. Rather than relying on naive averages, arbitrary percentage weightings, or data-leaking predictors, PACE formulates an empirical, hybrid predictive architecture:
- **Separation of Concerns:** Independent quantification of the **Driver Performance Index (DPI)** and **Constructor Performance Index (CPI)** using network-connected teammate comparisons and telemetry decomposition.
- **Hierarchical Driver–Circuit Intelligence:** Empirical Bayes shrinkage modeling track-specific performance while preventing small-sample distortion for rookies or debutants.
- **Context-Aware Dynamic Interaction:** Machine learning models that learn dynamic non-linear feature interactions conditioned on physical circuit characteristics.
- **Chronological Walk-Forward Backtesting:** Strictly leakage-free evaluation benchmarked against qualifying grid order and simple heuristics.

---

## 2. Driver Performance Index (DPI)

### 2.1 The Fundamental Challenge of Separating Driver from Car
In Formula 1, a driver's raw finishing position is heavily confounded by the performance of the constructor's car. A top-tier driver in an uncompetitive car will frequently finish behind an average driver in a championship-winning car.

The only direct "controlled experiment" in Grand Prix racing occurs between teammates competing in identical machinery during the same event under identical weather and track conditions. However, teammates cannot be evaluated in isolation: teammates change across seasons, creating an interconnected bipartite graph across the entire grid.

### 2.2 Mathematical Formulations

#### A. Teammate-Relative Qualifying Delta ($\Delta Q_{i,j,t}$)
For driver $i$ and teammate $j$ at event $t$, the qualifying pace delta is defined using the best mutually completed qualifying session (Q3, Q2, or Q1):
$$\Delta Q_{i,j,t} = \frac{T_{i,t}^{qual} - T_{j,t}^{qual}}{\min(T_{i,t}^{qual}, T_{j,t}^{qual})}$$
- $\Delta Q_{i,j,t} < 0$: Driver $i$ was faster than teammate $j$.
- In addition, the binary head-to-head metric is captured:
$$H2H_{i,j,t}^{qual} = \mathbb{I}(QPos_{i,t} < QPos_{j,t})$$

#### B. Teammate-Relative Race Delta ($\Delta R_{i,j,t}$)
When both teammates are classified at event $t$:
$$\Delta R_{i,j,t} = Pos_{j,t} - Pos_{i,t}$$
$$\text{H2H}_{i,j,t}^{race} = \mathbb{I}(Pos_{i,t} < Pos_{j,t})$$
When a retirement occurs:
- If driver $j$ suffers a mechanical failure or non-driver DNF, the event is treated as non-comparable (masked as null for H2H).
- If driver $i$ commits an unforced driver error / crash, $H2H_{i,j,t}^{race} = 0$.

#### C. Networked Rating Framework (Bayesian Glicko-2)
To propagate relative performance across teams and eras, PACE implements a Glicko-2 rating engine where each Grand Prix session acts as a pairwise match between teammates.
Each driver maintains:
- Rating $\mu_i$: Latent driver skill (centered at 1500).
- Rating Deviation $\phi_i$: Measure of uncertainty. For veterans with 200+ races (Hamilton, Alonso), $\phi_i$ is small; for rookies (Lindblad, Antonelli), $\phi_i$ is wider.
- Rating Volatility $\sigma_i$: Measure of performance variance over time.

The win probability of driver $i$ over driver $j$ is given by:
$$E(s \mid \mu_i, \mu_j, \phi_j) = \frac{1}{1 + 10^{-g(\phi_j)(\mu_i - \mu_j) / 400}}$$
where $g(\phi) = \frac{1}{\sqrt{1 + 3q^2\phi^2 / \pi^2}}$ and $q = \frac{\ln(10)}{400}$.

#### D. Machinery-Adjusted Position Progress ($\Delta Pos_{adj}$)
Raw positions gained ($Grid - Finish$) penalize drivers starting at the front (P1 can gain at most 0 positions, while P20 can gain 19 positions). PACE defines Machinery-Adjusted Progress by conditioning on starting position:
$$\Delta Pos_{adj}(i,t) = (Grid_{i,t} - Finish_{i,t}) - \mathbb{E}[\Delta Pos \mid Grid_{i,t}, Era_t]$$
where $\mathbb{E}[\Delta Pos \mid Grid, Era]$ is the empirical expected position change calculated from thousands of historical races within the corresponding regulation era.

#### E. Lap Time Consistency & Pace Variation ($CV_{pace}$)
Using lap times from race stints and practice sessions (filtering out pit in/out laps, yellow flag, and safety car periods):
$$CV_{pace}(i,t) = \frac{\sigma(T_{i,t}^{clean})}{\mu(T_{i,t}^{clean})}$$
Drivers with low $CV_{pace}$ demonstrate superior tyre preservation and lap-to-lap rhythm.

#### F. Career Progression & Experience Curve
Driver skill develops non-linearly with race starts:
$$Exp_{i,t} = \ln(1 + N_{starts}(i,t))$$
This logarithmic transformation models the steep initial rookie learning curve that asymptotically plateaus as experience surpasses 50+ starts.

---

## 3. Constructor Performance Index (CPI)

### 3.1 Isolating Car Competitiveness
A constructor's performance index must capture how fast the chassis and power unit can go, how well the car treats its tyres over long race distances, how aerodynamically efficient it is across different corner speed regimes, and how mechanically reliable it is.

### 3.2 Mathematical Formulations

#### A. Single-Lap Qualifying Pace Competitiveness
The car's peak single-lap capability is defined by the best lap achieved by either car relative to pole position:
$$\Delta T_{c,t}^{pole} = \frac{\min_{i \in c}(T_{i,t}^{qual}) - T_{pole,t}}{T_{pole,t}}$$
Expressing this as a percentage gap standardizes across circuits of different lengths (e.g. 1m10s at Spielberg vs 1m45s at Spa).

#### B. Sector Decomposition & Aerodynamic Efficiency Ratios
Using split times from OpenF1 practice telemetry (`historical_practice_laps.parquet`):
- **Straight-Line Speed Efficiency ($ST_{speed}$):** Median trap speed relative to the field maximum:
  $$Eff_{straight}(c,t) = \frac{\text{median}(st\_speed_{c,t})}{\max_{k}(\text{median}(st\_speed_{k,t}))}$$
- **High-Speed Downforce Index ($S_{high}$):** Relative time in high-lateral-g sectors (e.g. Silverstone Becketts, Suzuka Esses).
- **Low-Speed Mechanical Grip Index ($S_{low}$):** Relative time in low-speed traction sectors (e.g. Monaco Rascasse, Montreal Hairpin).

#### C. Tyre Degradation Rate ($k_{deg}$)
In FP2 race simulation runs and completed race stints, lap times follow a characteristic degradation slope:
$$T_{c,t}(l) = T_{c,0} + k_{deg} \cdot l + k_{fuel} \cdot (L_{tot} - l)$$
where $k_{fuel} \approx -0.06 \text{ s/lap}$ accounts for the decreasing mass of fuel burn-off. The constructor tyre degradation coefficient $k_{deg}$ (seconds of pace loss per lap of tyre age) is estimated via robust linear regression (Huber loss).

#### D. Mechanical Reliability Index ($Rel_c$)
Reliability is modeled as an exponentially weighted moving average of completed starts without mechanical failure:
$$Rel_c(t) = 1.0 - \frac{\sum_{\tau < t} e^{-\lambda(t-\tau)} \cdot \mathbb{I}(\text{Mechanical DNF}_{c,\tau})}{\sum_{\tau < t} e^{-\lambda(t-\tau)} \cdot \text{Starts}_{c,\tau}}$$
where $\lambda$ corresponds to a 10-race half-life, ensuring recent mechanical failures (e.g. engine blowouts, hydraulic leaks) lower the constructor's reliability score.

#### E. Recent Form and In-Season Development Slope
Constructors introduce upgrades throughout the season. The developmental trajectory is captured by the rolling linear trend of $\Delta T_{c,t}^{pole}$ over the previous 5 rounds:
$$\beta_{dev}(c,t) = \text{Slope}\left(\Delta T_{c,t-5}^{pole}, \dots, \Delta T_{c,t-1}^{pole}\right)$$
A negative slope indicates an accelerating car closing the gap to the front.

---

## 4. Driver–Circuit Intelligence Integration

### 4.1 The Small Sample Size Dilemma
Historical circuit datasets present an inherent statistical challenge:
- Veteran drivers (Hamilton, Alonso) have 15–20 appearances at classic tracks (Silverstone, Monza, Monaco).
- Sophomores (Antonelli, Bortoleto, Hadjar) have 1–2 appearances.
- Debutants (Lindblad, Hirakawa, Câmara) have 0–1 appearances.

If a driver finished P3 in their only career appearance at a track, a naive average would treat them as having an expected finish of 3.0 at that venue. Conversely, a veteran finishing P12 once in an uncompetitive car should not erase ten podiums.

### 4.2 Empirical Bayes Shrinkage Formulation
PACE resolves this using **Empirical Bayes Partial Pooling**. A driver's circuit-specific expected finishing performance $\widehat{\mu}_{i,c}$ is shrunk toward their global career baseline $\mu_{i}^{global}$:
$$\widehat{\mu}_{i,c} = \alpha_{i,c} \cdot \bar{y}_{i,c} + (1 - \alpha_{i,c}) \cdot \mu_{i}^{global}$$
where the shrinkage factor $\alpha_{i,c} \in [0, 1]$ is:
$$\alpha_{i,c} = \frac{N_{i,c}}{N_{i,c} + \kappa_0}$$
- $N_{i,c}$: Total recorded starts of driver $i$ at circuit $c$.
- $\bar{y}_{i,c}$: Sample mean finishing position of driver $i$ at circuit $c$.
- $\mu_{i}^{global}$: Overall career mean finishing position of driver $i$.
- $\kappa_0$: Empirical shrinkage hyperparameter ($\kappa_0 \approx 5.0$).

#### Shrinkage Properties:
- **$N_{i,c} = 1$ (Rookie single appearance):** $\alpha = \frac{1}{1 + 5} = 0.167$. 83.3% of the estimate is anchored to the driver's global capability, preventing overreaction to a single outlier.
- **$N_{i,c} = 15$ (Veteran track record):** $\alpha = \frac{15}{15 + 5} = 0.750$. 75.0% of the estimate is driven by the circuit track record.

### 4.3 Technical Regulation Era Decay
Circuit appearances are decayed according to technical era relevance:
$$w_{era} = \begin{cases} 
1.00 & \text{Ground Effect (2022–2026)} \\ 
0.60 & \text{Turbo-Hybrid (2014–2021)} \\ 
0.25 & \text{V8 Era (2006–2013)} \\ 
0.10 & \text{Pre-2006} 
\end{cases}$$

### 4.4 Circuit Characteristic Affinity
PACE matches driver and car performance against the technical attributes in `circuit_characteristics.parquet`:
- **Street Circuit Affinity:** Track record on low-grip, high-consequence temporary barriers (Monaco, Baku, Singapore, Vegas).
- **High Altitude Factor:** Aerodynamic density and turbocharger stress (Mexico City at 2,240m elevation vs Zandvoort at sea level).
- **Overtaking Difficulty Factor:** Track design classifying overtaking feasibility into `Low`, `Medium`, and `High`.

---

## 5. Context-Dependent Metric Weighting Architecture

### 5.1 Hybrid Domain-Guided & Machine-Learned Weighting
A central flaw of simplistic prediction models is adopting static, arbitrary weighting formulas (e.g., "always 40% car + 30% qualy + 20% driver + 10% circuit"). In reality:
- At **Monaco** (Overtaking Difficulty: High, narrow streets), Qualifying Grid Position determines >80% of finishing position outcome.
- At **Spa-Francorchamps** or **Baku** (Overtaking Difficulty: Low, long DRS straights), Qualifying position accounts for <50% of outcome, and straight-line speed, engine power, and tyre degradation dominate.

PACE addresses this through a **two-tier hybrid weighting architecture**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │          Engineering Domain Priors & Bounds            │
                       │   (Monotonicity, Min/Max Limits, Feature Hierarchy)   │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
┌───────────────────────┐              ┌──────────────────────┐             ┌─────────────────────────┐
│     Driver State      │              │   Constructor State  │             │   Circuit & Conditions  │
│  • Latent DPI Rating  │              │  • Latent CPI Pace   │             │  • Overtaking Difficulty│
│  • Teammate Deltas    │              │  • Sector Speeds     │             │  • Tyre Stress Rating   │
│  • Consistency CV     │              │  • Tyre Degradation  │             │  • Altitude & Weather   │
└───────────┬───────────┘              └──────────┬───────────┘             └────────────┬────────────┘
            │                                     │                                      │
            └─────────────────────────────┬───────┴──────────────────────────────────────┘
                                          │
                                          ▼
                       ┌────────────────────────────────────────────────────────┐
                       │       Context-Aware Non-Linear Interaction Engine      │
                       │           (XGBoost / LambdaMART Rank Engine)           │
                       │                                                        │
                       │  • GridPos × OvertakingDifficulty                      │
                       │  • TyreDeg × TyreStressRating                          │
                       │  • TopSpeed × CircuitLength/Altitude                   │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                                                  ▼
                       ┌────────────────────────────────────────────────────────┐
                       │           Explainable Attribution (SHAP Engine)        │
                       │    (Quantifies WHY driver is predicted in position)   │
                       └────────────────────────────────────────────────────────┘
```

### 5.2 Key Explicit Feature Interactions
1. $\text{GridPosition} \times \text{OvertakingDifficultyRating}$: Scales the retention probability of the starting position.
2. $\text{TyreDegradationSlope} \times \text{TyreStressLevel}$: Magnifies car tyre wear penalties on abrasive, high-energy circuits (Barcelona, Silverstone, Suzuka).
3. $\text{TopSpeedTrap} \times \text{TrackLength} \times \text{Altitude}$: Rewards straight-line efficiency at high-speed and high-altitude circuits (Monza, Mexico City).
4. $\text{DriverRating} \times \text{PositionsGainedBaseline}$: Empirically predicts how effectively a driver can slice through the field when starting out of position.

### 5.3 Attribution vs. Causality Principle
PACE strictly enforces that **feature importance metrics (such as SHAP values, Gain, or Permutation Importance) are explanatory attribution tools, NOT causal weights or arbitrary model dials**:
- SHAP values quantify the marginal contribution of each feature to a specific driver's predicted finishing position score.
- They explain *why* the model expects Hamilton to finish P2 or Lindblad P10, providing transparent reporting.

---

## 6. Modelling Approaches & Recommended Architecture

### 6.1 Evaluation of Candidate Paradigms

| Paradigm | Strengths | Limitations | PACE Suitability |
| :--- | :--- | :--- | :---: |
| **Linear / Regularised Regression (Ridge / Lasso)** | Highly interpretable, closed-form coefficients. | Incapable of modeling permutation constraints; fails on non-linear interactions; cannot guarantee unique P1–P20 ranks. | Baseline only |
| **Ordinal Logistic Regression** | Respects ordered nature of positions (P1 > P2 > P3). | Struggles with high-dimensional feature spaces and complex multi-way interactions. | Secondary Baseline |
| **Standard GBDTs (XGBoost $L_2$ Regressor)** | Excellent tabular accuracy, automatically discovers non-linear interactions. | Treats finishing positions as independent continuous variables; can predict duplicate positions; ignores race-level permutation structure. | Insufficient alone |
| **Bayesian Hierarchical Skill Models (TrueSkill / Glicko)** | Superb at isolating latent driver skill from paired teammate comparisons; provides uncertainty bounds. | Does not natively ingest real-time session telemetry, sector speeds, or weather interactions. | Core Feature Engine |
| **Listwise Learning-to-Rank (LambdaMART / XGBoost `rank:pairwise` / `rank:ndcg`)** | Treats each Grand Prix as a structured group; directly optimizes ranking metrics (NDCG, Kendall's Tau); inherently models competition. | Outputs latent ranking scores rather than direct calibrated probabilities. | **Recommended Core** |

### 6.2 Recommended Initial Architecture: Two-Stage Hybrid Ranker

PACE adopts a **Two-Stage Hybrid Predictive Architecture**:

```
Stage 1: Latent Feature Extractors
  ├── Glicko-2 Teammate Rating Engine ──> Latent Driver Skill (DPI) & Uncertainty (RD)
  ├── Constructor Relative Gap Engine ──> Car Base Pace (CPI) & Dev Trajectory
  ├── FP2 Telemetry Degradation Modeler ─> Tyre Wear Index & Sector Speeds
  └── Empirical Bayes Track Modeler ─────> Shrinkage Circuit Affinity (DCI)
                             │
                             ▼
Stage 2: Supervised Listwise Ranking Engine
  ├── Grouped by Event (event_id = season_round)
  ├── XGBoost LambdaMART / Pairwise Objective (rank:pairwise / rank:ndcg)
  ├── Inputs: DPI, CPI, DCI, Grid, Practice Telemetry, Circuit Characteristics, Interactions
  └── Outputs: Continuous Latent Performance Score s_i for each driver i
                             │
                             ▼
Stage 3: Permutation Solver & Calibrated Probability Decomposition
  ├── Sorting: Order drivers by latent score s_i descending ──> Deterministic P1–P20
  ├── Plackett-Luce Choice Probability: P(Driver i wins) = exp(s_i) / sum(exp(s_k))
  └── Binary Mechanical DNF Calibrator (Logistic/Isotonic) ──> DNF Probability
```

#### Why This Architecture is Justified:
1. **Mathematical Consistency:** Grouped listwise ranking guarantees that driver outcomes are modeled relative to the competitors in *that specific race*, not against an abstract absolute scale.
2. **Deterministic Output Guarantee:** Sorting latent scores $s_i$ mathematically guarantees an internally consistent P1 to P20 finishing classification without ties or duplicate positions.
3. **Probabilistic Realism:** The Plackett-Luce choice framework converts latent scores into full probability distributions across finishing positions, satisfying PACE's requirement to deliver probabilistic race forecasts.
4. **Engineering Explainability:** Tree SHAP decomposition on the Stage 2 ranker enables exact attribution for every driver's predicted position.

---

## 7. Chronological Backtesting & Validation Methodology

### 7.1 Prevention of Data Leakage
Predicting Formula 1 races requires strict adherence to temporal causality:
- **Zero Future Leakage:** For any race at $(Season, Round)$, models may ONLY train on data from events that occurred strictly prior to $(Season, Round)$.
- **Zero Label Leakage:** Race results, pit stops, and final classifications from the event being predicted are strictly isolated as target labels.
- **Session Timestamp Cutoff:** For pre-race predictions, only Friday practice (FP1, FP2, FP3) and Saturday Qualifying data from the current event are accessible.

### 7.2 Walk-Forward Expanding Window Backtesting Design

```
Fold 1: Train [2023 R1 .. 2023 R10] ───────────────> Test on [2023 R11]
Fold 2: Train [2023 R1 .. 2023 R11] ───────────────> Test on [2023 R12]
...
Fold k: Train [2023 R1 .. 2025 R24] ───────────────> Test on [2026 R1]
...
Fold N: Train [2023 R1 .. 2026 R19] ───────────────> Test on [2026 R20]
```

### 7.3 Benchmark Baselines
Every predictive model in PACE must be evaluated against three standard baselines:
1. **Qualifying Grid Baseline:** Predicted finish = Starting grid position ($Finish_i = Grid_i$).
2. **Current Championship Standing Baseline:** Predicted finish = Order in the World Drivers' Championship standings prior to the race.
3. **Rolling Constructor Baseline:** Predicted finish = Constructor's rolling average finish position over the prior 3 events.

### 7.4 Quantitative Evaluation Metrics

#### A. Rank Correlation & Ordinal Metrics
- **Spearman Rank Correlation ($\rho$):**
  $$\rho = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
  Measures monotonic ranking alignment across the full field ($-1.0$ to $+1.0$).
- **Kendall's Tau ($\tau$):** Measures the proportion of concordant vs discordant pairs, penalizing pairwise inversions.

#### B. Position Distance Metrics
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \left| \widehat{Pos}_i - Pos_i \right|$$
  Measures the average number of positions a prediction is off by.

#### C. Threshold Classification Accuracy
- **Top-1 Accuracy:** Precision on race winner prediction.
- **Top-3 Podium Accuracy:** Precision, recall, and F1-score on the three podium finishers.
- **Top-10 Points Accuracy:** Percentage of drivers predicted in the top 10 who actually score championship points.

#### D. Probabilistic Calibration
When probabilistic outputs are generated:
- **Brier Score:** Mean squared difference between predicted win/podium probabilities and binary outcomes.
- **Expected Calibration Error (ECE):** Assesses whether a driver assigned an 80% podium probability achieves a podium 80% of the time across backtested bins.

### 7.5 Sliced Evaluations & Ablation Protocol
To verify that the model is robust and not overfitting to specific teams or tracks, backtested results must be sliced across:
- **By Circuit Overtaking Class:** Performance on `High` (Monaco) vs `Low` (Monza, Spa) overtaking difficulty.
- **By Regulation Era:** Verification that performance holds across the Ground Effect era (2022–2026).
- **By Grid Tier:** Accuracy on Front-runners (P1–P6), Midfield (P7–P14), and Backmarkers (P15–P22).

#### Ablation Suite:
PACE will evaluate five progressive model tiers to verify each feature family's value:
1. **Model A:** Qualifying Grid Order alone.
2. **Model B:** Model A + Constructor Performance Index (CPI).
3. **Model C:** Model B + Driver Performance Index (DPI).
4. **Model D:** Model C + Driver–Circuit Historical Intelligence (DCI).
5. **Model E:** Full Hybrid Model with non-linear Circuit Topology interactions.

---

## 8. Proposed Stage 2 Implementation Plan

Upon approval of this methodology, Phase 3 Stage 2 implementation will proceed as follows:

1. **Step 1: Feature Engineering Infrastructure (`src/features/`)**
   - Implement `src/features/dpi.py`: Teammate relative qualifying/race deltas, Glicko-2 rating engine, consistency metrics.
   - Implement `src/features/cpi.py`: Constructor pace gaps, FP2 degradation slope extractor, reliability indices.
   - Implement `src/features/driver_circuit.py`: Empirical Bayes shrinkage calculator, circuit affinity vectors.
   - Implement `src/features/pipeline.py`: Leakage-free feature builder joining all features as of session cutoff.

2. **Step 2: Baseline Models (`src/models/baselines.py`)**
   - Implement Grid Order Baseline, Championship Standing Baseline, and Rolling Constructor Baseline.

3. **Step 3: Machine Learning Ranking Engine (`src/models/ranker.py`)**
   - Implement XGBoost LambdaMART / Pairwise ranking model.
   - Implement Plackett-Luce choice probability decomposition.
   - Implement deterministic P1–P20 permutation constraint solver.

4. **Step 4: Chronological Backtesting & Evaluation (`src/evaluation/`)**
   - Implement `src/evaluation/backtest.py`: Walk-forward expanding window cross-validator.
   - Implement `src/evaluation/metrics.py`: MAE, Spearman $\rho$, Kendall $\tau$, Top-1/3/10 accuracy, Brier score.
   - Implement `src/evaluation/explainability.py`: Tree SHAP driver attribution explainer.

5. **Step 5: Testing & Verification (`tests/`)**
   - Unit tests for feature calculators, shrinkage mathematics, ranker permutation constraints, and backtester leakage guards.
   - Backtest execution across historical seasons with automated performance benchmark reporting.

---

## 9. Conclusion & Sign-Off Request

This methodology establishes a mathematically sound, domain-informed, and leak-free foundation for PACE's performance modeling engine. It avoids arbitrary weighting guesswork, rigorously isolates driver capability from car pace through networked teammate comparisons, stabilizes small-sample circuit history with Bayesian shrinkage, and directly optimizes the competitive finishing order through listwise ranking.

We submit this Stage 1 methodology for formal approval before proceeding to model implementation in Stage 2.
