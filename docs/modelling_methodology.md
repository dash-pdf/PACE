# PACE Phase 3: Driver & Constructor Performance Modelling Methodology

**Predictive Analytics for Competitive Evaluation**  
**Document Version:** 2.0.0 (Stage 1 Architecture — Revised with Rigorous Methodological Refinements)  
**Author:** PACE Engineering Team  
**Status:** Under Final Review (Methodology Sign-Off Prior to Stage 2)

---

## 1. Executive Summary & Core Architectural Principles

Formula 1 race finishing positions are the emergent result of complex interactions between human skill, machine competitiveness, circuit characteristics, environmental conditions, and strategic execution. 

PACE Phase 3 establishes the mathematical and statistical foundations of the PACE prediction engine. Following extensive engineering review, this methodology explicitly incorporates the following foundational principles:
- **Separation of Directly Measured vs. Inferred Quantities:** Telemetry measurements (speed traps, micro-sector times, throttle application points) are strictly distinguished from inferred vehicle properties (aerodynamic downforce, mechanical grip). All inferred parameters carry explicit uncertainty bounds.
- **Confounder-Aware Teammate Comparison:** Teammate comparisons provide the primary intra-team control, but are treated with statistical caution to account for asymmetric mechanical reliability, strategy divergence, incidents, team orders, and aerodynamic upgrade disparities.
- **Decoupled Driver Experience vs. Latent Ability:** Experience is modeled as an uncertainty-reduction and learning-curve covariate; it does NOT automatically inflate estimated latent skill.
- **Empirical Validation of Temporal Weighting:** Technical regulation era weights are treated as testable scientific hypotheses to be benchmarked against continuous exponential half-life decay and learned hierarchical variances.
- **Unvalidated Assumptions Avoidance:** Fuel burn-off effects are modeled as latent variables with circuit sensitivity intervals rather than hardcoded point constants.
- **Complete and Internally Consistent Race Outcome Space:** Explicitly partitions the outcome space into classified finishes (P1 to P$N$), mechanical retirements, incident-induced DNFs, non-classifications, and disqualifications (DSQ) such that total probability sums strictly to 1.0.
- **End-to-End Uncertainty Propagation:** Latent driver skill, car pace, and circuit affinity carry probability distributions $(\mu, \sigma)$ into downstream ranking models.
- **Multi-Model Benchmark Protocol:** Interpretable statistical baselines (Linear mixed models, Bradley-Terry, Ordinal Logistic) are evaluated alongside tree-based learning-to-rank algorithms on identical chronological walk-forward splits using paired significance testing.
- **Explanatory Attribution vs. Causal Weights:** SHAP values and permutation importances are used exclusively for post-hoc prediction explanation, never converted into model weights or causal dials.

---

## 2. Driver Performance Index (DPI)

### 2.1 The Fundamental Challenge & Teammate Confounders
In Formula 1, a driver's raw finishing position is heavily confounded by constructor machinery. Teammates provide the closest approximation of a controlled experiment, as both drivers operate the same baseline car. However, teammate comparisons are subject to severe real-world confounders that must be statistically controlled:

1. **Asymmetric Mechanical Reliability:** One car may experience a sub-lethal failure (e.g. sensor drift, MGU-K de-rating, energy recovery clipping, aerodynamic floor damage from kerbs) that slows lap times without causing an outright retirement.
2. **On-Track Incidents & First-Lap Contact:** Incidents triggered by third-party cars distort finishing positions without reflecting true driver pace.
3. **Strategic Divergence:** Teams frequently split strategies (e.g. one-stop vs two-stop, starting on Hard vs Soft tyres, pitting under Safety Car vs staying out in dirty air). The driver on the compromised strategy suffers artificial pace degradation.
4. **Team Orders & Pit Wall Priority:** Contractual driver hierarchies, priority pit stop timing (undercut advantage), and direct radio instructions ("do not hold up your teammate") systematically bias intra-team deltas.
5. **Machinery & Upgrade Disparities:** Constructors regularly introduce aerodynamic upgrades (new floor edge, revised front wing, lighter chassis) on only *one* car due to manufacturing lead times.

### 2.2 Mitigation Strategy for Teammate Comparisons
PACE addresses these confounders through rigorous filtering and robust statistical weighting:
- **Incident & Damage Filtering:** Appearances where a driver sustained verified collision damage or floor perforation are flagged and excluded from baseline pace delta estimation.
- **Clean Air Lap Extraction:** Rather than using raw race duration, teammate race pace deltas are calculated exclusively from matched stint laps where both drivers were running in "clean air" (gap to car ahead $> 1.5\text{ seconds}$) on the same tyre compound of comparable age ($\Delta\text{tyre age} \le 3\text{ laps}$).
- **Strategy Normalization:** Finishing deltas are adjusted for compound baseline speed offsets derived from practice session tyre deltas.
- **Team Orders Masking:** Known historical instances of contractual position swaps (e.g. team order events documented in race control messages) are treated as ties or masked for head-to-head outcome modeling.

### 2.3 Candidate DPI Mathematical Formulations

To identify the optimal driver rating mechanism, PACE evaluates three competing statistical paradigms on identical backtested data:

#### Model 1: Glicko-2 Networked Paired Comparison Engine
Extends the classic Elo rating system to model driver skill as a probability distribution $N(\mu_i, \phi_i^2)$ with dynamic volatility $\sigma_i$:
- $\mu_i$: Latent driver skill estimate.
- $\phi_i$: Rating deviation (epistemic uncertainty). Shrinks with active, consistent data; widens across off-seasons or sabbaticals.
- $\sigma_i$: Rating volatility (aleatoric performance variability).
The probability that driver $i$ outperforms teammate $j$ in clean session execution is:
$$\mathbb{E}(s \mid \mu_i, \mu_j, \phi_j) = \frac{1}{1 + 10^{-g(\phi_j)(\mu_i - \mu_j) / 400}}$$
where $g(\phi) = \left(1 + \frac{3q^2\phi^2}{\pi^2}\right)^{-1/2}$ and $q = \frac{\ln(10)}{400}$.

#### Model 2: Hierarchical Bayesian Linear Mixed-Effects Model (LMM)
Models lap time delta relative to field median as a linear mixed model with crossed random effects:
$$y_{i,c,t} = \beta_0 + \beta_{grid} Grid_{i,t} + \mathbf{x}_{i,t}^T \boldsymbol{\gamma} + u_{driver(i)} + v_{constructor(c)} + w_{event(t)} + \epsilon_{i,c,t}$$
- $u_{driver(i)} \sim N(0, \sigma_{driver}^2)$: Pure driver effect, isolated from constructor.
- $v_{constructor(c)} \sim N(0, \sigma_{constructor}^2)$: Pure car effect.
- $w_{event(t)} \sim N(0, \sigma_{event}^2)$: Race-specific intercept.
- $\epsilon_{i,c,t} \sim N(0, \sigma_e^2)$: Observation noise.
This provides a closed-form, simultaneous separation of driver vs car capability with full Bayesian posterior standard errors.

#### Model 3: Bradley-Terry Model with Team-Level Shared Intercepts
Formulates paired comparisons conditioned on constructor covariates:
$$\text{logit}(P(i \succ j \mid c)) = (\alpha_i - \alpha_j) + \mathbf{z}_{ij}^T \boldsymbol{\delta}$$
where $\mathbf{z}_{ij}$ captures observed machinery deltas (e.g., upgrade version indicator, tyre age difference).

### 2.4 Decoupling Experience from Driver Ability
A critical requirement of PACE is that **experience must not automatically inflate a driver's estimated capability**:
- A veteran driver with 250 starts (e.g. Stroll, Hülkenberg) is not assumed to be faster than a superstar sophomore (e.g. Piastri) simply due to longevity.
- Longevity without elite performance indicates survival, not superiority.

In PACE, driver experience $N_{starts}$ is parameterized strictly as:
1. **An Uncertainty Reducer:** Lowering the prior rating deviation $\phi_i$ in Glicko-2, narrowing the posterior credible interval of the driver's skill parameter.
2. **A Separate Rookie Learning Rate Covariate:** Modeled as a decaying marginal benefit function:
   $$f_{learn}(N_{starts}) = 1.0 - \exp\left(-\frac{N_{starts}}{\tau_{learn}}\right)$$
   where $\tau_{learn} \approx 25\text{ starts}$ (~1 full season). Beyond 50 starts, $f_{learn} \approx 1.0$, and experience provides zero additive baseline score.
3. **An Interaction with Consistency:** High experience empirically reduces pace variance $CV_{pace}$, which is measured directly from lap times rather than imputed from starts.

---

## 3. Constructor Performance Index (CPI)

### 3.1 Directly Measured vs. Inferred Telemetry Features
To maintain scientific rigor, PACE establishes an unambiguous boundary between what is directly observed and what is mathematically inferred:

| Telemetry Quantity | Classification | Source | Nature of Measurement | Uncertainty / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Speed Trap Speeds ($ST_{speed}$)** | **Directly Measured** | OpenF1 Telemetry | Radar/laser timing line at end of longest straight | Affected by slipstream (tow), DRS status, and headwind. |
| **Sector Times ($S_1, S_2, S_3$)** | **Directly Measured** | FastF1 / OpenF1 | Exact transponder timing loop elapsed seconds | Conflates engine power, aero drag, tyre grip, setup, and driver error. |
| **Micro-Sector Speeds & Min Corner Apex Speed** | **Directly Measured** | OpenF1 / FastF1 | Wheel speed sensor and GPS velocity at apex | Affected by fuel load, track evolution, and driving line. |
| **Aerodynamic Downforce Level** | **Inferred Feature** | Physical Model | Ratio of high-speed corner minimum velocity to straight-line terminal velocity | **High Uncertainty.** Setup dependent; cannot distinguish pure aero downforce from tyre compound grip. |
| **Mechanical Grip / Low-Speed Traction** | **Inferred Feature** | Physical Model | Longitudinal acceleration ($a_x$) out of slow corners ($< 100\text{ km/h}$) | **Moderate Uncertainty.** Conflates differential tuning, tyre temperature, and throttle mapping. |
| **Tyre Degradation Slope ($k_{deg}$)** | **Inferred Feature** | Linear/Huber Regression | Stint lap times vs elapsed laps | **Moderate Uncertainty.** Conflates tyre wear with fuel burn-off, traffic, and engine mode management. |

> [!IMPORTANT]
> Sector times alone **never** isolate aerodynamic downforce or mechanical grip. A car with an exceptional power unit can set the fastest Sector 2 even with mediocre downforce. PACE treats downforce and mechanical grip as *latent inferred parameters* equipped with standard errors ($\sigma_{aero}, \sigma_{mech}$), never as direct sensor measurements.

### 3.2 Fuel Load Modeling & Degradation Without Hardcoded Constants
A fundamental limitation of public Formula 1 data is that **exact fuel mass is not directly observable**. FIA regulations mandate starting with $\le 110\text{ kg}$ of fuel in the race, but teams never publish their fuel load during Friday practice (FP1/FP2) or race stints.

#### Flaws of Hardcoded Fuel Corrections:
Assuming a static $-0.06\text{ s/lap}$ fuel burn-off correction introduces massive systemic error:
1. Fuel sensitivity ($S_{fuel}$) varies wildly by circuit layout:
   - At Monza (5.793 km, long full-throttle straights): $S_{fuel} \approx 0.035\text{ s per 10 kg}$.
   - At Marina Bay (Singapore, stop-start high braking mass): $S_{fuel} \approx 0.075\text{ s per 10 kg}$.
   - At Spa-Francorchamps (7.004 km, immense vertical elevation changes): $S_{fuel} \approx 0.080\text{ s per 10 kg}$.
2. In practice sessions, teams run completely unannounced run plans: a driver on a 5-lap run might have 15 kg of fuel (qualifying simulation), while their teammate on an 18-lap run carries 80 kg (race simulation).

#### PACE's Probabilistic Stint-Clustering & Latent Fuel Architecture:
Rather than hardcoding fuel mass, PACE models stint pace via **Latent Fuel State Clustering**:
1. **Stint Length Classification:** Runs are segmented by duration into:
   - *Short Runs ($\le 6\text{ laps}$):* Inferred low fuel ($15 \pm 10\text{ kg}$). Focuses on single-lap peak performance.
   - *Long Runs ($\ge 10\text{ laps}$):* Inferred high fuel ($60 \pm 20\text{ kg}$). Focuses on tyre degradation and race simulation.
2. **Circuit Fuel Sensitivity Factor ($S_{fuel, c}$):** Computed from physical circuit parameters rather than guessed:
   $$S_{fuel, c} = k_{mass} \cdot \left(\frac{L_{circuit}}{5.0}\right) \cdot (1 + \beta_{accel} \cdot N_{braking\_zones})$$
   where $k_{mass} \approx 0.05\text{ s/lap/10kg}$ is scaled by track length and braking-to-traction acceleration events.
3. **Robust Degradation Estimation ($k_{deg}$):** On identified long runs, lap times $T(l)$ are fitted using Huber M-estimation over laps $l \in [2, L-1]$ (excluding in/out laps):
   $$T(l) = T_0 + \left(k_{deg} - S_{fuel, c} \cdot \frac{m_{fuel,0}}{L_{race}}\right) \cdot l + \epsilon_l$$
   The resulting degradation slope $k_{deg}$ carries an empirical estimation standard error $\text{SE}(k_{deg})$ representing confidence in tyre life.

### 3.3 Constructor Pace Index Components
The final CPI feature vector incorporates:
1. **Peak Qualifying Delta ($\Delta T_{c,t}^{pole}$):** Percentage gap to pole across both team cars.
2. **Clean Air Race Pace Gap ($\Delta T_{c,t}^{race}$):** Median clean lap time gap to the fastest car on the same compound.
3. **Trap Speed Percentile ($P_{ST}$):** Terminal straight-line velocity rank.
4. **Degradation Coefficient and Uncertainty:** $(k_{deg}, \text{SE}(k_{deg}))$.
5. **Exponential Mechanical Reliability ($Rel_c$):** 10-race half-life failure rate.
6. **Rolling Development Trajectory ($\beta_{dev}$):** 5-race moving regression slope of relative pace.

---

## 4. Driver–Circuit Intelligence Integration

### 4.1 Empirical Bayes Shrinkage Formulation
To prevent small-sample distortion for drivers with limited circuit starts while leveraging extensive historical records for veterans, PACE implements **Empirical Bayes Hierarchical Shrinkage**.

For driver $i$ at circuit $c$, the circuit-specific performance index $\widehat{\mu}_{i,c}$ is formulated as a shrinkage estimator between their observed circuit sample mean $\bar{y}_{i,c}$ and their global career baseline $\mu_i^{global}$:
$$\widehat{\mu}_{i,c} = \alpha_{i,c} \cdot \bar{y}_{i,c} + (1 - \alpha_{i,c}) \cdot \mu_i^{global}$$
where the shrinkage weight $\alpha_{i,c} \in [0, 1]$ is governed by:
$$\alpha_{i,c} = \frac{N_{i,c}}{N_{i,c} + \kappa_0}$$
- $N_{i,c}$: Total recorded appearances of driver $i$ at circuit $c$.
- $\bar{y}_{i,c}$: Sample average finishing position (or position delta) of driver $i$ at circuit $c$.
- $\mu_i^{global}$: Overall career average finishing performance of driver $i$.
- $\kappa_0$: Empirical Bayes prior precision parameter.

#### Uncertainty of the Circuit Estimate:
The posterior variance of the circuit affinity estimate is:
$$\text{Var}(\widehat{\mu}_{i,c}) = \frac{\sigma_{within}^2}{N_{i,c} + \kappa_0}$$
This variance is explicitly passed downstream to the ranking engine, ensuring that a rookie's uncertain circuit affinity carries low confidence, whereas a veteran's affinity carries high statistical confidence.

### 4.2 Empirical Evaluation of Temporal Relevance (Hypothesis Testing)
Formula 1 technical regulations alter chassis dynamics, tyre behavior, and overtaking characteristics. However, treating discrete regulation eras as rigid, hardcoded constants ($1.0, 0.6, 0.25$) is a subjective heuristic.

PACE establishes a formal **chronological backtesting experiment** to test four competing temporal weighting hypotheses:

| Hypothesis | Mathematical Formulation | Underlying Assumption |
| :--- | :--- | :--- |
| **H1: Continuous Exponential Half-Life** | $w(t - \tau) = \exp\left(-\lambda_{half} \cdot (t - \tau)\right)$ | Performance relevance decays smoothly over time regardless of rule boundaries ($\lambda_{half} \approx 3\text{ years}$). |
| **H2: Discrete Era Step-Down** | $w(era) \in \{1.00, 0.60, 0.25, 0.10\}$ | Major regulation shifts (e.g. 2022 Ground Effect) reset vehicle physics; historical records across era boundaries lose substantial relevance. |
| **H3: Hierarchical Random Walk** | $\mu_{i,c}(t) = \mu_{i,c}(t-1) + \eta_t, \quad \eta_t \sim N(0, \sigma_{walk}^2)$ | Circuit affinity evolves dynamically as a latent state space process. |
| **H4: Era-Invariant Uniform** | $w(t - \tau) = 1.0 \quad \forall \tau$ | Circuit layout geometry and driver track compatibility are intrinsic and invariant across car eras. |

During Phase 3 model validation, all four hypotheses will be evaluated on historical seasons (2023–2026), and the optimal weighting function will be chosen strictly by out-of-sample log-likelihood and rank correlation, not developer intuition.

---

## 5. Complete Race Outcome Space: DNF & Classification Modelling

### 5.1 Mutually Exclusive & Exhaustive Outcome Space
A critical requirement of PACE is that **the probability calculations must represent the complete race outcome space consistently**:
- A driver cannot be predicted to finish P3 and simultaneously have a 30% probability of retiring without those outcomes being mathematically linked.
- The outcome space for every driver $i$ in a field of $N$ starters is strictly partitioned into mutually exclusive states:

$$\Omega_i = \{ \text{Classified at Position } k \mid k \in [1, N] \} \cup \{ \text{Mechanical DNF} \} \cup \{ \text{Incident DNF} \} \cup \{ \text{DNS} \} \cup \{ \text{DSQ} \}$$
$$\sum_{k=1}^N P(\text{Classified } k) + P(\text{Mechanical DNF}) + P(\text{Incident DNF}) + P(\text{DNS}) + P(\text{DSQ}) = 1.0$$

### 5.2 Two-Stage Joint Outcome Decomposition
To enforce this consistency, PACE decomposes the probability distribution hierarchically:

```
                                  [ Driver i Starts Race ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ Classified Finisher ]                        [ Non-Finisher / DNF ]
          P(Classified) = 1 - P(DNF)                               P(DNF)
                       │                                           │
                       ▼                                           ▼
             [ P1 to P20 Ranking ]                       [ Cause Decomposition ]
          P(Finish = k | Classified)                  ├── Mechanical Failure: P(Mech)
       via Plackett-Luce Choice Model                 ├── Collision/Driver Error: P(Inc)
                                                      └── Disqualification: P(DSQ)
```

#### Stage A: Calibrated Retirement Likelihood ($P(DNF_i)$)
The probability of driver $i$ failing to finish is modeled using a regularized logistic regression with isotonic probability calibration:
$$P(DNF_{i,t}) = \sigma\left(\theta_0 + \theta_1 (1 - Rel_{c(i)}) + \theta_2 P_{circuit}^{SC} + \theta_3 Grid_{i,t} + \theta_4 CV_{pace}(i)\right)$$
- Constructor mechanical unreliability ($1 - Rel_c$) drives mechanical failures.
- Circuit Safety Car history ($P_{circuit}^{SC}$) and starting in congested midfield zones (Grid P8–P16) drive collision incident risks.

#### Stage B: Conditional Finishing Rank ($P(\text{Rank } k \mid \text{Classified})$)
Conditioned on being classified, driver relative performance is modeled through a continuous latent score $s_i$ in a **Plackett-Luce choice framework**:
$$P(\text{Driver } i \text{ finishes ahead of } j \mid \text{both classified}) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)}$$
The probability that driver $i$ claims finishing position $k$ given the surviving field is computed via forward-backward dynamic programming over the Plackett-Luce permutation space.

#### Stage C: Expected Finishing Position
The complete expected finishing position combines both branches:
$$\mathbb{E}[Pos_i] = \sum_{k=1}^{N_{classified}} k \cdot P(\text{Finish } k \mid \text{Classified}) \cdot (1 - P(DNF_i)) + N_{classified} \cdot P(DNF_i)$$
This guarantees mathematical consistency: high DNF risk automatically penalizes expected championship points and finishing position expectations.

---

## 6. End-to-End Uncertainty Propagation

To prevent overconfidence, point estimates must not be passed into downstream models in isolation. Every upstream feature module outputs both a central tendency and a dispersion metric:

1. **Driver Skill:** Enters as $(DPI_{\mu}, DPI_{\sigma})$, where $DPI_{\sigma} = \phi_i$ from Glicko-2 or posterior credible standard error from Hierarchical Bayes.
2. **Constructor Pace:** Enters as $(CPI_{\mu}, CPI_{\sigma})$, where $CPI_{\sigma}$ reflects pace variance across practice sessions and stint laps.
3. **Circuit Affinity:** Enters as $(\widehat{\mu}_{i,c}, \text{SE}(\widehat{\mu}_{i,c}))$, carrying Bayesian shrinkage uncertainty.
4. **Tyre Degradation:** Enters as $(k_{deg}, \text{SE}(k_{deg}))$.

### Propagation Mechanisms
- **Feature Level:** Upstream variances enter the ranking model as explicit input columns (e.g. `dpi_uncertainty`, `cpi_pace_std`, `circuit_sample_variance`). This enables tree models to learn conservative split rules when inputs have high variance (e.g. discounting circuit affinity for debutants).
- **Inference Level (Monte Carlo Permutation Draws):** For probabilistic forecasting, PACE samples $M = 1,000$ draws from the joint posterior:
  $$s_i^{(m)} \sim N\left(\widehat{s}_i(\mathbf{x}), \Sigma_i(\mathbf{x})\right)$$
  For each draw $m$, sorting $\{s_1^{(m)}, \dots, s_N^{(m)}\}$ yields a complete, valid race permutation. Aggregating across the $M$ draws generates the full finishing position probability distribution and confidence intervals for every driver.

---

## 7. Context-Dependent Feature Interactions & Attribution Principles

### 7.1 Learned Interactions Guided by Physics Constraints
PACE rejects rigid, hardcoded weighting percentages (e.g. "40% car, 30% qualy, 20% driver, 10% circuit"). Instead, gradient boosting algorithms learn non-linear interaction surfaces guided by known physical constraints:

1. $\text{QualifyingGridPosition} \times \text{OvertakingDifficultyRating}$:
   - At high-difficulty tracks (Monaco, Hungaroring, Marina Bay), the model automatically learns that starting position dominates finishing outcome ($> 80\%$ correlation).
   - At low-difficulty tracks (Spa, Baku, Monza), the model learns that starting position weight diminishes, shifting importance to straight-line speed and race pace.
2. $\text{TyreDegradationSlope} \times \text{TyreStressRating}$:
   - A constructor with high tyre wear is penalised severely at Barcelona and Silverstone (high-energy lateral loading), but penalised minimally at Monaco (smooth, traction-limited tarmac).
3. $\text{TopSpeedTrap} \times \text{TrackLength} \times \text{Altitude}$:
   - Straight-line speed and turbocharger compressor efficiency dominate at Monza and Mexico City, but have negligible importance at Monaco or Zandvoort.

### 7.2 Rigorous Distinction: SHAP Attribution vs. Causal Weights
PACE strictly establishes the mathematical interpretation of SHAP (SHapley Additive exPlanations):

$$\widehat{f}(\mathbf{x}) = \phi_0 + \sum_{j=1}^M \phi_j(\mathbf{x})$$

- **What SHAP is:** A local, additive game-theoretic decomposition quantifying how much feature $j$ shifted driver $i$'s predicted latent score away from the dataset base expectation for *that specific race*.
- **What SHAP is NOT:** SHAP values are **not** causal regression coefficients, nor are they global percentage dials. A high SHAP value for grid position at Monaco does not mean the user can manually dial "grid weight" up or down.
- **Explainability Role:** SHAP values are used strictly to generate human-readable post-race explanations (e.g. *"Hamilton was predicted P2 because high sector 2 pace (+1.2) and grid position (+2.1) offset tyre degradation (-0.4)"*).

---

## 8. Model Comparison Plan & Candidate Architectures

To ensure the selection of XGBoost LambdaMART is scientifically justified, PACE will benchmark it against a progressive suite of interpretable statistical baselines on **strictly identical chronological splits**:

```
[ Tier 1: Non-Parametric Baselines ]
  ├── B1: Qualifying Grid Order (Finish = Grid)
  ├── B2: Current Championship Standings Order
  └── B3: Rolling Constructor 3-Race Average Finish

[ Tier 2: Interpretable Statistical Models ]
  ├── M1: Regularized Linear Regression (Ridge / Lasso) with position bounds
  ├── M2: Ordinal Logistic Regression (Proportional Odds Model)
  └── M3: Bradley-Terry Paired Choice Model with Covariates

[ Tier 3: Hierarchical Bayesian Latent Skill Models ]
  └── M4: Bayesian Linear Mixed-Effects Model (LMM Driver + Team crossed effects)

[ Tier 4: Supervised Learning-to-Rank GBDTs ]
  ├── M5: XGBoost Pairwise Ranking (rank:pairwise)
  └── M6: XGBoost Listwise LambdaMART (rank:ndcg)
```

### Statistical Significance Protocol
- All models are evaluated on the expanding window walk-forward dataset (2023–2026).
- Pairwise model comparisons are tested using **paired Wilcoxon signed-rank tests** and **Diebold-Mariano tests** on per-race MAE and Spearman $\rho$.
- A complex model (e.g. M6) will only be adopted over an interpretable baseline (e.g. M1 or M3) if it achieves a **statistically significant improvement ($p < 0.01$)** with meaningful effect size ($> 5\%$ reduction in MAE).

---

## 9. Feature Availability Protocol & Leakage Prevention

### 9.1 Temporal Prediction Horizons
PACE operates under strict, verifiable prediction timestamps:

| Horizon | Prediction Timestamp | Available Features | Strictly Prohibited Data |
| :--- | :--- | :--- | :--- |
| **H1: Pre-Weekend** | Thursday 23:59 UTC prior to event | Historical DPI, CPI, Circuit Affinity up to $(t-1)$, circuit characteristics, regulation era. | Any practice telemetry, qualifying results, race results. |
| **H2: Post-Friday Practice** | Friday 23:59 UTC (after FP2) | All H1 features + FP1/FP2 sector split times, trap speeds, practice stint degradation slopes. | Saturday FP3, Qualifying results, race results. |
| **H3: Post-Qualifying (Primary)**| Saturday 23:59 UTC (after Qualy) | All H2 features + FP3 times, official Q1/Q2/Q3 qualifying times, final confirmed starting grid. | Sunday race results, race lap times, race pit stops, race weather. |

### 9.2 Missing Data & Sensor Dropout Protocols
- **Sprint Weekends:** Feature builders detect sprint schedules (only 1 practice session before competitive sessions). Degradation models automatically adjust sample sizes and widen uncertainty $\text{SE}(k_{deg})$.
- **Wet Practice Sessions:** If rain invalidates dry telemetry, practice stint pace is masked, and the model automatically falls back to prior dry CPI baselines with widened credible intervals.
- **Grid Penalties:** Official starting grid positions are ingested post-penalties (engine changes, gearbox penalties, impeding infractions) to reflect the true physical grid.

---

## 10. Unresolved Limitations & Engineering Risks

PACE explicitly acknowledges physical and operational realities that cannot be eliminated by statistical modeling:
1. **Unobservable Engine Modes & Engine Saving:** Teams operate secretive engine power modes (e.g. "Strat mode", "Party mode", "Engine 1"). A car cruising in FP2 may unleash 25 additional horsepower in Qualifying and Race trim without prior telemetry warning.
2. **Fuel Load Secrecy:** Friday practice run plans are proprietary. While PACE's latent stint-clustering isolates fuel trends, exact onboard mass remains unobservable.
3. **First-Lap Chaos & Collision Stochasticity:** No statistical model can predict whether an over-ambitious divebomb at Turn 1 will eliminate a front-row starter. Incident risks are fundamentally probabilistic, not deterministic.
4. **Dynamic Weather Transitions:** Sudden mid-race rain showers transform race pace into tyre crossover strategy gambles. Pre-race predictions cannot foresee the exact minute a rain cell hits.

---

## 11. Conclusion & Sign-Off Request

This revised methodology incorporates all requested refinements:
- Fully accounts for teammate comparison confounders.
- Rigorously distinguishes directly measured telemetry from inferred vehicle properties.
- Eliminates hardcoded fuel burn-off assumptions in favor of circuit-specific latent estimation.
- Formulates regulation-era relevance as testable scientific hypotheses.
- Decouples driver experience from underlying talent.
- Unifies classified positions, mechanical retirements, and incident DNFs into a complete, sum-to-one outcome space.
- Propagates posterior uncertainty end-to-end.
- Establishes a rigorous multi-model comparison protocol before selecting machine learning algorithms.

We present this revised specification for final sign-off before commencing Phase 3 Stage 2 implementation.
