# PACE Phase 3: Driver & Constructor Performance Modelling Methodology

**Predictive Analytics for Competitive Evaluation**  
**Document Version:** 2.1.0 (Stage 1 Architecture — Final Pre-Implementation Specification)  
**Author:** PACE Engineering Team  
**Status:** Under Final Review (Awaiting Formal Stage 2 Sign-Off)

---

## 1. Executive Summary & Core Architectural Principles

Formula 1 race finishing positions are the emergent result of complex interactions between human skill, machine competitiveness, circuit topology, environmental conditions, and strategic execution. 

PACE Phase 3 establishes the mathematical and statistical foundations of the PACE prediction engine. Following rigorous technical review, this specification incorporates five fundamental architectural principles:
1. **Evidence-Based Rating Uncertainty:** Rating deviations ($\phi$) in paired-comparison engines shrink *exclusively* through informative, unconfounded competitive observations, never through crude career start tallies. Career experience is treated as an independent learning-curve covariate.
2. **Latent Inferred Fuel States:** Fuel loads are strictly treated as unobserved latent variables with quantified uncertainty intervals, subject to empirical ablation testing against baseline models.
3. **Formal Comparative Uncertainty Propagation:** Rejects the assumption that passing standard deviations into tree models propagates uncertainty. Formulates Bayesian posterior sampling, chronological bootstrapping, and Monte Carlo permutation draws to quantify forecast dispersion.
4. **Acknowledged Multi-Car Outcome Dependencies:** Fully documents the limitations of marginal DNF models (collision cascades, Safety Car field bunching, engine supplier failures) and establishes an explicit bridge to Phase 4 stochastic simulation.
5. **Systematic Empirical Feature Selection (Ablation):** Every feature group is subjected to out-of-sample ablation testing. Complexity is never retained for theoretical elegance alone; uninformative features are pruned.

---

## 2. Driver Performance Index (DPI)

### 2.1 Fundamental Challenge & Teammate Confounders
In Formula 1, a driver's raw finishing position is heavily confounded by constructor machinery. Teammates provide the closest approximation of a controlled experiment, as both drivers operate the same baseline car. However, teammate comparisons are subject to severe real-world confounders:
1. **Asymmetric Mechanical Reliability:** Sub-lethal electrical clipping, sensor drift, or kerb damage slowing one car without causing immediate retirement.
2. **On-Track Incidents & External Contact:** Multi-car first-lap collisions distorting finishing position independent of driver pace.
3. **Strategic Divergence:** Compound offset or dirty-air traffic traps created by alternative pit stop timing.
4. **Team Orders & Pit Wall Priority:** Contractual hierarchies and undercut prioritization.
5. **Machinery & Upgrade Disparities:** Unequal aero specifications due to manufacturing lead times.

### 2.2 Mitigation Strategy for Teammate Comparisons
PACE mitigates these confounders through rigorous filtering:
- **Incident & Damage Filtering:** Appearances with verified collision damage are excluded from clean pace delta estimation.
- **Clean Air Lap Extraction:** Teammate race deltas are computed exclusively on matched stint laps where both drivers ran in clean air (gap $> 1.5\text{s}$) on identical compound age ($\Delta\text{age} \le 3\text{ laps}$).
- **Strategy Normalization:** Lap times are adjusted for compound speed offsets derived from practice session benchmarks.
- **Team Orders Masking:** Documented contractual position swaps are masked from head-to-head outcome records.

### 2.3 Candidate DPI Mathematical Formulations
PACE evaluates three competing statistical paradigms on identical backtested data:

#### Model 1: Glicko-2 Networked Paired Comparison Engine
Models latent driver skill as $N(\mu_i, \phi_i^2)$ with volatility $\sigma_i$:
$$\mathbb{E}(s \mid \mu_i, \mu_j, \phi_j) = \frac{1}{1 + 10^{-g(\phi_j)(\mu_i - \mu_j) / 400}}$$
where $g(\phi) = \left(1 + \frac{3q^2\phi^2}{\pi^2}\right)^{-1/2}$ and $q = \frac{\ln(10)}{400}$.

#### Model 2: Hierarchical Bayesian Linear Mixed-Effects Model (LMM)
Models lap time delta relative to field median with crossed random effects:
$$y_{i,c,t} = \beta_0 + \beta_{grid} Grid_{i,t} + \mathbf{x}_{i,t}^T \boldsymbol{\gamma} + u_{driver(i)} + v_{constructor(c)} + w_{event(t)} + \epsilon_{i,c,t}$$
where $u_i \sim N(0, \sigma_u^2)$ isolates driver capability with closed-form posterior standard errors.

#### Model 3: Bradley-Terry Model with Team-Level Shared Intercepts
$$\text{logit}(P(i \succ j \mid c)) = (\alpha_i - \alpha_j) + \mathbf{z}_{ij}^T \boldsymbol{\delta}$$

### 2.4 Decoupling Experience from Rating Uncertainty
PACE strictly enforces that **career starts ($N_{starts}$) must not automatically decrement rating deviation ($\phi_i$)**:
- **Why automatic reduction is invalid:** Longevity does not guarantee certainty. A driver who spends 100 races in uncompetitive cars with uninformative teammates or high mechanical attrition provides far less information than a driver competing in 20 clean, highly competitive head-to-head battles against elite benchmarks.
- **Information-Theoretic Uncertainty Reduction:** Rating deviation $\phi_i$ shrinks *exclusively* through the formal Bayesian Glicko-2 update equations:
  $$\phi_{i, t+1}^2 = \left( \frac{1}{\phi_{i, t}^2 + \sigma_i^2} + \sum_{j \in \text{matches}} g(\phi_j)^2 \cdot \mathbb{E}(s_{ij}) \cdot (1 - \mathbb{E}(s_{ij})) \right)^{-1}$$
  Uncertainty shrinks only when facing reliable, low-$\phi$ opponents in unconfounded sessions. Confounded sessions are masked, contributing zero information.
- **Off-Season Epistemic Expansion:** Between seasons or during sabbaticals, volatility $\sigma_i$ naturally increases $\phi_i$ via $\phi^* = \sqrt{\phi^2 + \sigma^2}$.
- **Experience as an Independent Covariate:** Career experience is modeled strictly as an independent learning-curve covariate:
  $$f_{learn}(N_{starts}) = 1.0 - \exp\left(-\frac{N_{starts}}{\tau_{learn}}\right)$$
  where $\tau_{learn} \approx 25\text{ starts}$. It acts as an exogenous feature, never as an artificial multiplier on latent skill $\mu_i$ or deviation $\phi_i$.

---

## 3. Constructor Performance Index (CPI)

### 3.1 Directly Measured vs. Inferred Telemetry Quantities
PACE establishes an unambiguous separation:

| Telemetry Quantity | Classification | Source | Measurement Nature | Uncertainty / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Speed Trap Velocity ($ST_{speed}$)** | **Directly Measured** | OpenF1 / FastF1 | Transponder / radar velocity at timing line | Confounded by tow, DRS, wind. |
| **Sector Times ($S_1, S_2, S_3$)** | **Directly Measured** | Official Timing Loops | Transponder elapsed split seconds | Conflates power, drag, grip, setup. |
| **Micro-Sector & Apex Speeds** | **Directly Measured** | GPS / Wheel Sensors | Minimum speed at corner apex | Confounded by line, fuel, grip. |
| **Latent Fuel Mass ($m_{fuel}$)** | **Latent Inferred** | Mixture / Stint Model | Probabilistic run-plan classification | **High Uncertainty.** Teams hide fuel plans. |
| **Aerodynamic Downforce Ratio** | **Latent Inferred** | Physics Proxy | Ratio of high-speed corner min speed to $v_{max}$ | **High Uncertainty.** Setup dependent. |
| **Mechanical Traction Index** | **Latent Inferred** | Physics Proxy | Low-speed longitudinal acceleration ($a_x$) | **Moderate Uncertainty.** Diff/throttle confounded. |
| **Tyre Degradation Slope ($k_{deg}$)** | **Latent Inferred** | Huber M-Regression | Corrected lap time progression over stints | **Moderate Uncertainty.** Traffic/fuel confounded. |

> [!IMPORTANT]
> Sector times alone **never** isolate downforce or mechanical grip. PACE treats aerodynamic efficiency, mechanical traction, and degradation as *latent inferred parameters* carrying explicit standard errors $(\sigma_{aero}, \sigma_{mech}, \text{SE}(k_{deg}))$.

### 3.2 Latent Fuel-State Inference & Validation Protocol
Exact fuel mass is never directly observable in public timing data. Friday practice run plans are closely guarded team secrets.

#### Mathematical Formulation of Latent Fuel State:
Instead of treating fuel as known or hardcoding $-0.06\text{ s/lap}$, PACE formulates fuel load as a latent categorical mixture variable $Z \in \{\text{Low Fuel (Qualy Sim)}, \text{Medium Fuel}, \text{High Fuel (Race Sim)}\}$:
$$P(Z_k \mid L_{stint}, \bar{T}_{stint}, \sigma_T) = \frac{P(L_{stint} \mid Z_k) \cdot P(\bar{T}_{stint} \mid Z_k) \cdot P(Z_k)}{\sum_m P(L_{stint} \mid Z_m) \cdot P(\bar{T}_{stint} \mid Z_m) \cdot P(Z_m)}$$
- Short stints ($L \le 6\text{ laps}$): High posterior probability of Low Fuel ($15 \pm 10\text{ kg}$).
- Long stints ($L \ge 10\text{ laps}$): High posterior probability of High Fuel ($60 \pm 20\text{ kg}$).
- **Circuit Sensitivity Scaling:** Derived from physical track parameters:
  $$S_{fuel, c} = k_{mass} \cdot \left(\frac{L_{circuit}}{5.0}\right) \cdot (1 + \beta_{accel} \cdot N_{braking\_zones})$$
- **Degradation Huber Estimation:** Lap times $T(l)$ on long stints are fitted via Huber M-regression:
  $$T(l) = T_0 + \left(k_{deg} - S_{fuel, c} \cdot \frac{\mathbb{E}[m_{fuel,0} \mid Z]}{L_{race}}\right) \cdot l + \epsilon_l$$
  yielding degradation slope $k_{deg}$ and standard error $\text{SE}(k_{deg})$.

#### Fuel-State Ablation & Exclusion Protocol:
The latent fuel-state inference mechanism will be empirically validated through chronological ablation testing:
- **Model A (With Latent Fuel Inference):** Degradation and pace adjusted for latent fuel mixture state.
- **Model B (Unadjusted Baseline):** Raw stint progression without latent fuel state modeling.
If Model A fails to produce a statistically significant out-of-sample improvement in lap-time prediction error or race rank MAE ($p < 0.05$), the latent fuel feature will be **formally excluded** from the production pipeline to avoid unvalidated complexity.

---

## 4. Driver–Circuit Intelligence Integration

### 4.1 Empirical Bayes Hierarchical Shrinkage
To prevent small-sample distortion for debutants while capturing track mastery for veterans:
$$\widehat{\mu}_{i,c} = \alpha_{i,c} \cdot \bar{y}_{i,c} + (1 - \alpha_{i,c}) \cdot \mu_i^{global}, \quad \alpha_{i,c} = \frac{N_{i,c}}{N_{i,c} + \kappa_0}$$
Posterior variance is strictly propagated downstream:
$$\text{Var}(\widehat{\mu}_{i,c}) = \frac{\sigma_{within}^2}{N_{i,c} + \kappa_0}$$

### 4.2 Temporal Weighting Hypotheses
Evaluated via chronological backtesting against four competing hypotheses:
- **H1 (Exponential Half-Life):** $w = \exp(-\lambda \Delta t), \lambda_{half} = 3\text{ years}$.
- **H2 (Discrete Era Step-Down):** $w \in \{1.00, 0.60, 0.25, 0.10\}$.
- **H3 (Random Walk):** State-space latent evolution.
- **H4 (Uniform):** Time-invariant track geometry affinity.

---

## 5. Race Outcome Dependencies & Complete Outcome Space

### 5.1 The Complete Outcome Space Partition
For every driver $i$ across $N$ starters, the outcome space partitions into mutually exclusive states summing strictly to 1.0:
$$\sum_{k=1}^N P(\text{Classified } k) + P(\text{Mechanical DNF}) + P(\text{Incident DNF}) + P(\text{DNS}) + P(\text{DSQ}) = 1.0$$

### 5.2 Two-Stage Joint Outcome Decomposition
- **Stage A (Calibrated DNF):** Logistic regression with isotonic calibration predicting $P(DNF_{i,t})$ based on unreliability $(1 - Rel_c)$, midfield congestion (Grid P8–P16), circuit Safety Car rate, and pace variance $CV_{pace}$.
- **Stage B (Plackett-Luce Finishing Ranks):** Conditioned on classification, finishing position probabilities $P(\text{Rank } k \mid \text{Classified})$ are computed via choice utilities $s_i$.
- **Stage C (Expected Position):** Combines classified distribution and DNF penalty:
  $$\mathbb{E}[Pos_i] = \sum_{k=1}^N k \cdot P(\text{Finish } k \mid \text{Classified}) \cdot (1 - P(DNF_i)) + N \cdot P(DNF_i)$$

### 5.3 Limitations of Marginal DNF Modeling & Multi-Car Dependencies
PACE explicitly documents the physical limitations of evaluating DNFs marginally per driver:
1. **Collision Coupling:** First-lap collisions or wet-weather aquaplaning incidents eliminate multiple competitors in a single coupled event. Starters in close grid proximity have highly correlated crash hazards.
2. **Safety Car Field Bunches:** A single retirement triggers a Safety Car or Virtual Safety Car, which wipes out accumulated time deltas, eliminates dirty-air gaps, cools tyre carcasses, and causes synchronized pit stop rushes.
3. **Engine Supplier Correlated Failures:** Extreme ambient heat or altitude can cause systemic cooling or MGU-H failures across multiple customer teams sharing the same power unit supplier.
4. **Strategic Cascades:** An undercut or pit stop mistake by a race leader cascades into reactive pit calls across the entire chasing pack.

#### Architectural Extension Path to Phase 4 (Stochastic Simulation):
The Phase 3 analytical DNF model provides the **calibrated marginal baseline hazard rates** $P(DNF_i)$. In Phase 4, these marginal rates are integrated into a **lap-by-lap joint Markovian race simulator**:
- Lap-level hazard functions conditioned on track position and proximity.
- Dynamic Safety Car state triggers with probabilistic bunching.
- Multi-car collision clustering and team strategy response logic.

---

## 6. End-to-End Uncertainty Propagation

### 6.1 Flaws of Naive Standard Deviation Columns in GBDTs
> [!WARNING]
> Passing standard deviation columns (`dpi_uncertainty`, `circuit_affinity_se`) into gradient boosted decision trees (GBDTs / XGBoost) does **NOT** constitute mathematical uncertainty propagation. Decision tree split points on variance columns merely act as arbitrary heuristic gates; they do not propagate probability distributions into forecast variance.

### 6.2 Comparative Uncertainty Propagation Framework
PACE establishes and compares four rigorous uncertainty propagation mechanisms:

```
                                  [ Upstream Uncertainties ]
                DPI: (mu, phi) | CPI: (pace, SE) | Circuit: (aff, SE) | DNF: P(DNF)
                                             │
      ┌───────────────────────────┬──────────┴────────────────┬───────────────────────────┐
      ▼                           ▼                           ▼                           ▼
[ Approach 1:             [ Approach 2:               [ Approach 3:               [ Approach 4:
  Bayesian LMM Posterior    Monte Carlo Latent          Chronological Bootstrap     Plackett-Luce Direct
  Predictive Draws ]        Utility Permutations ]      Model Ensembles ]           Probabilistic Choice ]
  Full Gibbs/MCMC draw      Sample s^(m) ~ N(s, Sigma)  Train K models across       Forward-backward
  from joint posterior      Permutation solve P1..PN    expanding resamples         DP choice recursion
```

1. **Approach 1: Bayesian LMM Posterior Predictive Draws:**
   In the Hierarchical Linear Mixed Model, driver and team effects are drawn directly from their joint posterior multivariate normal distribution:
   $$\begin{bmatrix} \mathbf{u}^{(m)} \\ \mathbf{v}^{(m)} \end{bmatrix} \sim N\left( \begin{bmatrix} \widehat{\mathbf{u}} \\ \widehat{\mathbf{v}} \end{bmatrix}, \Sigma_{posterior} \right)$$
   Yields exact analytical posterior intervals for expected finish positions.
2. **Approach 2: Monte Carlo Latent Utility Permutations:**
   For tree rankers, latent utility vectors $\mathbf{s}$ are perturbed across $M = 1,000$ draws:
   $$\mathbf{s}^{(m)} \sim N(\widehat{\mathbf{s}}(\mathbf{x}), \Sigma_u(\mathbf{x})), \quad \Sigma_u = \text{diag}\left( \left(\frac{\phi_i}{350}\right)^2 \sigma_0^2 + \text{SE}(\widehat{\mu}_{i,c})^2 \right)$$
   Sorting $\mathbf{s}^{(m)}$ generates $M$ valid tie-free finishing permutations, producing empirical 90% credible intervals ($P05$ to $P95$).
3. **Approach 3: Chronological Bootstrap Ensembles:**
   Trains an ensemble of $K = 20$ ranking models across chronological block resamples of past events. Variance across ensemble predictions directly quantifies epistemic model uncertainty.
4. **Approach 4: Plackett-Luce Analytic Probabilities:**
   Computes exact marginal choice probabilities through forward-backward recursion over the permutation choice space.

### 6.3 Impact on Final Finishing Predictions
Upstream uncertainty directly impacts the final prediction:
- **Debutants / Low Information:** Drivers with high $\phi$ or limited circuit history have dispersed posterior utilities, expanding their 90% credible interval (e.g., P4 to P18) and flattening their win probability.
- **High Information / Low Variance:** Drivers with extensive clean data have narrow posterior bounds (e.g., P1 to P3).

---

## 7. Context-Dependent Feature Interactions & Explainability

### 7.1 Physics-Guided Learned Interactions
PACE evaluates interaction terms guided by vehicle dynamics:
- $\text{Grid} \times \text{OvertakingDifficulty}$: High starting position dominance at Monaco/Hungaroring vs. low at Spa/Monza.
- $\text{TyreDegradation} \times \text{TyreStress}$: Compound degradation penalty scaling with high-energy lateral tarmac loading.
- $\text{SpeedTrap} \times \text{TrackLength} \times \text{Altitude}$: Power unit terminal velocity importance at Monza and Mexico City.

### 7.2 SHAP Local Additive Attribution vs. Causal Weights
$$\widehat{f}(\mathbf{x}) = \phi_0 + \sum_{j=1}^M \phi_j(\mathbf{x})$$
- SHAP values quantify **local, post-hoc additive feature shifts** from baseline for a specific race.
- SHAP values are **never** converted into model weights or causal dials. Explanations carry mandatory disclaimers.

---

## 8. Empirical Feature Selection & Model Evaluation Plan

### 8.1 Progressive Multi-Model Benchmark Hierarchy
All models are evaluated on strictly identical chronological walk-forward splits:

```
[ Tier 1: Non-Parametric Baselines ]
  ├── B1: Qualifying Grid Order (Finish = Grid)
  ├── B2: Championship Standings Order
  └── B3: Rolling Constructor 3-Race Average Finish

[ Tier 2: Interpretable Statistical Models ]
  ├── M1: Regularized Linear Regression (Ridge / Lasso) with permutation projection
  ├── M2: Ordinal Logistic Regression (Proportional Odds)
  └── M3: Bradley-Terry Paired Choice Model with Covariates

[ Tier 3: Hierarchical Bayesian Latent Skill Models ]
  └── M4: Bayesian Linear Mixed-Effects Model (LMM Driver + Team crossed effects)

[ Tier 4: Supervised Learning-to-Rank GBDTs ]
  ├── M5: XGBoost Pairwise Ranking (rank:pairwise)
  └── M6: XGBoost Listwise LambdaMART (rank:ndcg)
```

### 8.2 Statistical Significance Testing Protocol
- Expanding-window walk-forward backtesting across 2023–2026.
- Pairwise comparisons evaluated using **paired Wilcoxon signed-rank tests** ($p < 0.01$).
- A complex model (e.g. M6) is only selected over an interpretable baseline if it achieves a **statistically significant improvement ($p < 0.01$)** with an effect size $> 5\%$ reduction in MAE.

### 8.3 Systematic Feature Group Ablation Protocol (Occam's Razor)
Every feature group will be evaluated through a rigorous **Leave-One-Group-Out (LOGO)** and **Forward-Group-Addition** ablation experiment across identical cross-validation splits:

| Feature Group | Constituents | Candidate Alternative / Ablation |
| :--- | :--- | :--- |
| **G1: Grid & Session Order** | Starting Grid Position, Q1/Q2/Q3 times | Exclude to measure pure pre-weekend capability |
| **G2: Driver Performance (DPI)**| Glicko-2 $\mu, \phi$, teammate qualifying/race deltas | Compare against raw driver championship points |
| **G3: Decoupled Experience** | $f_{learn}(starts)$, lap consistency $CV_{pace}$ | Exclude to verify whether experience adds predictive power |
| **G4: Constructor Pace (CPI)** | Pole gap %, clean air race pace, trap speed | Compare against raw constructor standings |
| **G5: Inferred Telemetry & Fuel**| Latent fuel mixture state, Huber $k_{deg}$, fuel sens $S_{fuel}$ | Compare against unadjusted stint lap regressions |
| **G6: Circuit Intelligence** | Empirical Bayes track shrinkage, temporal hypotheses | Compare against raw driver circuit averages |
| **G7: Physics Interactions** | $\text{Grid} \times \text{Overtaking}$, $\text{Deg} \times \text{Stress}$ | Compare against purely linear additive terms |

#### Pruning Criteria:
1. Any feature group that fails to produce a measurable reduction in out-of-sample MAE ($\ge 1.0\%$) or improvement in Spearman $\rho$ on identical validation splits will be **pruned from production inference**.
2. No feature group will be retained purely for perceived theoretical elegance.

---

## 9. Feature Availability & Leakage Prevention Protocol

### 9.1 Temporal Prediction Horizons

| Horizon | Prediction Cutoff | Available Data | Strictly Prohibited Future Data |
| :--- | :--- | :--- | :--- |
| **H1: Pre-Weekend** | Thursday 23:59 UTC | Historical DPI, CPI, Circuit Affinity up to $(t-1)$, circuit characteristics, regulation era. | Any practice telemetry, qualifying results, race results. |
| **H2: Post-Friday Practice** | Friday 23:59 UTC (after FP2) | All H1 features + FP1/FP2 sector split times, trap speeds, practice stint degradation slopes. | Saturday FP3, Qualifying results, race results. |
| **H3: Post-Qualifying (Primary)**| Saturday 23:59 UTC (after Qualy) | All H2 features + FP3 times, official Q1/Q2/Q3 times, confirmed starting grid. | Sunday race results, race lap times, race pit stops, race weather. |

### 9.2 Missing Data Protocols
- **Sprint Weekends:** Feature builders detect sprint formats (1 practice session); degradation estimation uncertainty bounds are widened accordingly.
- **Wet Practice Sessions:** Wet telemetry is masked; model falls back to prior dry baselines with widened credible intervals.
- **Grid Penalties:** Starting grid positions reflect confirmed post-penalty starting grids.

---

## 10. Summary of Version 2.1 Refinements & Sign-Off Request

This specification incorporates the five requested refinements:
1. **Experience & Rating Uncertainty:** Glicko-2 $\phi_i$ shrinks strictly via unconfounded competitive match evidence; experience $N_{starts}$ is treated as an independent learning-curve covariate.
2. **Latent Fuel-State Inference:** Fuel load is modeled probabilistically as a latent variable with confidence intervals; retained only if empirical ablation confirms out-of-sample benefit.
3. **Rigorous Uncertainty Propagation:** Compares Bayesian posterior draws, Monte Carlo permutations, and bootstrap ensembles; acknowledges the inadequacy of passing standard deviation columns to GBDTs.
4. **Multi-Car Outcome Dependencies:** Documents physical limitations of marginal DNF models (collision cascades, Safety Car bunching) and formalizes the transition bridge to Phase 4 joint simulation.
5. **Systematic Empirical Feature Selection:** Mandates Leave-One-Group-Out ablation testing across identical validation splits to prune uninformative complexity.

We submit this refined Version 2.1.0 specification for formal sign-off to proceed to Stage 2 implementation.
