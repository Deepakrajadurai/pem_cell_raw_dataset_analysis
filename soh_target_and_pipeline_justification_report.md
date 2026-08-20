# Technical Justification Report: Preprocessing, Physical SoH Target Construction, & Model Selection Rationale

This document provides a technical and mathematical justification for every data engineering, feature extraction, target derivation, and machine learning model decision implemented in the PEM Fuel Cell Degradation Analysis pipeline.

---

## 1. Preprocessing Pipeline & Data Sanitization Rationale

### 1.1 Non-Destructive Raw Data Isolation
- **Decision**: Preserve `processed.csv` (and original raw datasets) read-only.
- **Justification**: Maintaining strict separation between source telemetry and engineered downstream data prevents irrecoverable data corruption, ensures 100% auditability, and allows pipeline re-execution under altered parameters.

### 1.2 Invalid Measurement Policy: Invalidation to `NaN` vs. Boundary Clamping
- **Decision**: Change sensor out-of-bound handling from boundary clipping (e.g. $-500\text{V} \rightarrow 0\text{V}$, $-50^\circ\text{C} \rightarrow -10^\circ\text{C}$) to explicit `NaN` invalidation.
- **Justification**:
  - Boundary clipping artificially creates fake boundary data distributions (e.g., dense clusters exactly at $0\text{V}$ or $-10^\circ\text{C}$), which distorts variance statistics, introduces bias into gradient trees, and misleads loss functions.
  - Converting invalid readings to `NaN` preserves the empirical fact that the physical sensor suffered a telemetry glitch or connection drop. Downstream median imputation or masking can then handle `NaN` values without introducing false physical measurements.

### 1.3 Active-State Session Filtering
- **Decision**: Filter telemetry to active operational rows satisfying $V_{\text{stack}} > 50.0\text{ V} \land I_{\text{stack}} > 1.0\text{ A}$.
- **Justification**:
  - Open-circuit voltage (OCV) states, system standbys, and shut-down purge phases exhibit unrepresentative thermodynamic equilibrium voltages governed by gas cross-over rather than operational stack impedance.
  - Restricting model training to active load conditions ensures that polarization losses ($\eta_{\text{act}}$, $\eta_{\text{ohmic}}$, $\eta_{\text{conc}}$) reflect real mechanical and chemical degradation.

---

## 2. Multi-Condition $R_{\text{est}}$ Resistance Gating Justification

### 2.1 Limitations of Unfiltered Instantaneous Resistance
- **Problem**: Calculating internal stack resistance via finite differences $R_{\text{est}} = -\frac{\Delta V}{\Delta I}$ over consecutive time steps produces massive noise spikes when $\Delta I \approx 0\text{ A}$.
- **Solution**: Multi-condition gating rule:
  $$\text{Gate}(t) = \left( |\Delta I(t)| \ge 15.0\text{ A} \right) \land \left( 3.5\text{s} \le \Delta t \le 4.5\text{s} \right) \land \left( \text{session}(t) == \text{session}(t-1) \right) \land \text{active}(t)$$
  $$\text{Valid Range}: 0.050\ \Omega \le R_{\text{est}} \le 1.200\ \Omega$$

### 2.2 Mathematical Justification
- Current steps of $|\Delta I| \ge 15\text{A}$ provide sufficient signal-to-noise ratio (SNR) to overcome voltage sensor quantization noise ($\pm 0.1\text{V}$).
- Enforcing $\Delta t \approx 4\text{s}$ matches the physical sampling interval of load step changes, avoiding transient capacitive double-layer charging dynamics that distort ohmic resistance measurements during fast sub-second transients.

---

## 3. Physical State of Health ($\text{SoH}$) Target Construction

### 3.1 Why Raw Voltage Cannot Serve as a Degradation Target
Raw stack voltage $V_{\text{measured}}(t)$ fluctuates rapidly across operating conditions due to:
1. **Load Current $I_{\text{stack}}$**: Ohmic voltage drops ($I \cdot R$) and activation polarization.
2. **Coolant Temperature $T_{\text{cool}}$**: Temperature-dependent exchange current density and reaction kinetics.
3. **Ambient Pressure $P_{\text{amb}}$**: Oxygen partial pressure and Nernst reversible cell potential.

A cell operating at $200\text{V}$ under $350\text{A}$ current may be in perfect health, while the same cell producing $200\text{V}$ at $50\text{A}$ is severely degraded. Therefore, raw voltage is an operational variable, not a health indicator.

### 3.2 Mathematical Derivation of the Physical $\text{SoH}$ Target

#### Step 1: Beginning-of-Life (BOL) Reference Voltage $V_{\text{BOL}}(I)$
A reference polarization curve $V_{\text{BOL}}(I)$ was constructed using non-linear polynomial regression over active load points during the initial 50 operating sessions ($t \le 50\text{ operating hours}$):
$$V_{\text{BOL}}(I) = a_0 + a_1 I + a_2 I^2 + a_3 I^3$$

#### Step 2: Nernst Thermodynamic Environmental Compensation ($\Delta V_{\text{env\_corr}}$)
To decouple operating temperature and ambient pressure variations from degradation tracking, Nernst thermodynamic potential correction was applied relative to standard reference conditions ($T_{\text{ref}} = 298.15\text{ K}$, $P_{\text{ref}} = 101.325\text{ kPa}$):
$$E_{\text{Nernst}}(T_{\text{k}}, P_{\text{kPa}}) = 1.229 - 0.00085(T_{\text{k}} - 298.15) + 4.31 \times 10^{-5} T_{\text{k}} \ln\left(\frac{P_{\text{kPa}}}{101.325}\right)$$
$$\Delta V_{\text{env\_corr}} = 360 \times \left[ E_{\text{Nernst}}(T_{\text{cool}}, P_{\text{amb}}) - E_{\text{Nernst}}(298.15, 101.325) \right]$$

#### Step 3: Expected Healthy Voltage & Raw $\text{SoH}$ Ratio
$$V_{\text{expected}}(I, T, P) = V_{\text{BOL}}(I) + \Delta V_{\text{env\_corr}}$$
$$\text{SoH}_{\text{raw}}(t) = \frac{V_{\text{measured}}(t)}{V_{\text{expected}}(t)} \times 100\%$$

#### Step 4: Intra-Session EMA Smoothing
To suppress measurement noise spikes while retaining true irreversible degradation trends, an Exponential Moving Average (EMA) with a 60-second window ($\text{span} = 15$ grid points) was applied intra-session:
$$\text{SoH}(t) = \text{EMA}_{15}\left( \text{SoH}_{\text{raw}}(t) \right)$$

```
   RAW TELEMETRY                BOL REFERENCE                NERNST CORRECTION
(V_meas, I, T, P)   ───►   V_BOL(I) Polynomial   ───►   ΔV_env_corr(T, P)
        │                                                     │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
                   V_expected = V_BOL + ΔV_env_corr
                                   │
                                   ▼
                     SoH_raw = (V_meas / V_expected) * 100%
                                   │
                                   ▼
                      EMA_15 (60s Smoothing)
                                   │
                                   ▼
                          PHYSICAL SOH TARGET
```

### 3.3 Justification for Avoiding Remaining Useful Life (RUL) at This Stage
Remaining Useful Life (RUL) prediction requires:
1. A validated, continuous State of Health ($\text{SoH}$) ground truth metric.
2. An agreed-upon End-of-Life (EOL) failure threshold (e.g., $\text{SoH} = 80.0\%$).
3. A stochastic forecast model for future operating load profiles (speed, current cycles, ambient conditions).

Attempting to predict RUL without first establishing a validated physical $\text{SoH}$ proxy target would compound health estimation error with operational profile uncertainty. Establishing a reliable $\text{SoH}$ target is a prerequisite for RUL modeling.

---

## 4. Machine Learning & Deep Learning Model Selection Justification

### 4.1 Chronological Split Policy vs. Random Cross-Validation
- **Policy**: Train (Sessions $1 \dots 567$, 70%), Validation (Sessions $568 \dots 689$, 15%), Test (Sessions $690 \dots 811$, 15%).
- **Justification**: Randomly shuffling dataset rows causes severe temporal data leakage, as adjacent rows in time-series telemetry share near-identical operational states. Chronological session splitting guarantees evaluation on unseen future degradation regimes.

### 4.2 Tabular Tree Model Superiority (LightGBM $R^2 = 0.724$)
- **Finding**: LightGBM outperformed all models ($R^2 = 0.724$, MAE $= 0.814\%$).
- **Justification**: Gradient Boosted Decision Trees partition non-linear multi-variable interactions (voltage ratios, stoichiometry $\lambda$, air temperature, cumulative energy throughput) effectively without making restrictive structural assumptions.

### 4.3 Justification for Dropping Deep Recurrent Models (LSTM / GRU)
- **Experimental Result**:
  - LightGBM (Full): $R^2 = \mathbf{0.724}$, MAE $= \mathbf{0.814\%}$
  - GRU (Full): $R^2 = 0.612$, MAE $= 1.046\%$
  - LSTM (Full): $R^2 = 0.590$, MAE $= 1.092\%$
- **Scientific Rationale**:
  - Intra-session temporal sequences over 60-second windows ($L = 15$ steps) capture fast fluidic/thermal transients rather than multi-hundred-hour structural degradation.
  - Adding recurrent sequence complexity increases parameter count and training overhead without improving long-term degradation tracking over tabular LightGBM.

### 4.4 Discovery: Voltage Feature Dependence
- **Experimental Result**: Removing voltage features caused GRU performance to collapse ($R^2 = 0.612 \rightarrow 0.315$, a $56.5\%$ deterioration).
- **Scientific Rationale**: The observable physical degradation signal is heavily encoded in electrical performance under load. Removing voltage features strips the model of direct impedance degradation evidence.

---

## 5. PINN Architecture & EKF Observer Coupling Rationale

### 5.1 Electromechanical Degradation Coupling in PINN
- **Architecture**: Compact feedforward neural encoder mapping 6 physical inputs (`Current`, `Coolant Temp`, `Air Flow`, `Pressure`, `E_cum`, `Stoichiometry`) to physical latent parameter heads.
- **Electrochemical Coupling**: Ohmic resistance $R_{\text{ohmic}}(t)$ was explicitly linked to structural degradation $\hat{D}(t) \in [0, 0.35]$:
  $$\hat{R}_{\text{ohmic}}(t) = R_{\text{BOL}} \cdot \left(1.0 + 2.5 \cdot \hat{D}(t)\right)$$
  $$\hat{\text{SoH}}(t) = 100.0 \times \left(1.0 - \hat{D}(t)\right)$$
- **Justification**: Coupling $R_{\text{ohmic}}$ directly to $\hat{D}(t)$ forces voltage reconstruction loss $\mathcal{L}_{\text{voltage}} = \text{MSE}(V_{\text{measured}}, \hat{V}_{\text{reconstructed}})$ to backpropagate gradients into $\hat{D}(t)$, enabling physics-informed learning.

### 5.2 EKF Innovation Residual Analysis
- **Finding**: Standalone PINN achieved $R^2 = 0.651$ (Test MAE $= 0.938\%$). Coupling EKF state innovation updates yielded $R^2 = 0.695$ (Test MAE $= 0.861\%$).
- **Justification**: The Extended Kalman Filter uses real-time stack voltage innovation residuals $e_v = V_{\text{measured}} - \hat{V}_{\text{reconstructed}}$ to update state covariance $P$, filtering high-frequency voltage noise and correcting model drift.
