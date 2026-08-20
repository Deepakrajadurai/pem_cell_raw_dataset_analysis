# Comprehensive Data Cleaning & Preprocessing Pipeline Procedure Plan

This document outlines the systematic, step-by-step technical procedure to clean, filter, resample, impute, and feature-engineer the PEM Fuel Cell time-series telemetry dataset ([`processed.csv`](file:///d:/PEM_Cell_Dataset/Processed/processed.csv)) before feeding it into machine learning models for State of Health (SoH) and Remaining Useful Life (RUL) prediction.

---

## Data Anomaly & Audit Summary

Analysis of the raw and processed telemetry revealed six critical dataset defects that compromise ML model training if unaddressed:

1. **Negative Sensor Glitches**: 1,850 rows exhibit negative voltages (min down to $-500.0\text{ V}$), and 24 rows record negative motor temperatures (down to $-50.0^\circ\text{C}$) due to CAN bus disconnects and key-off power-down states.
2. **Operational Shutdown Transients**: Voltage rapidly decays from $\sim 374\text{ V} \rightarrow 4\text{ V}$ when $I_{\text{stack}} = 0\text{ A}$ during vehicle shutoff/purge sequences.
3. **Session Breaks & Non-Continuous Time Gaps**: The dataset contains 811 discrete drive sessions separated by time gaps ranging from $5\text{ minutes}$ up to $96.9\text{ days}$.
4. **Irregular Time Intervals**: Sample timestamps vary dynamically ($\Delta t \in [4\text{s}, 8\text{s}]$, median $4.0\text{s}$) due to event-driven CAN bus logging.
5. **Missing Telemetry Data**: Core sensor channels exhibit $1.1\% - 1.3\%$ missing values ($\sim 3,900$ rows), while CAN vehicle channels exhibit $6.5\%$ missingness ($\sim 19,700$ rows).
6. **Dynamic Load & Environment Non-Stationarity**: Ambient temperature ($T_{\text{amb}} \in [-10^\circ\text{C}, 29.5^\circ\text{C}]$) and pressure variations mask intrinsic electrochemical degradation.

---

## User Review Required

> [!IMPORTANT]
> **Filtering Operational Shutoff Transients**:
> Rows with zero stack current ($I_{\text{stack}} \le 1.0\text{ A}$) and low stack voltage ($V_{\text{stack}} \le 50.0\text{ V}$) will be isolated from continuous degradation regression training. However, these shutdown events will be preserved in a separate session metadata table to track total Start-Stop cycle counts ($N_{\text{cycles}}$), which drive catalyst degradation.

> [!WARNING]
> **Intra-Session Interpolation Window**:
> To prevent non-physical artifact generation, linear interpolation/PCHIP resampling will ONLY occur within individual drive sessions (`session_id`). Interpolation will strictly NEVER cross session boundaries ($> 300\text{s}$ gap).

---

## Proposed Cleaning & Preprocessing Procedure Pipeline

```
  Raw CSV Telemetry [processed.csv]
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 1: Drive Session Segmentation & Labeling  │ ──► Identify 811 sessions (gap > 300 s)
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 2: Outlier Removal & Invalid Range Clamp  │ ──► Clamp V < 0, T_comp < -10°C, P < 80 kPa
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 3: State-Gated Operational Filtering      │ ──► Separate Active (I>1A, V>50V) vs Idle/Purge
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 4: Intra-Session Resampling & Imputation  │ ──► Resample to uniform Δt = 4.0s (PCHIP + LOCF)
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 5: Electro-Thermal Feature Engineering    │ ──► E_Nernst, R_est, λ_O2, E_cum, Moving Averages
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 6: Iso-Current Normalization & Scaling    │ ──► RobustScaler / StandardScaler output
└─────────────────────────────────────────────────┘
```

---

### Step 1: Drive Session Segmentation & Continuity Enforcement
- **Threshold Rule**: Compute $\Delta t_i = t_i - t_{i-1}$. If $\Delta t_i > 300\text{ seconds}$, trigger a new session boundary and assign incremented `session_id`.
- **Derivatives Protection**: All temporal rolling features ($dV/dt$, $dT/dt$, moving averages) are strictly grouped by `session_id` to prevent cross-boundary pollution.

### Step 2: Physical Outlier Removal & Sensor Sanitization
- **Voltage Range Gate**: Set invalid values where $V_{\text{stack}} < 0.0\text{ V}$ or $V_{\text{stack}} > 450.0\text{ V}$ to `NaN`.
- **Temperature Gate**: Set invalid values where $T_{\text{comp}} < -10.0^\circ\text{C}$ or $T > 120.0^\circ\text{C}$ to `NaN`.
- **Pressure Gate**: Set $P_{\text{amb}} < 80.0\text{ kPa}$ or $P_{\text{amb}} > 120.0\text{ kPa}$ to `NaN`.

### Step 3: Operational State Decomposition
Decompose dataset into two parallel streams:
1. **Active Power Delivery Dataset** ($I_{\text{stack}} > 1.0\text{ A} \quad \land \quad V_{\text{stack}} > 50.0\text{ V}$): Primary dataset for training ML models predicting $\text{SoH}$, Ohmic resistance decay, and voltage polarization degradation.
2. **Session Cycle Summary Dataset**: Aggregates session-level metrics (start timestamp, end timestamp, session duration, start-stop count, idle fraction, energy delivered per session).

### Step 4: Intra-Session Uniform Resampling & Imputation
- For each session:
  - Generate a regular 4-second timestamp grid: $t_{\text{grid}} = t_{\text{start}}, t_{\text{start}}+4\text{s}, \dots, t_{\text{end}}$.
  - Apply **Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)** for continuous thermal and voltage/current signals to prevent overshooting/ringing artifacts.
  - For missing sensor gaps $\le 20\text{s}$, apply forward-fill (LOCF). For gaps $> 20\text{s}$ inside a session, flag as `gap_imputed`.

### Step 5: Physics-Informed Feature Engineering
Generate electro-thermal features to enhance ML predictive capability:
1. **Nernst Thermodynamic Voltage Deviation ($\Delta V_{\text{Nernst}}$)**:
   $$E_{\text{Nernst}}(T, P_{\text{amb}}) = E^0 - \frac{R \cdot T_{\text{cool\_out}}}{2F} \ln\left( \frac{1}{P_{\text{amb}} \cdot 1.0} \right)$$
   $$\Delta V_{\text{Nernst}} = V_{\text{stack}} - N_{\text{cells}} \cdot E_{\text{Nernst}}$$
2. **Instantaneous Internal Resistance Proxy ($R_{\text{est}}$)**:
   $$R_{\text{est}}(t) = \left| \frac{V_{\text{stack}}(t) - V_{\text{stack}}(t-4\text{s})}{I_{\text{stack}}(t) - I_{\text{stack}}(t-4\text{s})} \right| \quad \text{for } |\Delta I| \ge 15\text{ A}$$
3. **Cathode Air Stoichiometry ($\lambda_{\text{O}_2}$)**:
   $$\lambda_{\text{O}_2} = \frac{\dot{m}_{\text{air}}}{I_{\text{stack}} \cdot k_{\text{stoch}}}$$
4. **Cumulative Degradation Stress Integrals**:
   - Active Energy: $E_{\text{cum}} = \int P dt \quad (\text{kWh})$
   - High Load Residence Time: $t_{I > 150\text{A}} = \sum \mathbb{I}(I > 150\text{A}) \cdot \Delta t$
   - Thermal Gradient Stress: $\Delta T_{\text{cool}} = T_{\text{cool\_out}} - T_{\text{cool\_in}}$

### Step 6: Dataset Export & Verification
Export two clean, ML-ready CSV datasets to `d:\PEM_Cell_Dataset\Processed\`:
- `clean_pem_telemetry_active.csv`: Processed 4-second uniform active power series with engineered features.
- `pem_session_summary.csv`: Aggregated session metadata and lifetime cycle counters.

---

## Proposed Changes

### Preprocessing Script & Data Outputs

#### [NEW] [preprocess_pem_dataset.py](file:///d:/PEM_Cell_Dataset/Processed/preprocess_pem_dataset.py)
Python script implementing the complete 6-stage data sanitization, intra-session resampling, PCHIP interpolation, and electro-thermal feature engineering pipeline.

#### [NEW] [clean_pem_telemetry_active.csv](file:///d:/PEM_Cell_Dataset/Processed/clean_pem_telemetry_active.csv)
ML-ready clean telemetry dataset containing sanitized core sensors, uniform 4-second grid timestamps, and engineered features ($\Delta V_{\text{Nernst}}$, $R_{\text{est}}$, $\lambda_{\text{O}_2}$, $E_{\text{cum}}$).

#### [NEW] [pem_session_summary.csv](file:///d:/PEM_Cell_Dataset/Processed/pem_session_summary.csv)
Session summary table tracking Start-Stop cycle counts, session durations, energy output per drive cycle, and thermal stress metrics.

---

## Verification Plan

### Automated Tests
Execute Python verification script `verify_preprocessing.py` to confirm:
1. **Zero Outliers**: Verify $0$ negative voltage records ($V < 0\text{ V}$) and $0$ negative motor temperature records ($T_{\text{comp}} < -10^\circ\text{C}$).
2. **Uniform Time Step**: Assert $\Delta t = 4.0\text{ seconds}$ exactly for all intra-session rows.
3. **Zero Cross-Boundary Interpolation**: Assert $0$ interpolated rows across session breaks ($> 300\text{s}$).
4. **Feature Range Audit**: Check non-null values for $R_{\text{est}}$, $\lambda_{\text{O}_2}$, and monotonic increase of $E_{\text{cum}}$.

```powershell
python d:\PEM_Cell_Dataset\Processed\preprocess_pem_dataset.py
python C:\Users\vijayakr\.gemini\antigravity-ide\brain\17419cdd-121e-45ff-aca9-e35a4225c491\scratch\verify_preprocessing.py
```

### Manual Verification
- Inspect statistical distribution summaries (mean, std, min, max) before vs after preprocessing.
- Verify session segmentation count aligns with detected 811 drive sessions.
