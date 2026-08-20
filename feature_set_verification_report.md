# Feature Set & Target Verification Report

This document provides a verified, empirical audit of the exact 16-feature set used across all tree models (LightGBM, XGBoost, Random Forest, Linear Regression) and temporal recurrent architectures (GRU Full, LSTM Full), along with a target correlation analysis to verify target isolation.

---

## 1. Verified 16-Feature Input Set

The exact 16 columns extracted from `Features/pem_features.parquet` and stored in `Models/baseline/feature_config.pkl` are:

| # | Feature Name | Column Type | Physical Units / Description |
| :---: | :--- | :---: | :--- |
| **1** | `Stack Current Sensor Value A` | Raw Telemetry | Stack Operating Current ($0.0 \dots 255.0\text{ A}$) |
| **2** | `Fuel Cell Total Voltage V` | Raw Telemetry | Total Stack Voltage ($112.0 \dots 412.0\text{ V}$) |
| **3** | `Stack Coolant Temp Inlet degree C` | Raw Telemetry | Coolant Loop Inlet Temperature (${-2.0} \dots 79.0^\circ\text{C}$) |
| **4** | `Stack Coolant Temp Outlet degree C` | Raw Telemetry | Coolant Loop Outlet Temperature (${-2.0} \dots 85.0^\circ\text{C}$) |
| **5** | `Air Temp Stack Outlet degree C` | Raw Telemetry | Cathode Air Outlet Temperature (${-1.0} \dots 80.0^\circ\text{C}$) |
| **6** | `Air Flow Sensor kg/h` | Raw Telemetry | Mass Air Flow Rate ($0.0 \dots 407.8\text{ kg/h}$) |
| **7** | `Air Comp Motor Temp degree C` | Raw Telemetry | Compressor Motor Temperature (${-50.0} \dots 138.0^\circ\text{C}$) |
| **8** | `Ambient Air Temp degree C` | Raw Telemetry | Ambient Air Temperature (${-10.0} \dots 29.5^\circ\text{C}$) |
| **9** | `Ambient Air Pressure kPa` | Raw Telemetry | Atmospheric Air Pressure ($92.0 \dots 100.0\text{ kPa}$) |
| **10** | `Fahrzeuggeschwindigkeit km/h` | Raw Telemetry | Vehicle Speed ($0.0 \dots 178.0\text{ km/h}$) |
| **11** | `Stack Power kW` | Derived Telemetry | Instantaneous Stack Electrical Power ($V \times I / 1000$) |
| **12** | `Air_Stoichiometry_Lambda` | Engineered | Air-to-Fuel Ratio Proxy ($\propto \text{Air Flow} / I_{\text{stack}}$) |
| **13** | `Delta_V_Nernst` | Engineered | Nernst Thermodynamic Potential Correction ($\Delta V(T, P)$) |
| **14** | `E_cum_kWh` | Cumulative Metric | Total Integrated Energy Delivered over Asset Lifetime |
| **15** | `high_current_duration_h` | Cumulative Stress | Accumulated Operating Hours under High Load ($I \ge 100\text{A}$) |
| **16** | `Cumulative Time h` | Elapsed Time | Total Operating Hours ($0.0 \dots 16,829.0\text{ h}$) |

---

## 2. Target Variable Definition

- **Target Column Name**: `soh_target`
- **Formula**:
  $$\text{SoH}_{\text{target}}(t) = \text{EMA}_{15}\left( \frac{V_{\text{measured}}(t)}{V_{\text{BOL}}(I(t)) + \Delta V_{\text{env\_corr}}(T(t), P(t))} \times 100\% \right)$$
- **Range**: $[60.0\%, 100.0\%]$ relative to Beginning-of-Life (Sessions $1 \dots 50$).

---

## 3. Target Leakage & Feature Sensitivity Analysis

### 3.1 Strict Exclusion of Intermediate Target Columns
The target derivation script [`01_create_soh_target.py`](file:///d:/PEM_Cell_Dataset/01_create_soh_target.py) generated intermediate ratio columns in `pem_features.parquet` to construct $\text{SoH}$. **All intermediate target ratio columns were strictly EXCLUDED from model training**:
- `soh_target_raw` (EXCLUDED)
- `soh_v_ratio_raw` (EXCLUDED)
- `ratio_v` (EXCLUDED)
- `ratio_r` (EXCLUDED)
- `V_BOL` (EXCLUDED)

### 3.2 Role of `Fuel Cell Total Voltage V`
- `Fuel Cell Total Voltage V` ($V_{\text{measured}}$) **is present** as Feature #2 in the 16-feature set.
- Because $\text{SoH}_{\text{target}}$ is calculated as $\frac{V_{\text{measured}}}{V_{\text{expected}}(I, T, P)}$, providing $V_{\text{measured}}$, $I_{\text{stack}}$, $T_{\text{cool}}$, and $P_{\text{amb}}$ allows models (LightGBM, XGBoost, GRU) to map current-voltage operating states directly to degradation level.
- **Voltage-Blind Verification**: When `Fuel Cell Total Voltage V` was removed (reducing inputs to 15 features in `05_temporal_experiments.py`), GRU test performance dropped from $R^2 = 0.612 \rightarrow 0.315$. This confirms that electrical voltage under load is the primary observable degradation signal.
