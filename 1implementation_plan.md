# Refined Data Cleaning, Preprocessing & SoH Modeling Implementation Plan

This updated implementation plan incorporates user corrections for the PEM Fuel Cell time-series dataset ([`processed.csv`](file:///d:/PEM_Cell_Dataset/Processed/processed.csv)).

---

## Technical Corrections & Architectural Rules

1. **Non-Destructive Pipeline**: `processed.csv` remains strictly read-only. All generated outputs will be written to distinct subdirectories under `d:\PEM_Cell_Dataset\`.
2. **Invalidate to NaN (No Hard Clamping)**: Invalid sensor readings (e.g. $-500\text{ V}$ voltage disconnects or $-50^\circ\text{C}$ motor temp glitches) are converted directly to `NaN` rather than clipped to boundary values, preserving sensor invalidity signal.
3. **Phased Modeling Roadmap (SoH Priority over RUL)**:
   - **Phase A**: Data Quality & Sanitization
   - **Phase B**: Operating-State Separation (Active vs. Idle/Shutdown)
   - **Phase C**: Electro-Thermal Physics Features & Normalization
   - **Phase D**: Degradation / SoH Proxy Identification ($V_{\text{iso-current}}$, $R_{\text{est}}$)
   - **Phase E**: SoH Baseline Models (XGBoost / LightGBM / Random Forest) $\rightarrow$ Temporal Models (LSTM/GRU) $\rightarrow$ PINN + EKF
   - **Phase F**: RUL Modeling (Deferred until defensible EOL ground truth criterion is formally established)
4. **Strict Quality Gating for Internal Resistance ($R_{\text{est}}$)**:
   $R_{\text{est}}$ is computed as $\left| \frac{\Delta V}{\Delta I} \right|$ **ONLY IF ALL** of the following conditions hold:
   - $|\Delta I| \ge 15.0\text{ A}$
   - $\Delta t \approx 4.0\text{ s}$
   - Both $(V_t, I_t)$ and $(V_{t-1}, I_{t-1})$ are valid non-NaN numbers
   - Both observations belong to the **same `session_id`**
   - Both observations are in the **Active State** ($I > 1.0\text{ A} \land V > 50.0\text{ V}$)
   - *Otherwise*: $R_{\text{est}} = \text{NaN}$.

---

## Target Project Layout

```
PEM_Cell_Dataset/
│
├── Raw/
│   └── raw_dataset.csv
│
├── Processed/
│   └── processed.csv                     (Read-only source)
│
├── ML/
│   ├── clean_pem_telemetry_active.csv   (Sanitized active telemetry)
│   └── pem_session_summary.csv          (811 session metadata summaries)
│
├── Features/
│   └── pem_features.parquet             (Engineered feature matrix)
│
├── Splits/
│   ├── train.parquet                    (Time/session-based train split)
│   ├── validation.parquet               (Time/session-based val split)
│   └── test.parquet                     (Time/session-based test split)
│
├── Models/
│   ├── baseline/                        (XGBoost / LightGBM baseline artifacts)
│   ├── soh/                             (SoH regression model checkpoints)
│   └── rul/                             (Future RUL model checkpoints)
│
└── Reports/
    ├── preprocessing_report.json        (Execution metrics & anomaly counters)
    ├── session_report.csv               (Per-session data stats)
    └── feature_report.csv               (Feature correlations & missingness stats)
```

---

## Detailed Data Processing Pipeline

```
  processed.csv (Read-Only Source)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Session Segmentation (gap > 300 s -> 811 id)  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Sensor Sanitization (Invalid -> NaN)           │
│   • V < 0 or V > 450 V -> NaN                          │
│   • T_comp < -10 or T_comp > 120 °C -> NaN             │
│   • P_amb < 80 or P_amb > 120 kPa -> NaN               │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Operating-State Separation                    │
│   • Active State: I_stack > 1.0 A AND V_stack > 50.0 V │
│   • Idle / Key-Off: Preserved in session_summary.csv   │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 4: Intra-Session 4s Uniform Resampling           │
│   • PCHIP interpolation inside session (max gap <=20s) │
│   • NO interpolation across session breaks (>300 s)    │
│   • Set gap_imputed boolean flag                       │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 5: Multi-Condition Physics Feature Engineering   │
│   • E_Nernst(T, P) baseline compensation               │
│   • Gated R_est calculation                            │
│   • Cathode Air Stoichiometry λ_O2                     │
│   • Cumulative Energy E_cum & High-Current Duration    │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 6: Export & Chronological Train/Val/Test Split  │
└────────────────────────────────────────────────────────┘
```

---

## Comprehensive Verification Test Suite

### 1. Data Integrity & Continuity
- [x] No duplicate timestamps across the dataset.
- [x] Timestamps monotonically increasing within each session.
- [x] `session_id` monotonically assigned ($1 \dots 811$).
- [x] Every inter-sample gap $> 300\text{s}$ generates a new `session_id`.
- [x] No individual `session_id` contains an internal gap $> 300\text{s}$.

### 2. Sensor Quality & Sanitization (Invalid $\rightarrow$ NaN)
- [x] $0$ records with $V_{\text{stack}} < 0.0\text{ V}$ or $V_{\text{stack}} > 450.0\text{ V}$ in cleaned data.
- [x] $0$ records with $T_{\text{comp}} < -10.0^\circ\text{C}$ or $T_{\text{comp}} > 120.0^\circ\text{C}$ in cleaned data.
- [x] $0$ records with $P_{\text{amb}} < 80.0\text{ kPa}$ or $P_{\text{amb}} > 120.0\text{ kPa}$ in cleaned data.

### 3. Temporal Integrity & Resampling
- [x] Resampled time interval $\Delta t = 4.0\text{s}$ exactly for all intra-session rows.
- [x] $0$ cross-session interpolation records.
- [x] `gap_imputed` boolean flag present and correctly set for imputed points.
- [x] Original vs interpolated rows clearly identifiable.

### 4. Physical Sanity Checks
- [x] Calculated Stack Power $P_{\text{stack}} = \frac{V \cdot I}{1000}$ physically bounded ($0 \le P \le 90\text{ kW}$).
- [x] $R_{\text{est}}$ finite and non-negative where calculated; `NaN` elsewhere.
- [x] Cathode Air Stoichiometry $\lambda_{\text{O}_2}$ finite during active power delivery.
- [x] Cumulative Energy $E_{\text{cum}}$ strictly monotonically increasing across active time.
- [x] High-current residence time $t_{I > 150\text{A}}$ strictly monotonically increasing.

### 5. Data Leakage Prevention
- [x] Zero future information in feature windows.
- [x] Rolling window features rely exclusively on past/current intra-session data.
- [x] Cumulative features integrated chronologically without lookahead.
- [x] Train/Validation/Test splits performed strictly along session time boundaries (e.g., 70% Train, 15% Val, 15% Test chronologically).

---

## Proposed Changes & Script Implementation

#### [NEW] [preprocess_pem_dataset.py](file:///d:/PEM_Cell_Dataset/Processed/preprocess_pem_dataset.py)
Master Python pipeline script that:
1. Creates required directory structure (`ML`, `Features`, `Splits`, `Models`, `Reports`).
2. Reads `Processed/processed.csv` in chunks/session-wise.
3. Applies segmentation, invalid-to-NaN sanitization, state separation, 4s PCHIP resampling, multi-condition $R_{\text{est}}$ gating, and cumulative feature extraction.
4. Saves parquet/csv outputs and JSON preprocessing report.

#### [NEW] [verify_preprocessing.py](file:///d:/PEM_Cell_Dataset/Processed/verify_preprocessing.py)
Verification test runner executing the 5-point verification matrix and printing pass/fail audit metrics.

---

## Execution Plan & Deliverables

1. Create target directories under `d:\PEM_Cell_Dataset\`.
2. Write and execute `preprocess_pem_dataset.py`.
3. Write and execute `verify_preprocessing.py`.
4. Inspect output dataset statistics, session counts, and generated reports.
