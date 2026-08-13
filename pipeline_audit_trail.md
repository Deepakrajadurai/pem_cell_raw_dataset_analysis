# PEM Fuel Cell Data Processing — Process Audit Trail & Backtesting Guide

> [!IMPORTANT]
> This document provides a complete audit trail and reproducible track record for all processing, statistical analysis, feature engineering, session segmentation, polarization modeling, and visual plot generation performed on the raw PEM Fuel Cell dataset ([raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)).

---

## 1. Directory & File Inventory Track Record

Below is the complete file manifest for the processing pipeline:

```
d:\PEM_Cell_Dataset\Raw\
├── raw_dataset.csv                        # Raw input telemetry dataset (302,212 rows, 25.69 MB)
├── pem_cell_raw_dataset_analysis_report.md# Master technical report
├── pipeline_audit_trail.md               # This backtesting audit guide
├── polarization_curve.png                 # Generated polarization curve figure
├── operating_regimes.png                  # Generated operating regime pie chart
├── correlation_matrix.png                 # Generated correlation matrix heatmap
├── sample_drive_session.png               # Generated drive session time-series chart
├── scripts/                               # Reproducible Python scripts folder
│   ├── step01_analyze_dataset.py          # Step 1: Profiling & statistics script
│   ├── step02_deep_analysis.py            # Step 2: Session, polarization & anomaly script
│   ├── step03_generate_charts.py          # Step 3: Chart generation script
│   └── run_pipeline.py                    # Master end-to-end pipeline runner
└── outputs/                               # Pipeline output artifacts folder
    ├── analysis_summary.json              # Statistical profile JSON output
    ├── deep_analysis.json                 # Deep time series & correlation JSON output
    ├── polarization_curve.png             # Output polarization chart copy
    ├── operating_regimes.png              # Output load regime chart copy
    ├── correlation_matrix.png             # Output correlation matrix chart copy
    └── sample_drive_session.png           # Output drive session chart copy
```

---

## 2. Step-by-Step Pipeline Audit Record

### Step 1: Data Ingestion, Profiling & Summary Statistics
- **Script**: [step01_analyze_dataset.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step01_analyze_dataset.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output File**: [outputs/analysis_summary.json](file:///d:/PEM_Cell_Dataset/Raw/outputs/analysis_summary.json)
- **Key Operations**:
  1. Parses raw timestamp strings into UTC datetime objects (`Time_dt`).
  2. Sorts dataset chronologically and validates time ordering.
  3. Computes dataset span (**701.21 days** / 16,829.05 hours).
  4. Calculates nominal sampling interval (Median: **4.0 s**, Mean: **200.47 s** due to inter-session gaps).
  5. Derives **Stack Electrical Power ($P = V \times I$)** and **Coolant Delta Temp ($\Delta T = T_{outlet} - T_{inlet}$)**.
  6. Calculates mean, standard deviation, min, 25%, 50%, 75%, max, skewness, kurtosis, missing count, and missing percentage for all 13 numeric parameters.
  7. Categorizes operating regimes ($I \le 1\text{A}$, $1-50\text{A}$, $50-150\text{A}$, $>150\text{A}$).
  8. Integrates total electrical energy delivered (**7,153.93 kWh**).

**Backtest Execution Command**:
```bash
python scripts/step01_analyze_dataset.py
```

---

### Step 2: Deep Time-Series Analysis & Operational Modeling
- **Script**: [step02_deep_analysis.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step02_deep_analysis.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output File**: [outputs/deep_analysis.json](file:///d:/PEM_Cell_Dataset/Raw/outputs/deep_analysis.json)
- **Key Operations**:
  1. **Drive Session Boundary Detection**: Identifies inter-sample gaps $> 300\text{ seconds}$ to partition the time series into **811 distinct driving sessions** (793 active sessions $>10$ samples).
  2. **Session Metrics**: Computes average (**26.9 min**), median (**12.5 min**), and maximum (**98.7 min**) session durations.
  3. **Pearson Correlation Matrix**: Computes pair-wise linear correlation coefficients across 10 primary telemetry streams.
  4. **Polarization Curve Extraction**: Groups positive voltage and current samples into 10 A current bins ($0-260\text{ A}$) and calculates average stack voltage and standard deviation for each load interval.
  5. **Data Quality & Anomaly Audit**: Counts sensor power-down drops ($V \le 0\text{ V}$: 1,850 rows) and temperature sensor spikes ($T_{comp} < -20\text{ °C}$: 24 rows).

**Backtest Execution Command**:
```bash
python scripts/step02_deep_analysis.py
```

---

### Step 3: Visualization & Figure Generation
- **Script**: [step03_generate_charts.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step03_generate_charts.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output Figures**:
  - `polarization_curve.png`: Dual-axis $V-I$ and $P-I$ curve with standard deviation shading.
  - `operating_regimes.png`: Pie chart of time distribution across load levels.
  - `correlation_matrix.png`: Annotated Pearson correlation heatmap matrix.
  - `sample_drive_session.png`: Multi-channel time-series plot of a 65-minute drive session.

**Backtest Execution Command**:
```bash
python scripts/step03_generate_charts.py
```

---

### Step 4: Master End-to-End Pipeline Execution
- **Script**: [run_pipeline.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/run_pipeline.py)
- **Function**: Executes Step 1, Step 2, and Step 3 sequentially, verifying data integrity and reporting stage timings.

**End-to-End Backtest Command**:
```bash
python scripts/run_pipeline.py
```

---

## 3. Backtesting Verification Matrix

Use the validation table below to verify that your re-execution produces identical analytical results:

| Verification Metric | Expected Backtest Value | Pass Criteria |
| :--- | :--- | :--- |
| **Total Row Count** | 302,212 rows | Exact match |
| **Time Span Start** | `2024-08-12 09:22:15+00:00` | Exact match |
| **Time Span End** | `2026-07-14 14:25:29+00:00` | Exact match |
| **Total Energy Output** | 7,153.93 kWh | $\pm 0.1\text{ kWh}$ |
| **Peak Power Output** | 80.96 kW | $\pm 0.01\text{ kW}$ |
| **Active Drive Sessions ($>10$ samples)** | 793 sessions | Exact match |
| **Negative Voltage Glitch Count** | 1,850 rows | Exact match |
| **Current vs Speed Correlation ($r$)** | 0.62 | Exact match |
| **Pipeline Execution Time** | ~3.1 seconds | $< 10.0\text{ seconds}$ |

---

## 4. Environment & Dependencies

To execute backtests on any clean machine, ensure the following Python environment is configured:

```bash
python -m pip install pandas numpy matplotlib scipy
```

- **Python Version**: 3.10+ (Tested on Python 3.14.0)
- **Dependencies**: `pandas >= 2.0.0`, `numpy >= 1.24.0`, `matplotlib >= 3.7.0`

---
*Audit Trail Document maintained by AI Data Processing Agent.*
