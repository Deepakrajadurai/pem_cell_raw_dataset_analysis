# PEM Fuel Cell Data Processing — Process Audit Trail & Backtesting Guide

> [!IMPORTANT]
> This document provides a complete audit trail and reproducible track record for all processing, statistical analysis, feature engineering, session segmentation, polarization modeling, health indicator (HI) baseline regression, and visual plot generation performed on the raw PEM Fuel Cell dataset ([raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)).

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
├── health_degradation_curve.png           # Defensible SoH degradation curve figure
├── voltage_residual_over_hours.png        # Voltage residual (V_actual - V_expected) figure
├── v_expected_vs_actual.png               # Baseline model parity scatter plot
├── scripts/                               # Reproducible Python scripts folder
│   ├── step01_analyze_dataset.py          # Step 1: Profiling & statistics script
│   ├── step02_deep_analysis.py            # Step 2: Session, polarization & anomaly script
│   ├── step03_generate_charts.py          # Step 3: Visual chart generation script
│   ├── step04_health_indicator.py        # Step 4: Baseline ML model & HI degradation script
│   └── run_pipeline.py                    # Master end-to-end 4-step pipeline runner
└── outputs/                               # Pipeline output artifacts folder
    ├── analysis_summary.json              # Statistical profile JSON output
    ├── deep_analysis.json                 # Deep time series & correlation JSON output
    ├── health_indicator_summary.json      # HI model trajectory JSON output
    ├── health_indicator_data.csv          # Binned hourly SoH data CSV output
    └── *.png                              # Output figure copies
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
  6. Calculates statistics for all 13 parameters.
  7. Categorizes operating regimes ($I \le 1\text{A}$, $1-50\text{A}$, $50-150\text{A}$, $>150\text{A}$).
  8. Integrates total electrical energy delivered (**7,153.93 kWh**).

---

### Step 2: Deep Time-Series Analysis & Operational Modeling
- **Script**: [step02_deep_analysis.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step02_deep_analysis.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output File**: [outputs/deep_analysis.json](file:///d:/PEM_Cell_Dataset/Raw/outputs/deep_analysis.json)
- **Key Operations**:
  1. **Drive Session Boundary Detection**: Identifies inter-sample gaps $> 300\text{ s}$ to partition time series into **811 distinct driving sessions** (793 active sessions $>10$ samples).
  2. **Session Metrics**: Computes average (**26.9 min**), median (**12.5 min**), and maximum (**98.7 min**) session durations.
  3. **Pearson Correlation Matrix**: Computes pair-wise linear correlation coefficients across 10 primary telemetry features.
  4. **Polarization Curve Extraction**: Groups samples into 10 A current bins ($0-260\text{ A}$) and calculates average voltage and std dev.
  5. **Data Quality & Anomaly Audit**: Counts negative voltage drops ($V \le 0\text{ V}$: 1,850 rows) and temperature spikes ($T_{comp} < -20\text{ °C}$: 24 rows).

---

### Step 3: Visualization & Figure Generation
- **Script**: [step03_generate_charts.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step03_generate_charts.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output Figures**: `polarization_curve.png`, `operating_regimes.png`, `correlation_matrix.png`, `sample_drive_session.png`.

---

### Step 4: Health Indicator (HI) Baseline Model & Lifetime Degradation Tracking
- **Script**: [step04_health_indicator.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/step04_health_indicator.py)
- **Input File**: [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv)
- **Output Files**: [outputs/health_indicator_summary.json](file:///d:/PEM_Cell_Dataset/Raw/outputs/health_indicator_summary.json), [outputs/health_indicator_data.csv](file:///d:/PEM_Cell_Dataset/Raw/outputs/health_indicator_data.csv)
- **Output Figures**: `health_degradation_curve.png`, `voltage_residual_over_hours.png`, `v_expected_vs_actual.png`
- **Key Operations**:
  1. **Baseline Regressor**: Fits $V_{\text{expected}} = f(I, T_{\text{coolant\_in}}, T_{\text{coolant\_out}}, T_{\text{air}}, \text{Airflow}, \text{Ambient\_Temp}, \text{Ambient\_Pressure})$ on fresh operating hours ($\le 76,000\text{ h}$). Achieves **$R^2 = 0.9144$** and **$\text{MAE} = 3.29\text{ V}$**.
  2. **Residual Calculation**: Computes $V_{\text{residual}} = V_{\text{actual}} - V_{\text{expected}}$ for every observation.
  3. **State of Health (SoH %)**: Evaluates normalized $\text{SoH}(\%) = 100 + \frac{V_{\text{residual}}}{V_{\text{expected\_nominal}}} \times 100\%$.
  4. **Degradation Tracking**: Aggregates $V_{\text{residual}}$ across `Kumulative Betriebszeit h` (71,492 h to 108,112 h) to establish a defensible degradation index.

---

### Step 5: Master End-to-End Pipeline Execution
- **Script**: [run_pipeline.py](file:///d:/PEM_Cell_Dataset/Raw/scripts/run_pipeline.py)
- **Function**: Executes Step 1, Step 2, Step 3, and Step 4 sequentially in **~6.0 seconds**.

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
| **Baseline Model $R^2$** | 0.9144 | $\pm 0.005$ |
| **Baseline Model MAE** | 3.29 V | $\pm 0.1\text{ V}$ |
| **Total Energy Output** | 7,153.93 kWh | $\pm 0.1\text{ kWh}$ |
| **Peak Power Output** | 80.96 kW | $\pm 0.01\text{ kW}$ |
| **Active Drive Sessions ($>10$ samples)** | 793 sessions | Exact match |
| **Negative Voltage Glitch Count** | 1,850 rows | Exact match |
| **Current vs Speed Correlation ($r$)** | 0.62 | Exact match |
| **Full Pipeline Execution Time** | ~6.0 seconds | $< 15.0\text{ seconds}$ |

---

## 4. Environment & Dependencies

```bash
python -m pip install pandas numpy matplotlib scipy scikit-learn
```

- **Python Version**: 3.10+ (Tested on Python 3.14.0)
- **Dependencies**: `pandas >= 2.0.0`, `numpy >= 1.24.0`, `matplotlib >= 3.7.0`, `scikit-learn >= 1.3.0`

---
*Audit Trail Document maintained by AI Data Processing Agent.*
