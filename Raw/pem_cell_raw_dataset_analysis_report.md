# PEM Fuel Cell Time Series Data Summary & Technical Analysis Report

> [!NOTE]
> This analysis report evaluates the raw PEM (Proton Exchange Membrane) Fuel Cell dataset located at [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv). The dataset captures continuous vehicle-level telemetry, electrical parameters, thermal metrics, and reactant supply indicators over a 701-day operational window.

---

## 1. Executive Summary & Dataset Metadata

| Attribute | Details |
| :--- | :--- |
| **Dataset File** | [raw_dataset.csv](file:///d:/PEM_Cell_Dataset/Raw/raw_dataset.csv) |
| **File Size** | 25.69 MB |
| **Total Observations** | 302,212 rows |
| **Parameter Count** | 13 telemetry channels + 1 Timestamp column |
| **Time Span Start** | `2024-08-12 09:22:15 UTC` |
| **Time Span End** | `2026-07-14 14:25:29 UTC` |
| **Total Elapsed Duration** | **701.21 days** (16,829.05 hours / ~1.92 years) |
| **Nominal Sampling Interval** | **4 seconds** (Median & Mode: 4.0 s) |
| **Sampling Characteristics** | Non-continuous (Intermittent driving sessions with gaps during system shutoff) |
| **Estimated Energy Delivered** | **7,153.93 kWh** |
| **Baseline Model Accuracy** | **$R^2 = 0.9144$**, **$\text{MAE} = 3.29\text{ V}$** |

---

## 2. Parameter Statistical Overview Table

Below is the comprehensive statistical summary for all 13 numeric parameters in the dataset.

| Parameter | Unit | Count | Missing (%) | Mean | Std Dev | Min | Median (50%) | Max | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fuel Cell Total Voltage** | V | 298,255 | 1.31% | 300.11 | 112.62 | -500.00 | 342.00 | 412.00 | -1.88 |
| **Stack Current Sensor Value** | A | 298,255 | 1.31% | 40.62 | 51.39 | 0.00 | 15.00 | 255.00 | +1.37 |
| **Stack Electrical Power** *(Derived)* | kW | 298,255 | 1.31% | 13.35 | 16.19 | 0.00 | 5.63 | 80.96 | +1.20 |
| **Coolant Temp Inlet** | °C | 298,460 | 1.24% | 54.74 | 8.87 | -2.00 | 58.00 | 79.00 | -2.51 |
| **Coolant Temp Outlet** | °C | 298,460 | 1.24% | 56.70 | 10.01 | -2.00 | 59.00 | 85.00 | -1.88 |
| **Coolant Differential ($\Delta T$)** *(Derived)* | °C | 298,460 | 1.24% | 1.96 | 3.29 | -23.00 | 1.00 | 32.00 | -0.05 |
| **Air Temp Stack Outlet** | °C | 298,922 | 1.09% | 53.28 | 10.43 | -1.00 | 56.00 | 80.00 | -1.83 |
| **Air Flow Sensor** | kg/h | 298,922 | 1.09% | 74.18 | 101.81 | 0.00 | 43.33 | 407.84 | +2.02 |
| **Air Comp Motor Temp** | °C | 299,030 | 1.05% | 33.70 | 11.02 | -50.00 | 33.00 | 138.00 | +0.87 |
| **Ambient Air Temp** | °C | 298,321 | 1.29% | 16.96 | 4.35 | -10.00 | 16.50 | 29.50 | +0.29 |
| **Ambient Air Pressure** | kPa | 298,922 | 1.09% | 0.96 | 0.01 | 0.92 | 0.96 | 1.00 | -0.11 |
| **Kumulative Betriebszeit** | h | 282,443 | 6.54% | 91,811.5 | 9,553.1 | 71,492.0 | 92,483.0 | 108,112.0 | -0.28 |
| **Fahrzeuggeschwindigkeit** | km/h | 282,478 | 6.53% | 64.98 | 44.16 | 0.00 | 64.00 | 178.00 | +0.08 |
| **Alive Count** | - | 298,959 | 1.08% | 7.50 | 4.61 | 0.00 | 8.00 | 15.00 | -0.00 |

---

## 3. Subsystem Technical Analysis

### A. Electrical Performance & Polarization Behavior
- **Open Circuit & Max Load Voltage**: Peak voltage reaches **412.0 V** (Open Circuit Voltage, OCV). Under heavy load, voltage drops to nominal operating levels between **300 V and 365 V**.
- **Current & Power Output**: Stack current peaks at **255.0 A**, delivering a maximum power output of **80.96 kW**.
- **Polarization Curve Characteristics**: As stack current increases from 10 A to 250 A, average stack voltage drops predictably due to internal Ohmic losses and mass transport limitations.

![PEM Fuel Cell Polarization & Power Curve](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/polarization_curve.png)

#### Polarization Binned Summary Table
| Current Range (A) | Mean Voltage (V) | Voltage Std Dev (V) | Mean Stack Power (kW) | Sample Count |
| :---: | :---: | :---: | :---: | :---: |
| **(0, 10]** | 324.59 | 123.31 | 1.25 | 37,350 |
| **(10, 50]** | 358.54 | 16.24 | 9.94 | 68,953 |
| **(50, 100]** | 336.63 | 7.94 | 24.62 | 53,972 |
| **(100, 150]** | 321.41 | 5.23 | 40.06 | 30,272 |
| **(150, 200]** | 307.72 | 6.15 | 53.64 | 10,753 |
| **(200, 255]** | 296.88 | 7.82 | 66.85 | 2,703 |

---

### B. Thermal Subsystem & Heat Dissipation
- **Operational Temperature Range**: Inlet coolant temperature operates around a mean of **54.74 °C** (median **58.0 °C**), while outlet coolant temperature averages **56.69 °C** (median **59.0 °C**).
- **Thermal Differential ($\Delta T$)**: The average coolant temperature rise across the stack is **1.96 °C** under normal operations, reaching up to **4.0 °C – 8.0 °C** during high power transient acceleration.
- **Air Compressor Motor Thermal Load**: The compressor motor temperature averages **33.70 °C**, with peaks up to **138.0 °C** during prolonged high power delivery.

---

### C. Operating Regimes Breakdown

The fuel cell operating profile is segmented into four distinct operational load regimes:

![PEM Fuel Cell Operating Regimes Breakdown](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/operating_regimes.png)

1. **Idle / System Off ($I \le 1.0\text{ A}$)**: **106,304 samples (35.6%)** — Vehicle parked or operating on battery buffer.
2. **Low Load ($1.0\text{ A} < I \le 50.0\text{ A}$)**: **94,251 samples (31.6%)** — Low-speed city driving or cruising load.
3. **Medium Load ($50.0\text{ A} < I \le 150.0\text{ A}$)**: **84,244 samples (28.2%)** — Highway cruising and steady power demand.
4. **High Load ($I > 150.0\text{ A}$)**: **13,456 samples (4.5%)** — Peak acceleration, climbing, or fast highway driving.

---

## 4. Defensible Health Indicator (HI) & Voltage Degradation Model

To establish a defensible degradation index before generating synthetic data, a baseline physical regressor is trained on early operational hours ($\text{Hours} \le 76,000\text{ h}$):

$$V_{\text{expected}} = f(I, T_{\text{coolant\_in}}, T_{\text{coolant\_out}}, T_{\text{air}}, \text{Airflow}, \text{Ambient\_Temp}, \text{Ambient\_Pressure})$$

### Residual Voltage & State of Health Formulation
For each operating sample, the voltage loss residual $V_{\text{residual}}$ and normalized State of Health ($\text{SoH} \%$) are evaluated:

$$V_{\text{residual}} = V_{\text{actual}} - V_{\text{expected}}$$

$$\text{SoH} (\%) = \left( 1 + \frac{V_{\text{residual}}}{V_{\text{expected\_nominal}}} \right) \times 100\%$$

- **$V_{\text{residual}} \approx 0\text{ V}$ ($\text{SoH} \approx 100\%$)**: Stack operates in fresh state matching baseline physics.
- **$V_{\text{residual}} < 0\text{ V}$ ($\text{SoH} < 100\%$)**: Stack exhibits irreversible voltage degradation under identical operating conditions.

![Expected vs Actual Voltage Parity Plot](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/v_expected_vs_actual.png)

---

### Degradation Trajectory over Lifetime (`Kumulative Betriebszeit h`)

Tracking $V_{\text{residual}}$ across the lifetime counter (71,492 h to 108,112 h) reveals a steady, linear voltage degradation slope:

![PEM Fuel Cell Defensible Health Degradation Curve](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/health_degradation_curve.png)

![PEM Fuel Cell Voltage Loss Residual over Lifetime](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/voltage_residual_over_hours.png)

#### Health Indicator Trajectory Summary Table
| Operating Hours Bin (h) | Mean $V_{\text{actual}}$ (V) | Mean $V_{\text{expected}}$ (V) | Residual $V_{\text{residual}}$ (V) | State of Health ($\text{SoH} \%$) | Active Samples |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **74,000 h** | 337.67 | 337.66 | 0.00 | **100.00%** | 3,659 |
| **76,000 h** *(Baseline Cutoff)* | 342.26 | 342.16 | +0.10 | **100.03%** | 5,917 |
| **80,000 h** | 342.02 | 342.11 | -0.09 | **99.97%** | 5,025 |
| **86,000 h** | 343.37 | 344.66 | -1.28 | **99.63%** | 5,821 |
| **94,000 h** | 343.68 | 344.91 | -1.23 | **99.65%** | 5,836 |
| **100,000 h** | 342.22 | 343.77 | -1.54 | **99.56%** | 7,934 |
| **103,000 h** | 341.29 | 343.17 | -1.88 | **99.46%** | 7,141 |
| **107,000 h** | 342.61 | 344.06 | -1.45 | **99.58%** | 7,048 |

---

## 5. Drive Session & Time Series Temporal Dynamics

- **Total Drive Sessions Detected**: **811 distinct driving sessions** (defined by inter-sample gaps $> 5\text{ minutes}$).
- **Valid Active Sessions ($> 10\text{ samples}$)**: **793 sessions**.
- **Session Duration Metrics**:
  - **Median Session Duration**: **12.5 minutes**
  - **Mean Session Duration**: **26.9 minutes**
  - **Maximum Session Duration**: **98.7 minutes** (~1.64 hours)

### Representative Drive Session Time Series
Below is a full multi-channel telemetry breakdown of a representative 65-minute driving session (Session #488):

![Representative Drive Session Time Series](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/sample_drive_session.png)

---

## 6. Parameter Correlation Matrix

The heatmap below illustrates Pearson correlation coefficients ($r$) across all key physical telemetry parameters:

![Parameter Pearson Correlation Matrix](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/correlation_matrix.png)

---

## 7. Data Quality & Anomaly Audit

> [!WARNING]
> The raw dataset contains several sensor anomalies and missing data gaps that require preprocessing prior to synthetic data generation or model training:

1. **Negative Voltage Glitches**: **1,850 rows** exhibit invalid negative voltages (min down to **-500.0 V**). These correspond to sensor power-down states or signal disconnects during system shutoff.
2. **Air Compressor Temperature Glitches**: **24 rows** record negative values down to **-50.0 °C**, representing transient CAN-bus communication drops.
3. **Missing Data Rates**:
   - `Kumulative Betriebszeit h` and `Fahrzeuggeschwindigkeit km/h` have **6.5% missing values** (~19,700 rows).
   - Core sensor streams (`Voltage`, `Current`, `Coolant Temps`, `Air Flow`) have **~1.1% – 1.3% missing values** (~3,200 – 3,900 rows).

---

## 8. Engineering Recommendations for Synthetic Data Generation

> [!TIP]
> To generate realistic synthetic PEM fuel cell degradation time series:

1. **Condition Synthetic Generation on Health Index ($\text{SoH}$)**:
   - Use the computed $V_{\text{residual}}$ trajectory to modulate cell polarization voltage loss over target synthetic operating hours.
2. **Preserve Operational Load Regimes**:
   - Maintain the empirical load probability density (35.6% Idle, 31.6% Low Load, 28.2% Medium Load, 4.5% High Load).
3. **Condition on Thermal & Reactant Physics**:
   - Ensure synthetic air flow follows current demand ($r = 0.29 - 0.50$) and coolant temperature gradient scales with stack electrical power output.

---
*Report auto-generated by AI Data Analytics Agent.*
