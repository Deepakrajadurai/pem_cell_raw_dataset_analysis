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

## 4. Drive Session & Time Series Temporal Dynamics

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

## 5. Parameter Correlation Matrix

The heatmap below illustrates Pearson correlation coefficients ($r$) across all key physical telemetry parameters:

![Parameter Pearson Correlation Matrix](C:/Users/vijayakr/.gemini/antigravity-ide/brain/59a3a9d7-4e23-4a0b-ba43-c6464ce27fe3/correlation_matrix.png)

### Key Correlation Insights
- **Current vs. Vehicle Speed ($r = 0.62$)**: Strong positive coupling reflecting direct power demand driven by vehicle dynamics.
- **Current vs. Coolant Outlet Temp ($r = 0.39$)**: Ohmic and electrochemical heat generation elevates coolant temperature with current.
- **Air Flow vs. Vehicle Speed ($r = 0.50$)**: Air mass flow scales proportionally to supply oxygen stoichiometry for power demand.
- **Coolant Inlet vs. Outlet Temp ($r = 0.95$)**: High thermal coupling across the stack cooling jacket.

---

## 6. Data Quality & Anomaly Audit

> [!WARNING]
> The raw dataset contains several sensor anomalies and missing data gaps that require preprocessing prior to model training or degradation analysis:

1. **Negative Voltage Glitches**: **1,850 rows** exhibit invalid negative voltages (min down to **-500.0 V**). These correspond to sensor power-down states or signal disconnects during system shutoff.
2. **Air Compressor Temperature Glitches**: **24 rows** record negative values down to **-50.0 °C**, representing transient CAN-bus communication drops.
3. **Missing Data Rates**:
   - `Kumulative Betriebszeit h` and `Fahrzeuggeschwindigkeit km/h` have **6.5% missing values** (~19,700 rows).
   - Core sensor streams (`Voltage`, `Current`, `Coolant Temps`, `Air Flow`) have **~1.1% – 1.3% missing values** (~3,200 – 3,900 rows).
4. **Time Continuity**: Gaps between sessions range up to **96.9 days**, confirming the file represents a multi-year collection of intermittent drive cycles rather than a single continuous run.

---

## 7. Engineering Recommendations for Modeling & Preprocessing

> [!TIP]
> To utilize this dataset for Remaining Useful Life (RUL) estimation, State of Health (SoH) monitoring, or digital twin development:

1. **Filtering Invalid Data**:
   - Filter rows where `Fuel Cell Total Voltage V <= 0` or `Stack Current Sensor Value A < 0`.
   - Remove temperature outliers outside physical operating bounds (e.g. $T < -10\text{ °C}$ or $T > 120\text{ °C}$).
2. **Session Segmentation**:
   - Treat each drive cycle (separated by $> 5\text{ min}$ gap) as an independent sequence time series.
   - Do not compute rolling derivatives (e.g. $dI/dt$, $dT/dt$) across session boundaries.
3. **Feature Engineering**:
   - **Internal Resistance ($R_{est}$)**: Compute $R_{est} = \frac{\Delta V}{\Delta I}$ during load transients.
   - **Stoichiometry Ratio**: Ratio of `Air Flow Sensor kg/h` to `Stack Current Sensor Value A`.
   - **Cumulative Energy Throughput**: Integrate electrical power over operational hours for degradation modeling.

---
*Report auto-generated by AI Data Analytics Agent.*
