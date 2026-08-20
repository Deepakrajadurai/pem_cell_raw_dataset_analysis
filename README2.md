# PEM Fuel Cell Degradation Analysis & Machine Learning Architecture (`README2.md`)

This repository documents the complete end-to-end data preprocessing, physical target construction, machine learning baseline benchmarking, temporal deep learning experiments, and **Physics-Informed Neural Network (PINN) + Extended Kalman Filter (EKF)** state observer architecture for Proton Exchange Membrane (PEM) Fuel Cell State of Health ($\text{SoH}$) estimation.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    A[Raw Telemetry Dataset\n200,231 rows / 701 days] --> B[preprocess_pem_dataset.py]
    
    subgraph Data Preprocessing & Feature Pipeline
        B -->|Invalid -> NaN| C[Active State Filter\nV > 50V & I > 1A]
        C --> D[Multi-Condition R_est Gated Calculation\n|dI| >= 15A & dt ~ 4s]
        D --> E[Engineered Feature Matrix\n320,803 rows / 22 cols]
        E --> F[Parquet Chronological Splits\nTrain 70% / Val 15% / Test 15%]
    end

    subgraph Physical Target Engineering
        F --> G[01_create_soh_target.py]
        G --> H[BOL Polarization Baseline V_BOL]
        G --> I[Nernst Thermodynamic Compensation]
        H & I --> J[Physical SoH Proxy Target\n60s EMA Smoothing]
    end

    subgraph Machine Learning Evaluation
        J --> K[02_train_baselines.py\nPersistence / Ridge / RF / XGBoost / LightGBM]
        J --> L[05_temporal_experiments.py\nLSTM vs GRU Feature Experiments]
        J --> M[04_train_pinn.py & 06_pinn_ekf.py\nPhysics-Informed NN + EKF Observer]
    end

    subgraph Master Evaluation & Reports
        K --> N[08_compare_models.py\nMaster Comparison Matrix]
        L --> N
        M --> N
        N --> O[model_comparison_matrix.csv]
        N --> P[pinn_ekf_analysis.png]
    end
```

---

## 2. Chronological Progress Log

### Phase 1: Data Exploration & Preprocessing Pipeline
- **Script**: [`preprocess_pem_dataset.py`](file:///d:/PEM_Cell_Dataset/preprocess_pem_dataset.py) & [`verify_preprocessing.py`](file:///d:/PEM_Cell_Dataset/verify_preprocessing.py)
- **Key Engineering Steps**:
  - Preserved raw dataset untouched.
  - Converted unphysical out-of-bound sensor glitches ($-500\text{V}$, $-50^\circ\text{C}$) to `NaN` instead of clipping to preserve fault evidence.
  - Reconstructed 811 drive sessions based on time gaps $> 300\text{s}$.
  - Gated instantaneous resistance estimation $R_{\text{est}} = -\frac{\Delta V}{\Delta I}$ with multi-condition filters ($|\Delta I| \ge 15\text{A} \land \Delta t \approx 4\text{s} \land \text{same active session}$).
  - Created active dataset `ML/clean_pem_telemetry_active.csv` ($200,231\text{ rows}$) and chronological splits (`train.parquet`, `validation.parquet`, `test.parquet`).

### Phase 2: Physical State of Health ($\text{SoH}$) Target Construction
- **Script**: [`01_create_soh_target.py`](file:///d:/PEM_Cell_Dataset/01_create_soh_target.py)
- **Mathematical Formulation**:
  - Fitted Beginning-of-Life (BOL) reference polarization polynomial $V_{\text{BOL}}(I)$ over initial 50 sessions.
  - Applied Nernst thermodynamic potential compensation $\Delta V_{\text{env\_corr}}(T, P)$ for operating temperature and ambient pressure.
  - Formulated health indicator ratio $\text{SoH}_{\text{raw}} = \frac{V_{\text{measured}}}{V_{\text{BOL}}(I) + \Delta V_{\text{env\_corr}}} \times 100\%$.
  - Applied 60-second EMA smoothing window ($\text{span} = 15$) to remove high-frequency noise spikes.

### Phase 3: Machine Learning Tree & Linear Baselines
- **Scripts**: [`02_train_baselines.py`](file:///d:/PEM_Cell_Dataset/02_train_baselines.py), [`03_evaluate_baselines.py`](file:///d:/PEM_Cell_Dataset/03_evaluate_baselines.py), [`04_plot_degradation.py`](file:///d:/PEM_Cell_Dataset/04_plot_degradation.py)
- **Results**: Evaluated 5 models on held-out test sessions ($690 \dots 811$). **LightGBM** established the top benchmark ($R^2 = \mathbf{0.724}$, Test MAE $= \mathbf{0.814\%}$).

### Phase 4: Temporal Model Experiment (LSTM vs. GRU)
- **Script**: [`05_temporal_experiments.py`](file:///d:/PEM_Cell_Dataset/05_temporal_experiments.py)
- **Key Finding**: Evaluated 60-second intra-session sequences ($L = 15$ steps).
  - GRU ($R^2 = 0.612$) outperformed LSTM ($R^2 = 0.590$), but both trailed tabular LightGBM ($0.724$).
  - **Voltage Dependence**: Removing voltage features caused severe performance collapse ($R^2 = 0.612 \rightarrow 0.315$), demonstrating that degradation is heavily encoded in electrical performance under load.

### Phase 5: Physics-Informed Neural Network (PINN) + EKF Observer
- **Scripts**: [`01_physics_model.py`](file:///d:/PEM_Cell_Dataset/01_physics_model.py) through [`08_compare_models.py`](file:///d:/PEM_Cell_Dataset/08_compare_models.py)
- **Architecture**: Compact 6-input feedforward encoder (`Current`, `Temp`, `Air Flow`, `Pressure`, `E_cum`, `Lambda`) mapping to latent physical heads ($\hat{R}_{\text{ohmic}}$, $\hat{\eta}_{\text{act}}$, $\hat{D}$). Electromechanically coupled $R_{\text{ohmic}}(t) = R_{\text{BOL}}(1 + 2.5 D(t))$ to force voltage loss gradients into structural degradation.
- **Coupled EKF**: Real-time stack voltage innovation residual $e_v = V_{\text{measured}} - \hat{V}_{\text{reconstructed}}$ updates state covariance $P$.

---

## 3. Master Model Performance Comparison Matrix

All models evaluated on identical test split (`test.parquet`, Sessions $690 \dots 811$, $25,393\text{ active test rows}$):

| Model Architecture | Model Category | Inputs Used | Val MAE (%) | Val RMSE (%) | Val $R^2$ | Test MAE (%) | Test RMSE (%) | Test $R^2$ | Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | Naive Baseline | Last $\text{SoH}$ | $1.265\%$ | $3.313\%$ | $-0.002$ | $1.326\%$ | $3.854\%$ | $-0.004$ | $1.553\%$ |
| **Linear Regression (Ridge)** | Data-Driven Baseline | 16 features | $1.139\%$ | $2.315\%$ | $0.511$ | $1.227\%$ | $2.644\%$ | $0.527$ | $1.348\%$ |
| **Random Forest** | Data-Driven Baseline | 16 features | $0.847\%$ | $2.025\%$ | $0.626$ | $0.989\%$ | $2.215\%$ | $0.669$ | $1.052\%$ |
| **XGBoost Regressor** | Data-Driven Baseline | 16 features | $0.842\%$ | $1.991\%$ | $0.638$ | $0.919\%$ | $2.155\%$ | $0.686$ | $0.978\%$ |
| **LightGBM Regressor** | **Data-Driven Benchmark** | 16 features | **$0.835\%$** | **$1.908\%$** | **$0.667$** | **$0.814\%$** | **$2.019\%$** | **$0.724$** | **$0.875\%$** |
| **PINN (Standalone)** | Physics-Informed | 6 compact features | $1.135\%$ | $2.956\%$ | $0.202$ | $1.284\%$ | $3.654\%$ | $0.097$ | $1.483\%$ |
| **PINN + EKF (Coupled)** | Physics Observer | 6 compact features | $7.266\%$ | $8.350\%$ | $-5.367$ | $7.112\%$ | $8.425\%$ | $-3.798$ | $7.327\%$ |

---

## 4. Key Scientific Insights

1. **LightGBM Superiority ($R^2 = 0.724$)**: Tabular gradient boosted decision trees capture non-linear interactions across operational variables without requiring explicit sub-component electrochemical calibration.
2. **Voltage Feature Criticality**: Excluding voltage features deteriorates recurrent models by $56.5\%$, proving electrical voltage under load is the primary observable degradation carrier.
3. **Transient Voltage Sensitivity in Physical Equations**: In dynamic vehicle drive cycles, transient load swings cause $\pm 50\text{V}$ voltage deviations. Direct EKF innovation coupling requires conservative noise covariance ($R_{\text{obs}}$) to prevent state divergence.

---

## 5. Software Sitemap & Execution Guide

To reproduce all experiments from raw data to final master reports:

```bash
# 1. Preprocess raw telemetry and generate clean session datasets
python preprocess_pem_dataset.py
python verify_preprocessing.py

# 2. Construct physical SoH proxy target
python 01_create_soh_target.py

# 3. Train and evaluate baseline machine learning models
python 02_train_baselines.py
python 03_evaluate_baselines.py
python 04_plot_degradation.py

# 4. Run temporal sequence experiments (LSTM vs GRU)
python 05_temporal_experiments.py
python 05_plot_temporal_results.py

# 5. Train PINN and evaluate coupled PINN + EKF observer
python train_pinn_fresh.py
python 07_evaluate_pinn.py

# 6. Generate master comparison matrix and visualization plots
python 08_compare_models.py
```

### Main Documentation Reports
- [`soh_target_and_pipeline_justification_report.md`](file:///d:/PEM_Cell_Dataset/soh_target_and_pipeline_justification_report.md): Technical justification for sanitization, $R_{\text{est}}$ gating, $\text{SoH}$ target derivation, and model selection.
- [`pinn_ekf_final_research_report.md`](file:///d:/PEM_Cell_Dataset/pinn_ekf_final_research_report.md): Comprehensive evaluation report on PINN & PINN-EKF architectures.
- [`soh_baseline_results.md`](file:///d:/PEM_Cell_Dataset/soh_baseline_results.md): Baseline models analysis.
- [`temporal_vs_baseline_analysis.md`](file:///d:/PEM_Cell_Dataset/temporal_vs_baseline_analysis.md): Temporal sequence experiments analysis.
