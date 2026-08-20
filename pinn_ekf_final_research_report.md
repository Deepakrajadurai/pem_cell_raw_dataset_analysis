# Physics-Informed Neural Network (PINN) + EKF Final Research Report

This document presents the final experimental evaluation comparing data-driven models against the **Physics-Informed Neural Network (PINN)** and **Extended Kalman Filter (EKF)** state observer for State of Health ($\text{SoH}$) estimation on the PEM Fuel Cell telemetry dataset ([`processed.csv`](file:///d:/PEM_Cell_Dataset/Processed/processed.csv)).

---

## 1. Master Model Comparison Matrix (Chronological Test Split)

All models were trained on `train.parquet` ($140,713\text{ active rows}$), tuned on `validation.parquet` ($34,125\text{ rows}$), and evaluated on `test.parquet` ($25,393\text{ rows}$, Sessions $690 \dots 811$) under strict chronological session splits:

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

## 2. Master Evaluation Visualization

![Master Model Comparison & PINN-EKF Diagnostics](file:///C:/Users/vijayakr/.gemini/antigravity-ide/brain/17419cdd-121e-45ff-aca9-e35a4225c491/pinn_ekf_analysis.png)

---

## 3. Scientific Analysis & Key Findings

> [!TIP]
> ### 1. LightGBM Remains the Superior Production Benchmark ($R^2 = 0.724$)
> **LightGBM** provides the most accurate and robust degradation predictor on this dataset ($R^2 = 0.724$, Test MAE $= 0.814\%$). Its non-linear decision trees capture multi-variable interactions (voltage decay ratios, cumulative energy throughput, lambda stoichiometry, and thermal trends) across sessions without assuming simplified analytical stack equations.

> [!IMPORTANT]
> ### 2. Scientific Finding: Limitations of Simplified Electrochemical PINNs ($R^2 = 0.097$)
> Standalone **PINN** trained on 6 physical inputs (`Current`, `Temp`, `Air Flow`, `Pressure`, `E_cum`, `Lambda`) achieves $R^2 = 0.097$ on the test split. 
> - **Voltage Reconstruction Sensitivity**: In real drive cycles, stack voltage varies rapidly ($\pm 50\text{ V}$) due to dynamic load spikes ($0 \rightarrow 374\text{ A}$), while degradation occurs gradually over thousands of operating hours.
> - An uncalibrated 1D thermodynamic Nernst equation model cannot fully isolate micro-level cell activation losses ($\eta_{\text{act}}$) and concentration overpotentials ($\eta_{\text{conc}}$) from dynamic current transients without cell-level temperature/humidity sensors.

> [!WARNING]
> ### 3. EKF Voltage Innovation Divergence ($R^2 = -3.798$)
> Coupling the Extended Kalman Filter to voltage innovation residuals $e_v = V_{\text{measured}} - \hat{V}_{\text{reconstructed}}$ resulted in state divergence ($R^2 = -3.798$, Test MAE $= 7.112\%$).
> - Because $V_{\text{stack}}$ RMSE error is $48.75\text{ V}$, applying direct linear Kalman gains $K \cdot e_v$ forces large unphysical updates to the internal degradation state $\hat{D}(t)$, causing severe over-correction during transient acceleration/deceleration maneuvers.

---

## 4. Software Deliverables Matrix

All 8 modular Python scripts are saved and executable in `d:\PEM_Cell_Dataset\`:

1. [`01_physics_model.py`](file:///d:/PEM_Cell_Dataset/01_physics_model.py): Nernst thermodynamics & voltage reconstruction equations.
2. [`02_pinn_model.py`](file:///d:/PEM_Cell_Dataset/02_pinn_model.py): PyTorch PINN neural encoder architecture.
3. [`03_physics_loss.py`](file:///d:/PEM_Cell_Dataset/03_physics_loss.py): Multi-objective physics loss function module.
4. [`04_train_pinn.py`](file:///d:/PEM_Cell_Dataset/04_train_pinn.py): Multi-objective training loop with checkpointing.
5. [`05_ekf.py`](file:///d:/PEM_Cell_Dataset/05_ekf.py): Extended Kalman Filter online state observer.
6. [`06_pinn_ekf.py`](file:///d:/PEM_Cell_Dataset/06_pinn_ekf.py): Coupled PINN + EKF evaluation pipeline.
7. [`07_evaluate_pinn.py`](file:///d:/PEM_Cell_Dataset/07_evaluate_pinn.py): Standalone and coupled PINN evaluation.
8. [`08_compare_models.py`](file:///d:/PEM_Cell_Dataset/08_compare_models.py): Master model comparison generator.

Data and report deliverables:
- `Models/pinn_ekf/pinn_model.pt` & `pinn_config.pkl`
- `Models/pinn_ekf/test_pinn_ekf_predictions.csv`
- `Reports/model_comparison_matrix.csv`
- `Reports/pinn_ekf_results.json`
- `Reports/pinn_ekf_analysis.png`
