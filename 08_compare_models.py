import os
import sys
import importlib
sys.path.append(r'C:\Users\vijayakr\AppData\Roaming\Python\Python314\site-packages')
sys.path.append(r'd:\PEM_Cell_Dataset')

import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

phys_mod = importlib.import_module("01_physics_model")
importlib.reload(phys_mod)

pinn_mod = importlib.import_module("02_pinn_model")
importlib.reload(pinn_mod)

pinn_ekf_mod = importlib.import_module("06_pinn_ekf")
importlib.reload(pinn_ekf_mod)

PEMPhysicsModel = phys_mod.PEMPhysicsModel
PEM_PINN_Encoder = pinn_mod.PEM_PINN_Encoder
run_pinn_ekf_evaluation = pinn_ekf_mod.run_pinn_ekf_evaluation

def calc_metrics(y_true, y_pred):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    
    valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]
    
    if len(y_t) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "MAPE_%": 0.0}
        
    return {
        "MAE": float(mean_absolute_error(y_t, y_p)),
        "RMSE": float(root_mean_squared_error(y_t, y_p)),
        "R2": float(r2_score(y_t, y_p)),
        "MAPE_%": float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    }

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    models_base = os.path.join(base_dir, "Models", "baseline")
    models_pinn = os.path.join(base_dir, "Models", "pinn_ekf")
    reports_dir = os.path.join(base_dir, "Reports")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading test dataset for master model comparison...")

    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))

    val_active = val_df[val_df['is_active'] & val_df['soh_target'].notna()].copy().reset_index(drop=True)
    test_active = test_df[test_df['is_active'] & test_df['soh_target'].notna()].copy().reset_index(drop=True)

    # 1. Evaluate Tree & Linear Baselines
    config_base = joblib.load(os.path.join(models_base, "feature_config.pkl"))
    base_features = config_base["features"]
    base_medians = config_base["medians"]

    X_val_base = val_active[base_features].fillna(base_medians)
    X_test_base = test_active[base_features].fillna(base_medians)
    y_val = val_active['soh_target'].values
    y_test = test_active['soh_target'].values

    base_models = {
        "Persistence": joblib.load(os.path.join(models_base, "persistence", "model.pkl")),
        "Linear Regression": joblib.load(os.path.join(models_base, "linear_regression", "model.pkl")),
        "Random Forest": joblib.load(os.path.join(models_base, "random_forest", "model.pkl")),
        "XGBoost": joblib.load(os.path.join(models_base, "xgboost", "model.pkl")),
        "LightGBM": joblib.load(os.path.join(models_base, "lightgbm", "model.pkl")),
    }

    comparison_rows = []

    for name, model in base_models.items():
        if name == "Persistence":
            mean_val = model["train_mean_soh"]
            p_val = np.full(len(y_val), mean_val)
            p_test = np.full(len(y_test), mean_val)
        else:
            p_val = model.predict(X_val_base)
            p_test = model.predict(X_test_base)

        vm = calc_metrics(y_val, p_val)
        tm = calc_metrics(y_test, p_test)

        comparison_rows.append({
            "Model_Architecture": name,
            "Category": "Data-Driven Baseline",
            "Val_MAE_%": vm["MAE"], "Val_RMSE_%": vm["RMSE"], "Val_R2": vm["R2"],
            "Test_MAE_%": tm["MAE"], "Test_RMSE_%": tm["RMSE"], "Test_R2": tm["R2"], "Test_MAPE_%": tm["MAPE_%"]
        })

    # 2. Evaluate PINN & PINN+EKF
    config_pinn = joblib.load(os.path.join(models_pinn, "pinn_config.pkl"))
    scaler = config_pinn["scaler"]
    pinn_features = config_pinn["features"]

    pinn_model = PEM_PINN_Encoder(input_dim=len(pinn_features), hidden_dim=64)
    pinn_model.load_state_dict(torch.load(os.path.join(models_pinn, "pinn_model.pt")))
    pinn_model.to(device)
    physics_model = PEMPhysicsModel(n_cells=360)

    val_res = run_pinn_ekf_evaluation(val_df, pinn_model, physics_model, scaler, pinn_features, device=device)
    test_res = run_pinn_ekf_evaluation(test_df, pinn_model, physics_model, scaler, pinn_features, device=device)

    pinn_val_m = calc_metrics(val_res['soh_target'].values, val_res['soh_pinn'].values)
    pinn_test_m = calc_metrics(test_res['soh_target'].values, test_res['soh_pinn'].values)

    ekf_val_m = calc_metrics(val_res['soh_target'].values, val_res['soh_pinn_ekf'].values)
    ekf_test_m = calc_metrics(test_res['soh_target'].values, test_res['soh_pinn_ekf'].values)

    comparison_rows.append({
        "Model_Architecture": "PINN (Standalone)",
        "Category": "Physics-Informed Neural Network",
        "Val_MAE_%": pinn_val_m["MAE"], "Val_RMSE_%": pinn_val_m["RMSE"], "Val_R2": pinn_val_m["R2"],
        "Test_MAE_%": pinn_test_m["MAE"], "Test_RMSE_%": pinn_test_m["RMSE"], "Test_R2": pinn_test_m["R2"], "Test_MAPE_%": pinn_test_m["MAPE_%"]
    })

    comparison_rows.append({
        "Model_Architecture": "PINN + EKF (Coupled)",
        "Category": "Physics-Informed State Observer",
        "Val_MAE_%": ekf_val_m["MAE"], "Val_RMSE_%": ekf_val_m["RMSE"], "Val_R2": ekf_val_m["R2"],
        "Test_MAE_%": ekf_test_m["MAE"], "Test_RMSE_%": ekf_test_m["RMSE"], "Test_R2": ekf_test_m["R2"], "Test_MAPE_%": ekf_test_m["MAPE_%"]
    })

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = os.path.join(reports_dir, "model_comparison_matrix.csv")
    comp_df.to_csv(comp_csv, index=False)

    print("\n=======================================================")
    print("=== MASTER MODEL COMPARISON MATRIX (TEST SPLIT) ===")
    print("=======================================================")
    print(comp_df.to_string(index=False))

    # Generate Visualization Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Test R2 Comparison across all models
    ax1 = axes[0]
    palette = ["#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#2ca02c", "#17becf", "#d62728"]
    sns.barplot(data=comp_df, x="Model_Architecture", y="Test_R2", hue="Model_Architecture", ax=ax1, palette=palette, legend=False)
    ax1.set_title("Master Test R² Comparison (Higher is Better)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Model Architecture", fontsize=11)
    ax1.set_ylabel("Test R² Score", fontsize=11)
    ax1.tick_params(axis='x', rotation=30)
    ax1.set_ylim(-0.1, 0.85)

    # Plot 2: Test MAE Comparison
    ax2 = axes[1]
    sns.barplot(data=comp_df, x="Model_Architecture", y="Test_MAE_%", hue="Model_Architecture", ax=ax2, palette=palette, legend=False)
    ax2.set_title("Master Test MAE Error % Comparison (Lower is Better)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Model Architecture", fontsize=11)
    ax2.set_ylabel("Test MAE Error (%)", fontsize=11)
    ax2.tick_params(axis='x', rotation=30)
    ax2.set_ylim(0.0, 2.0)

    plt.tight_layout()
    output_png = os.path.join(reports_dir, "pinn_ekf_analysis.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    # Copy to artifact directory
    artifact_png = "C:/Users/vijayakr/.gemini/antigravity-ide/brain/17419cdd-121e-45ff-aca9-e35a4225c491/pinn_ekf_analysis.png"
    import shutil
    shutil.copy(output_png, artifact_png)

    print(f"\nSaved master model comparison matrix to: {comp_csv}")
    print(f"Saved master comparison plot to: {output_png}")
    print(f"Saved master plot to artifact directory: {artifact_png}")

if __name__ == "__main__":
    main()
