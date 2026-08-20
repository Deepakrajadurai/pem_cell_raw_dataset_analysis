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
    models_dir = os.path.join(base_dir, "Models", "pinn_ekf")
    reports_dir = os.path.join(base_dir, "Reports")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating PINN & PINN+EKF models on Test Split...")

    config = joblib.load(os.path.join(models_dir, "pinn_config.pkl"))
    scaler = config["scaler"]
    feature_cols = config["features"]
    
    pinn_model = PEM_PINN_Encoder(input_dim=len(feature_cols), hidden_dim=64)
    pinn_model.load_state_dict(torch.load(os.path.join(models_dir, "pinn_model.pt")))
    pinn_model.to(device)
    
    physics_model = PEMPhysicsModel(n_cells=360)

    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))

    val_res = run_pinn_ekf_evaluation(val_df, pinn_model, physics_model, scaler, feature_cols, device=device)
    test_res = run_pinn_ekf_evaluation(test_df, pinn_model, physics_model, scaler, feature_cols, device=device)

    # Metrics
    val_pinn_m = calc_metrics(val_res['soh_target'].values, val_res['soh_pinn'].values)
    val_ekf_m = calc_metrics(val_res['soh_target'].values, val_res['soh_pinn_ekf'].values)

    test_pinn_m = calc_metrics(test_res['soh_target'].values, test_res['soh_pinn'].values)
    test_ekf_m = calc_metrics(test_res['soh_target'].values, test_res['soh_pinn_ekf'].values)

    v_recon_rmse = float(root_mean_squared_error(test_res['Fuel Cell Total Voltage V'], test_res['v_reconstructed']))

    summary = {
        "PINN_Validation": val_pinn_m,
        "PINN_EKF_Validation": val_ekf_m,
        "PINN_Test": test_pinn_m,
        "PINN_EKF_Test": test_ekf_m,
        "Voltage_Reconstruction_Test_RMSE_V": v_recon_rmse
    }

    json_path = os.path.join(reports_dir, "pinn_ekf_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n--- PINN & PINN+EKF EVALUATION SUMMARY ---")
    print(f"PINN Standalone Test R2: {test_pinn_m['R2']:.4f} | Test MAE: {test_pinn_m['MAE']:.4f}%")
    print(f"PINN + EKF Coupled Test R2: {test_ekf_m['R2']:.4f} | Test MAE: {test_ekf_m['MAE']:.4f}%")
    print(f"Voltage Reconstruction Test RMSE: {v_recon_rmse:.2f} V")
    print(f"\nSaved PINN evaluation summary to: {json_path}")

if __name__ == "__main__":
    main()
