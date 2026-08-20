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

phys_mod = importlib.import_module("01_physics_model")
pinn_mod = importlib.import_module("02_pinn_model")
ekf_mod = importlib.import_module("05_ekf")

PEMPhysicsModel = phys_mod.PEMPhysicsModel
PEM_PINN_Encoder = pinn_mod.PEM_PINN_Encoder
ExtendedKalmanFilter = ekf_mod.ExtendedKalmanFilter

def run_pinn_ekf_evaluation(df, pinn_model, physics_model, scaler, feature_cols, device='cpu'):
    pinn_model.eval()
    
    # Active state filter
    v_act = df['Fuel Cell Total Voltage V'] > 50.0
    i_act = df['Stack Current Sensor Value A'] > 1.0
    soh_valid = df['soh_target'].notna()
    
    active_df = df[v_act & i_act & soh_valid].copy().reset_index(drop=True)
    
    X_raw = active_df[feature_cols].copy()
    X_scaled = scaler.transform(X_raw.fillna(X_raw.median()))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    v_meas = active_df['Fuel Cell Total Voltage V'].values
    i_stack = active_df['Stack Current Sensor Value A'].values
    t_cool = active_df['Stack Coolant Temp Outlet degree C'].fillna(58.0).values
    p_amb = active_df['Ambient Air Pressure kPa'].fillna(96.0).values
    soh_target = active_df['soh_target'].values
    
    with torch.no_grad():
        outputs = pinn_model(X_tensor)
        
        i_tensor = torch.tensor(i_stack, dtype=torch.float32).to(device)
        t_tensor = torch.tensor(t_cool, dtype=torch.float32).to(device)
        p_tensor = torch.tensor(p_amb, dtype=torch.float32).to(device)
        
        v_reconstructed = physics_model.reconstruct_v_stack_torch(
            i_tensor, t_tensor, p_tensor, outputs["r_ohmic"], outputs["eta_act"]
        ).cpu().numpy()
        
    soh_pinn = outputs["soh_pred"].cpu().numpy()
    r_ohmic_pinn = outputs["r_ohmic"].cpu().numpy()
    eta_act_pinn = outputs["eta_act"].cpu().numpy()
    
    # Run EKF state observer online sequence
    ekf = ExtendedKalmanFilter(q_cov=1e-4, r_obs=4.0)
    soh_ekf = []
    
    for sess_id, s_group in active_df.groupby('session_id'):
        idxs = s_group.index.values
        ekf.reset(init_soh=soh_pinn[idxs[0]], init_r=r_ohmic_pinn[idxs[0]], init_eta=eta_act_pinn[idxs[0]])
        
        for idx in idxs:
            res = ekf.step(
                v_meas[idx],
                v_reconstructed[idx],
                soh_pinn[idx],
                r_ohmic_pinn[idx],
                eta_act_pinn[idx]
            )
            soh_ekf.append(res["ekf_soh"])
            
    active_df['soh_pinn'] = soh_pinn
    active_df['soh_pinn_ekf'] = np.array(soh_ekf)
    active_df['v_reconstructed'] = v_reconstructed
    
    return active_df

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    models_dir = os.path.join(base_dir, "Models", "pinn_ekf")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Coupled PINN + EKF Evaluation on Device: {device}")

    # Load Model & Config
    config = joblib.load(os.path.join(models_dir, "pinn_config.pkl"))
    scaler = config["scaler"]
    feature_cols = config["features"]
    
    pinn_model = PEM_PINN_Encoder(input_dim=len(feature_cols), hidden_dim=64)
    pinn_model.load_state_dict(torch.load(os.path.join(models_dir, "pinn_model.pt")))
    pinn_model.to(device)
    
    physics_model = PEMPhysicsModel(n_cells=360)
    
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))
    results_df = run_pinn_ekf_evaluation(test_df, pinn_model, physics_model, scaler, feature_cols, device=device)
    
    print(f"Coupled PINN + EKF evaluated on Test Set ({len(results_df)} active test rows).")
    results_csv = os.path.join(models_dir, "test_pinn_ekf_predictions.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved coupled predictions to: {results_csv}")

if __name__ == "__main__":
    main()
