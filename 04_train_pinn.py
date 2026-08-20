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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

phys_mod = importlib.import_module("01_physics_model")
importlib.reload(phys_mod)

pinn_mod = importlib.import_module("02_pinn_model")
importlib.reload(pinn_mod)

loss_mod = importlib.import_module("03_physics_loss")
importlib.reload(loss_mod)

PEMPhysicsModel = phys_mod.PEMPhysicsModel
PEM_PINN_Encoder = pinn_mod.PEM_PINN_Encoder
MultiObjectivePINNLoss = loss_mod.MultiObjectivePINNLoss

class PINNDataset(Dataset):
    def __init__(self, df, feature_cols, scaler=None, is_train=True):
        active_df = df[df['is_active'] & df['soh_target'].notna()].copy().reset_index(drop=True)
        X_raw = active_df[feature_cols].copy()
        
        if is_train and scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw.fillna(X_raw.median()))
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_raw.fillna(X_raw.median()))
            
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.v_meas = torch.tensor(active_df['Fuel Cell Total Voltage V'].values, dtype=torch.float32)
        self.i_stack = torch.tensor(active_df['Stack Current Sensor Value A'].values, dtype=torch.float32)
        self.t_cool = torch.tensor(active_df['Stack Coolant Temp Outlet degree C'].fillna(58.0).values, dtype=torch.float32)
        self.p_amb = torch.tensor(active_df['Ambient Air Pressure kPa'].fillna(96.0).values, dtype=torch.float32)
        self.soh_target = torch.tensor(active_df['soh_target'].values, dtype=torch.float32)
        
        r_est_vals = active_df['R_est_Ohm'].values
        self.r_est = torch.tensor(np.where(np.isnan(r_est_vals), np.nan, r_est_vals), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "v_meas": self.v_meas[idx],
            "i_stack": self.i_stack[idx],
            "t_cool": self.t_cool[idx],
            "p_amb": self.p_amb[idx],
            "soh_target": self.soh_target[idx],
            "r_est": self.r_est[idx]
        }

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = "d:/PEM_Cell_Dataset/Splits"
    models_dir = "d:/PEM_Cell_Dataset/Models/pinn_ekf"
    os.makedirs(models_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training PINN Model on Device: {device}")

    feature_cols = [
        'Stack Current Sensor Value A',
        'Stack Coolant Temp Outlet degree C',
        'Air Flow Sensor kg/h',
        'Ambient Air Pressure kPa',
        'E_cum_kWh',
        'Air_Stoichiometry_Lambda'
    ]

    print("Loading Train and Validation splits...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))

    train_ds = PINNDataset(train_df, feature_cols, is_train=True)
    val_ds = PINNDataset(val_df, feature_cols, scaler=train_ds.scaler, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    print(f"PINN Dataset Size -> Train: {len(train_ds)} active rows, Val: {len(val_ds)} active rows")

    pinn = PEM_PINN_Encoder(input_dim=len(feature_cols), hidden_dim=64).to(device)
    physics_model = PEMPhysicsModel(n_cells=360)
    loss_fn = MultiObjectivePINNLoss()
    
    optimizer = torch.optim.Adam(pinn.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 15

    print("\nStarting PINN Multi-Objective Training Loop...")
    for epoch in range(1, epochs + 1):
        pinn.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x_b = batch["x"].to(device)
            v_meas = batch["v_meas"].to(device)
            i_stack = batch["i_stack"].to(device)
            t_cool = batch["t_cool"].to(device)
            p_amb = batch["p_amb"].to(device)
            soh_target = batch["soh_target"].to(device)
            r_est = batch["r_est"].to(device)
            
            optimizer.zero_grad()
            outputs = pinn(x_b)
            
            v_reconstructed = physics_model.reconstruct_v_stack_torch(
                i_stack, t_cool, p_amb, outputs["r_ohmic"], outputs["eta_act"]
            )
            outputs["v_reconstructed"] = v_reconstructed
            
            loss, _ = loss_fn(outputs, v_meas, soh_target, r_est)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(x_b)
            
        train_loss /= len(train_ds)
        
        pinn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_b = batch["x"].to(device)
                v_meas = batch["v_meas"].to(device)
                i_stack = batch["i_stack"].to(device)
                t_cool = batch["t_cool"].to(device)
                p_amb = batch["p_amb"].to(device)
                soh_target = batch["soh_target"].to(device)
                r_est = batch["r_est"].to(device)
                
                outputs = pinn(x_b)
                v_reconstructed = physics_model.reconstruct_v_stack_torch(
                    i_stack, t_cool, p_amb, outputs["r_ohmic"], outputs["eta_act"]
                )
                outputs["v_reconstructed"] = v_reconstructed
                
                loss, _ = loss_fn(outputs, v_meas, soh_target, r_est)
                val_loss += loss.item() * len(x_b)
                
        val_loss /= len(val_ds)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Remove existing pinn_model.pt if it exists and write fresh state_dict
    model_pt_path = "d:/PEM_Cell_Dataset/Models/pinn_ekf/pinn_model.pt"
    config_pkl_path = "d:/PEM_Cell_Dataset/Models/pinn_ekf/pinn_config.pkl"
    
    if os.path.exists(model_pt_path):
        os.remove(model_pt_path)
    if os.path.exists(config_pkl_path):
        os.remove(config_pkl_path)
        
    torch.save(pinn.state_dict(), model_pt_path)
    joblib.dump({"scaler": train_ds.scaler, "features": feature_cols}, config_pkl_path)
    
    print(f"\nSuccessfully saved trained PINN model weights to: {model_pt_path}")
    print(f"Successfully saved PINN scaler config to: {config_pkl_path}")

if __name__ == "__main__":
    main()
