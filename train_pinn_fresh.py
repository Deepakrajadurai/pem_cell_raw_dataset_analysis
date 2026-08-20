import os
import sys
sys.path.append(r'C:\Users\vijayakr\AppData\Roaming\Python\Python314\site-packages')

import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --- 1. Electrochemical Physics Equations ---
class PEMPhysicsModel:
    def __init__(self, n_cells=360, e0=1.229):
        self.n_cells = n_cells
        self.e0 = e0

    def compute_e_nernst_torch(self, T_celsius, P_kPa):
        T_c = torch.nan_to_num(T_celsius, nan=58.0)
        P_k = torch.nan_to_num(P_kPa, nan=96.0)
        T_k = torch.clamp(T_c + 273.15, min=263.15, max=393.15)
        P_k = torch.clamp(P_k, min=80.0, max=120.0)
        
        e_cell = self.e0 - 0.00085 * (T_k - 298.15) + (4.31e-5 * T_k * torch.log(P_k / 101.325))
        return self.n_cells * e_cell

    def reconstruct_v_stack_torch(self, I_stack, T_celsius, P_kPa, R_ohmic, eta_act):
        E_nernst = self.compute_e_nernst_torch(T_celsius, P_kPa)
        I_s = torch.nan_to_num(I_stack, nan=0.0)
        r_o = torch.nan_to_num(R_ohmic, nan=0.10)
        eta_a = torch.nan_to_num(eta_act, nan=5.0)
        
        V_ohmic = I_s * r_o
        V_hat = E_nernst - V_ohmic - eta_a
        return V_hat

# --- 2. PINN Neural Encoder Architecture ---
class PEM_PINN_Encoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU()
        )
        
        # Degradation state head D in [0.0, 0.35]
        self.degradation_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Activation Overpotential head eta_act in [1.0, 40.0] V
        self.eta_act_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.net(x)
        
        # Structural Degradation State D(t)
        degradation = 0.35 * self.degradation_head(feat).squeeze(-1)
        
        # Ohmic Resistance is explicitly coupled to structural degradation D(t)
        # R_ohmic(t) = R_BOL * (1 + 2.5 * D(t))
        r_ohmic = 0.080 * (1.0 + 2.5 * degradation)
        
        # Activation overpotential
        eta_act = 1.0 + 39.0 * self.eta_act_head(feat).squeeze(-1)
        
        # SoH Target = 100 * (1 - D)
        soh_pred = (1.0 - degradation) * 100.0
        
        return {
            "r_ohmic": r_ohmic,
            "eta_act": eta_act,
            "degradation": degradation,
            "soh_pred": soh_pred
        }

# --- 3. Multi-Objective Physics Loss ---
class MultiObjectivePINNLoss(nn.Module):
    def __init__(self, w_v=0.005, w_phys=1.0, w_mono=0.5, w_smooth=0.1, w_soh=1.0):
        super().__init__()
        self.w_v = w_v
        self.w_phys = w_phys
        self.w_mono = w_mono
        self.w_smooth = w_smooth
        self.w_soh = w_soh
        self.mse = nn.MSELoss()

    def forward(self, outputs, v_measured, soh_target, r_est_measured=None):
        r_ohmic = outputs["r_ohmic"]
        degradation = outputs["degradation"]
        soh_pred = outputs["soh_pred"]
        v_reconstructed = outputs["v_reconstructed"]
        
        l_v = self.mse(v_reconstructed, v_measured)
        
        if r_est_measured is not None:
            valid_r = ~torch.isnan(r_est_measured)
            if valid_r.sum() > 0:
                l_phys = self.mse(r_ohmic[valid_r], r_est_measured[valid_r])
            else:
                l_phys = torch.tensor(0.0, device=v_measured.device)
        else:
            l_phys = torch.tensor(0.0, device=v_measured.device)
            
        dD = degradation[1:] - degradation[:-1]
        l_mono = torch.mean(torch.relu(-dD))
        l_smooth = torch.mean(dD**2)
        l_soh = self.mse(soh_pred, soh_target)
        
        total_loss = (
            self.w_v * l_v + 
            self.w_phys * l_phys + 
            self.w_mono * l_mono + 
            self.w_smooth * l_smooth + 
            self.w_soh * l_soh
        )
        return total_loss

# --- 4. Dataset Class ---
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
    print(f"Training Clean Coupled PINN Model on Device: {device}")

    feature_cols = [
        'Stack Current Sensor Value A',
        'Stack Coolant Temp Outlet degree C',
        'Air Flow Sensor kg/h',
        'Ambient Air Pressure kPa',
        'E_cum_kWh',
        'Air_Stoichiometry_Lambda'
    ]

    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))

    train_ds = PINNDataset(train_df, feature_cols, is_train=True)
    val_ds = PINNDataset(val_df, feature_cols, scaler=train_ds.scaler, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

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
            
            loss = loss_fn(outputs, v_meas, soh_target, r_est)
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
                
                loss = loss_fn(outputs, v_meas, soh_target, r_est)
                val_loss += loss.item() * len(x_b)
                
        val_loss /= len(val_ds)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    model_pt_path = "d:/PEM_Cell_Dataset/Models/pinn_ekf/pinn_model.pt"
    config_pkl_path = "d:/PEM_Cell_Dataset/Models/pinn_ekf/pinn_config.pkl"
    
    if os.path.exists(model_pt_path):
        os.remove(model_pt_path)
    if os.path.exists(config_pkl_path):
        os.remove(config_pkl_path)
        
    torch.save(pinn.state_dict(), model_pt_path)
    joblib.dump({"scaler": train_ds.scaler, "features": feature_cols}, config_pkl_path)
    
    print(f"\nFresh PINN model state_dict successfully saved to: {model_pt_path}")
    print(f"Fresh PINN scaler config successfully saved to: {config_pkl_path}")

if __name__ == "__main__":
    main()
