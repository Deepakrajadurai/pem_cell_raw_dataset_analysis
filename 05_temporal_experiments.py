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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. PyTorch Dataset for Intra-Session Sequences ---
class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len=15, scaler=None, is_train=True):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        
        # Prepare feature matrix and target
        X_raw = df[feature_cols].copy()
        
        if is_train and scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw.fillna(X_raw.median()))
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_raw.fillna(X_raw.median()))
            
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        df_scaled['session_id'] = df['session_id'].values
        df_scaled['soh_target'] = df['soh_target'].values
        df_scaled['is_active'] = df['is_active'].values
        
        # Create sequences per session
        for sess_id, s_df in df_scaled.groupby('session_id'):
            s_df = s_df.reset_index(drop=True)
            n_rows = len(s_df)
            if n_rows < seq_len:
                continue
            
            x_arr = s_df[feature_cols].values
            y_arr = s_df['soh_target'].values
            active_arr = s_df['is_active'].values
            
            for i in range(seq_len - 1, n_rows):
                if active_arr[i] and not np.isnan(y_arr[i]):
                    seq = x_arr[i - seq_len + 1 : i + 1]
                    self.sequences.append(seq)
                    self.targets.append(y_arr[i])
                    
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])

# --- 2. Neural Network Architectures ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)

# --- 3. Training Loop ---
def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_weights = None
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                loss = criterion(pred, y_b)
                val_loss += loss.item() * len(y_b)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            
    model.load_state_dict(best_weights)
    return model

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    preds = []
    actuals = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            pred = model(x_b)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y_b.numpy())
            
    y_t = np.array(actuals)
    y_p = np.array(preds)
    
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}, y_t, y_p

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    reports_dir = os.path.join(base_dir, "Reports")
    models_dir = os.path.join(base_dir, "Models", "temporal")
    os.makedirs(models_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Temporal Models Experiment on Device: {device}")

    print("Loading Parquet splits...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))

    # Feature Set Definitions
    feature_configs = {
        "Full Features": [
            'Stack Current Sensor Value A', 'Fuel Cell Total Voltage V', 'Stack Coolant Temp Inlet degree C',
            'Stack Coolant Temp Outlet degree C', 'Air Temp Stack Outlet degree C', 'Air Flow Sensor kg/h',
            'Air Comp Motor Temp degree C', 'Ambient Air Temp degree C', 'Ambient Air Pressure kPa',
            'Fahrzeuggeschwindigkeit km/h', 'Stack Power kW', 'Air_Stoichiometry_Lambda', 'Delta_V_Nernst',
            'E_cum_kWh', 'high_current_duration_h', 'Cumulative Time h'
        ],
        "Voltage-Blind": [
            'Stack Current Sensor Value A', 'Stack Coolant Temp Inlet degree C', 'Stack Coolant Temp Outlet degree C',
            'Air Temp Stack Outlet degree C', 'Air Flow Sensor kg/h', 'Air Comp Motor Temp degree C',
            'Ambient Air Temp degree C', 'Ambient Air Pressure kPa', 'Fahrzeuggeschwindigkeit km/h',
            'Stack Power kW', 'Air_Stoichiometry_Lambda', 'R_est_Ohm', 'E_cum_kWh', 'high_current_duration_h', 'Cumulative Time h'
        ],
        "Physics-Only": [
            'Stack Current Sensor Value A', 'Air_Stoichiometry_Lambda', 'Delta_V_Nernst', 'R_est_Ohm',
            'E_cum_kWh', 'high_current_duration_h'
        ]
    }

    model_types = ["LSTM", "GRU"]
    results = []

    for cfg_name, feature_cols in feature_configs.items():
        print(f"\n=======================================================")
        print(f"=== EXPERIMENT CONFIGURATION: {cfg_name} ({len(feature_cols)} features) ===")
        print(f"=======================================================")

        # Build PyTorch Datasets
        train_ds = SequenceDataset(train_df, feature_cols, seq_len=15, is_train=True)
        val_ds = SequenceDataset(val_df, feature_cols, seq_len=15, scaler=train_ds.scaler, is_train=False)
        test_ds = SequenceDataset(test_df, feature_cols, seq_len=15, scaler=train_ds.scaler, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        print(f"Sequence Datasets built -> Train seqs: {len(train_ds)}, Val seqs: {len(val_ds)}, Test seqs: {len(test_ds)}")

        for m_type in model_types:
            exp_name = f"{m_type}_{cfg_name.lower().replace(' ', '_').replace('-', '_')}"
            print(f"\n--> Training {m_type} [{cfg_name}]...")
            
            if m_type == "LSTM":
                model = LSTMRegressor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
            else:
                model = GRURegressor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)

            trained_model = train_model(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)
            
            # Save Model Checkpoint
            save_path = os.path.join(models_dir, f"{exp_name}.pt")
            torch.save(trained_model.state_dict(), save_path)

            # Evaluate
            test_m, y_true, y_pred = evaluate_model(trained_model, test_loader, device=device)
            val_m, _, _ = evaluate_model(trained_model, val_loader, device=device)

            print(f"    Results for {m_type} [{cfg_name}]: Test R2 = {test_m['R2']:.4f}, Test MAE = {test_m['MAE']:.4f}%, Test RMSE = {test_m['RMSE']:.4f}%")

            results.append({
                "Architecture": m_type,
                "Feature_Set": cfg_name,
                "Num_Features": len(feature_cols),
                "Val_MAE": val_m["MAE"], "Val_RMSE": val_m["RMSE"], "Val_R2": val_m["R2"], "Val_MAPE_%": val_m["MAPE_%"],
                "Test_MAE": test_m["MAE"], "Test_RMSE": test_m["RMSE"], "Test_R2": test_m["R2"], "Test_MAPE_%": test_m["MAPE_%"]
            })

    # Save Experiment Results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(reports_dir, "temporal_experiment_results.csv")
    results_df.to_csv(results_csv, index=False)

    print("\n\n=======================================================")
    print("=== TEMPORAL MODELS VS FEATURE CONFIGURATIONS MATRIX ===")
    print("=======================================================")
    print(results_df.to_string(index=False))

    # Compare with Baseline LightGBM ($R^2 = 0.724$, Test MAE = $0.814\%$)
    lgbm_row = {
        "Architecture": "LightGBM (Baseline)",
        "Feature_Set": "Full Features",
        "Num_Features": 16,
        "Val_MAE": 0.834896, "Val_RMSE": 1.908310, "Val_R2": 0.667467, "Val_MAPE_%": 0.884561,
        "Test_MAE": 0.813623, "Test_RMSE": 2.019290, "Test_R2": 0.724411, "Test_MAPE_%": 0.874896
    }
    
    comp_df = pd.concat([pd.DataFrame([lgbm_row]), results_df], ignore_index=True)
    comp_csv = os.path.join(reports_dir, "temporal_vs_baseline_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved temporal comparison matrix to: {comp_csv}")

    # Plot Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_df, x="Feature_Set", y="Test_R2", hue="Architecture")
    plt.title("Test R² Performance: Baseline LightGBM vs Temporal Models (LSTM / GRU)", fontsize=14, fontweight='bold')
    plt.xlabel("Feature Set Configuration", fontsize=12)
    plt.ylabel("Test R² Score (Higher is Better)", fontsize=12)
    plt.axhline(0.7244, color='red', linestyle='--', label='LightGBM Baseline R² = 0.7244')
    plt.ylim(-0.1, 0.85)
    plt.legend(loc='lower right', frameon=True)
    
    plot_png = os.path.join(reports_dir, "temporal_experiment_analysis.png")
    plt.savefig(plot_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal experiment plot to: {plot_png}")

if __name__ == "__main__":
    main()
