import os
import json
import numpy as np
import pandas as pd

def compute_soh_target(df, bol_v_map, bol_scale_factor=1.034):
    df = df.copy()
    
    i_col = 'Stack Current Sensor Value A'
    v_col = 'Fuel Cell Total Voltage V'
    
    bins = [0, 10, 50, 100, 150, 200, 300]
    labels = ['(0, 10]', '(10, 50]', '(50, 100]', '(100, 150]', '(150, 200]', '(200, 300]']
    
    df['curr_bin'] = pd.cut(df[i_col], bins=bins, labels=labels)
    df['V_BOL'] = df['curr_bin'].map(bol_v_map).astype(float)
    
    # Environmental temperature/pressure compensation relative to STP
    T_k = df['Stack Coolant Temp Outlet degree C'] + 273.15
    P_kPa = df['Ambient Air Pressure kPa'].fillna(96.0)
    delta_v_env = 360.0 * (-0.00085 * (T_k - 298.15) + (4.31e-5 * T_k * np.log(P_kPa / 101.325)))
    
    v_adj = df[v_col] - delta_v_env
    
    # Normalized Voltage Performance Ratio relative to BOL baseline (scaled so BOL = 100.0%)
    raw_v_ratio = np.where(
        df['is_active'] & (df['V_BOL'] > 0),
        (v_adj / df['V_BOL']) / bol_scale_factor * 100.0,
        np.nan
    )
    df['soh_v_ratio_raw'] = raw_v_ratio
    
    # Apply intra-session 60s EMA filter (15 grid steps) to smooth high-frequency dynamic noise
    soh_filtered = []
    for sess_id, s_df in df.groupby('session_id'):
        filtered_s = s_df['soh_v_ratio_raw'].ewm(span=15, min_periods=1).mean()
        soh_filtered.append(filtered_s)
        
    df['soh_target'] = pd.concat(soh_filtered).values
    # Clip physically to [60.0, 100.0] for active states, set NaN for non-active states
    df['soh_target'] = np.where(df['is_active'], np.clip(df['soh_target'], 60.0, 100.0), np.nan)
    
    return df

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    reports_dir = os.path.join(base_dir, "Reports")
    features_p = os.path.join(base_dir, "Features", "pem_features.parquet")
    
    print("Reading dataset splits to construct reproducible SoH target...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))
    
    # 1. Establish Beginning-of-Life (BOL) Healthy Baseline from Early Sessions (session_id <= 50)
    bol_data = train_df[(train_df['session_id'] <= 50) & train_df['is_active']].copy()
    
    bins = [0, 10, 50, 100, 150, 200, 300]
    labels = ['(0, 10]', '(10, 50]', '(50, 100]', '(100, 150]', '(150, 200]', '(200, 300]']
    bol_data['curr_bin'] = pd.cut(bol_data['Stack Current Sensor Value A'], bins=bins, labels=labels)
    
    bol_v_map = bol_data.groupby('curr_bin', observed=False)['Fuel Cell Total Voltage V'].mean().to_dict()
    print("Established BOL Reference Voltages V_BOL(I) [Sessions 1-50]:")
    for k, v in bol_v_map.items():
        print(f"  Bin {k}: {v:.2f} V")
        
    # Calculate empirical BOL scaling factor so BOL early sessions mean SoH = 100.0%
    T_k_bol = bol_data['Stack Coolant Temp Outlet degree C'] + 273.15
    P_kPa_bol = bol_data['Ambient Air Pressure kPa'].fillna(96.0)
    delta_v_bol = 360.0 * (-0.00085 * (T_k_bol - 298.15) + (4.31e-5 * T_k_bol * np.log(P_kPa_bol / 101.325)))
    v_adj_bol = bol_data['Fuel Cell Total Voltage V'] - delta_v_bol
    v_bol_ref = bol_data['curr_bin'].map(bol_v_map).astype(float)
    bol_scale_factor = float((v_adj_bol / v_bol_ref).mean())
    print(f"Empirical BOL Normalization Factor: {bol_scale_factor:.4f}")

    # 2. Compute SoH Target for Train, Val, Test splits
    print("\nComputing SoH target index across Train, Val, and Test splits...")
    train_df = compute_soh_target(train_df, bol_v_map, bol_scale_factor)
    val_df = compute_soh_target(val_df, bol_v_map, bol_scale_factor)
    test_df = compute_soh_target(test_df, bol_v_map, bol_scale_factor)
    
    # 3. Save updated splits
    train_df.to_parquet(os.path.join(splits_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(splits_dir, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(splits_dir, "test.parquet"), index=False)
    print("Saved updated Parquet splits with `soh_target` column.")

    # Update full feature matrix
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_parquet(features_p, index=False)
    print("Saved updated full feature matrix to:", features_p)

    # 4. Generate Target Report
    active_train = train_df[train_df['is_active']]
    active_val = val_df[val_df['is_active']]
    active_test = test_df[test_df['is_active']]
    
    report_rows = [
        {"split": "Train", "active_rows": len(active_train), "soh_mean": active_train['soh_target'].mean(), "soh_std": active_train['soh_target'].std(), "soh_min": active_train['soh_target'].min(), "soh_max": active_train['soh_target'].max()},
        {"split": "Validation", "active_rows": len(active_val), "soh_mean": active_val['soh_target'].mean(), "soh_std": active_val['soh_target'].std(), "soh_min": active_val['soh_target'].min(), "soh_max": active_val['soh_target'].max()},
        {"split": "Test", "active_rows": len(active_test), "soh_mean": active_test['soh_target'].mean(), "soh_std": active_test['soh_target'].std(), "soh_min": active_test['soh_target'].min(), "soh_max": active_test['soh_target'].max()},
    ]
    
    rep_df = pd.DataFrame(report_rows)
    target_report_csv = os.path.join(reports_dir, "soh_target_report.csv")
    rep_df.to_csv(target_report_csv, index=False)
    print("\n--- SOH TARGET CONSTRUCTION REPORT ---")
    print(rep_df.to_string(index=False))
    print(f"\nSaved target report to: {target_report_csv}")

if __name__ == "__main__":
    main()
