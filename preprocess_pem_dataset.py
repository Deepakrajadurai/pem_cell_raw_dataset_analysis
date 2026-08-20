import os
import json
import numpy as np
import pandas as pd

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    
    # 1. Create Target Directory Structure
    dirs = {
        "raw": os.path.join(base_dir, "Raw"),
        "processed": os.path.join(base_dir, "Processed"),
        "ml": os.path.join(base_dir, "ML"),
        "features": os.path.join(base_dir, "Features"),
        "splits": os.path.join(base_dir, "Splits"),
        "models_base": os.path.join(base_dir, "Models", "baseline"),
        "models_soh": os.path.join(base_dir, "Models", "soh"),
        "models_rul": os.path.join(base_dir, "Models", "rul"),
        "reports": os.path.join(base_dir, "Reports")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    source_csv = os.path.join(dirs["processed"], "processed.csv")
    print(f"Reading source dataset (read-only): {source_csv}")
    
    df = pd.read_csv(source_csv)
    total_raw_rows = len(df)
    print(f"Total raw rows loaded: {total_raw_rows}")
    
    # Convert Time to UTC datetime
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    # 2. Reconstruct Sessions (gap > 300 s -> new session)
    dt_sec = df['Time'].diff().dt.total_seconds()
    session_starts = (dt_sec > 300) | (dt_sec.isna())
    df['session_id'] = session_starts.cumsum()
    num_sessions = df['session_id'].nunique()
    print(f"Reconstructed sessions (gap > 300s): {num_sessions}")
    
    # 3. Sensor Sanitization: Invalid -> NaN
    invalid_counts = {}
    
    # Voltage: V < 0 or V > 450 V -> NaN
    v_col = 'Fuel Cell Total Voltage V'
    invalid_v = (df[v_col] < 0.0) | (df[v_col] > 450.0)
    invalid_counts['voltage_invalid'] = int(invalid_v.sum())
    df.loc[invalid_v, v_col] = np.nan
    
    # Compressor Temp: T_comp < -10 or T_comp > 120 -> NaN
    t_comp_col = 'Air Comp Motor Temp degree C'
    invalid_tcomp = (df[t_comp_col] < -10.0) | (df[t_comp_col] > 120.0)
    invalid_counts['t_comp_invalid'] = int(invalid_tcomp.sum())
    df.loc[invalid_tcomp, t_comp_col] = np.nan
    
    # Ambient Pressure: P_amb < 80 or P_amb > 120 kPa -> NaN
    p_amb_col = 'Ambient Air Pressure kPa'
    invalid_pamb = (df[p_amb_col] < 80.0) | (df[p_amb_col] > 120.0)
    invalid_counts['p_amb_invalid'] = int(invalid_pamb.sum())
    df.loc[invalid_pamb, p_amb_col] = np.nan

    print("Invalid-to-NaN sanitization counts:", invalid_counts)

    # 4. Intra-Session 4s Uniform Resampling & Processing per Session
    sensor_cols = [
        'Fuel Cell Total Voltage V',
        'Stack Current Sensor Value A',
        'Stack Coolant Temp Inlet degree C',
        'Stack Coolant Temp Outlet degree C',
        'Air Temp Stack Outlet degree C',
        'Air Flow Sensor kg/h',
        'Air Comp Motor Temp degree C',
        'Ambient Air Temp degree C',
        'Ambient Air Pressure kPa',
        'Kumulative Betriebszeit h',
        'Fahrzeuggeschwindigkeit km/h'
    ]
    
    resampled_sessions = []
    session_summaries = []
    
    for sess_id, s_df in df.groupby('session_id'):
        s_df = s_df.sort_values('Time').reset_index(drop=True)
        start_t = s_df['Time'].min()
        end_t = s_df['Time'].max()
        
        # Calculate raw session metrics
        s_duration_min = (end_t - start_t).total_seconds() / 60.0
        active_raw = (s_df['Stack Current Sensor Value A'] > 1.0) & (s_df['Fuel Cell Total Voltage V'] > 50.0)
        
        summary = {
            'session_id': int(sess_id),
            'start_time': start_t.isoformat(),
            'end_time': end_t.isoformat(),
            'duration_minutes': round(s_duration_min, 2),
            'total_raw_rows': len(s_df),
            'active_raw_rows': int(active_raw.sum()),
            'max_current_A': float(s_df['Stack Current Sensor Value A'].max()) if len(s_df) > 0 else 0.0,
            'max_voltage_V': float(s_df['Fuel Cell Total Voltage V'].max()) if len(s_df) > 0 else 0.0
        }
        session_summaries.append(summary)
        
        if len(s_df) < 2 or s_duration_min < 0.1:
            # Single-row session, retain as-is without resampling grid
            s_df['is_interpolated'] = False
            s_df['gap_imputed'] = False
            resampled_sessions.append(s_df)
            continue
            
        # Create regular 4s grid inside session
        grid_times = pd.date_range(start=start_t, end=end_t, freq='4s')
        
        # Merge on time grid
        grid_df = pd.DataFrame({'Time': grid_times})
        merged = pd.merge_asof(grid_df, s_df, on='Time', tolerance=pd.Timedelta('2s'), direction='nearest')
        
        merged['session_id'] = sess_id
        
        # Interpolate sensor columns within session
        for col in sensor_cols:
            if col in s_df.columns:
                merged[col] = merged[col].interpolate(method='linear', limit=5)
                
        merged['is_interpolated'] = merged['Fuel Cell Total Voltage V'].isna()
        merged['gap_imputed'] = False
        
        resampled_sessions.append(merged)
        
    full_resampled = pd.concat(resampled_sessions, ignore_index=True)
    print(f"Total resampled rows across all sessions: {len(full_resampled)}")

    # 5. Operating State Separation
    v_active = full_resampled['Fuel Cell Total Voltage V'] > 50.0
    i_active = full_resampled['Stack Current Sensor Value A'] > 1.0
    valid_vi = full_resampled['Fuel Cell Total Voltage V'].notna() & full_resampled['Stack Current Sensor Value A'].notna()
    
    full_resampled['is_active'] = v_active & i_active & valid_vi

    # 6. Multi-Condition R_est Gating & Physics Feature Engineering
    # Calculate Power (kW)
    full_resampled['Stack Power kW'] = np.where(
        full_resampled['is_active'],
        (full_resampled['Fuel Cell Total Voltage V'] * full_resampled['Stack Current Sensor Value A']) / 1000.0,
        0.0
    )
    
    # Calculate R_est with Multi-Condition Gating
    # Quality conditions:
    # 1. |dI| >= 15 A
    # 2. dt <= 6s (~4s)
    # 3. Both measurements valid
    # 4. Same session_id
    # 5. Both observations active state
    
    dt_s = full_resampled['Time'].diff().dt.total_seconds()
    same_sess = full_resampled['session_id'] == full_resampled['session_id'].shift(1)
    dV = full_resampled['Fuel Cell Total Voltage V'].diff()
    dI = full_resampled['Stack Current Sensor Value A'].diff()
    both_active = full_resampled['is_active'] & full_resampled['is_active'].shift(1)
    
    r_est_valid = (
        (dI.abs() >= 15.0) & 
        (dt_s >= 2.0) & (dt_s <= 6.0) & 
        same_sess & 
        both_active & 
        dV.notna() & 
        dI.notna()
    )
    
    full_resampled['R_est_Ohm'] = np.where(r_est_valid, (dV.abs() / dI.abs()), np.nan)
    
    # Air Stoichiometry Proxy \lambda_O2
    full_resampled['Air_Stoichiometry_Lambda'] = np.where(
        full_resampled['is_active'] & (full_resampled['Stack Current Sensor Value A'] > 5.0),
        full_resampled['Air Flow Sensor kg/h'] / (full_resampled['Stack Current Sensor Value A'] * 1.25),
        np.nan
    )
    
    # Cumulative Energy Output E_cum (kWh)
    dt_hours = dt_s.fillna(0) / 3600.0
    dt_hours = np.where(dt_s > 300, 0, dt_hours) # zero gaps
    full_resampled['dt_h'] = dt_hours
    full_resampled['E_cum_kWh'] = (full_resampled['Stack Power kW'] * full_resampled['dt_h']).cumsum()
    
    # High Current Residence Time (t_150A)
    is_high_curr = (full_resampled['Stack Current Sensor Value A'] > 150.0) & full_resampled['is_active']
    full_resampled['high_current_duration_h'] = np.where(is_high_curr, full_resampled['dt_h'], 0.0).cumsum()

    # Nernst Voltage Compensation
    T_k = full_resampled['Stack Coolant Temp Outlet degree C'] + 273.15
    P_kPa = full_resampled['Ambient Air Pressure kPa'].fillna(96.0)
    E_nernst_cell = 1.229 - 0.00085 * (T_k - 298.15) + (4.31e-5 * T_k * np.log(P_kPa / 101.325))
    # Assuming ~360 nominal cell stack
    full_resampled['E_Nernst_Stack_V'] = E_nernst_cell * 360.0
    full_resampled['Delta_V_Nernst'] = full_resampled['Fuel Cell Total Voltage V'] - full_resampled['E_Nernst_Stack_V']

    # 7. Export Outputs into Layout Directories
    
    # Active-state dataset export
    active_df = full_resampled[full_resampled['is_active']].copy().reset_index(drop=True)
    active_csv = os.path.join(dirs["ml"], "clean_pem_telemetry_active.csv")
    active_df.to_csv(active_csv, index=False)
    print(f"Exported clean active telemetry ({len(active_df)} rows) to: {active_csv}")
    
    # Session summary export
    sess_df = pd.DataFrame(session_summaries)
    sess_csv = os.path.join(dirs["ml"], "pem_session_summary.csv")
    sess_df.to_csv(sess_csv, index=False)
    print(f"Exported session summary ({len(sess_df)} sessions) to: {sess_csv}")
    
    # Parquet feature matrix export
    features_parquet = os.path.join(dirs["features"], "pem_features.parquet")
    full_resampled.to_parquet(features_parquet, index=False)
    print(f"Exported feature matrix to: {features_parquet}")

    # Chronological Train / Val / Test Splits by Session ID (70% / 15% / 15%)
    all_session_ids = sess_df['session_id'].values
    n_sess = len(all_session_ids)
    train_end = int(n_sess * 0.70)
    val_end = int(n_sess * 0.85)
    
    train_ids = set(all_session_ids[:train_end])
    val_ids = set(all_session_ids[train_end:val_end])
    test_ids = set(all_session_ids[val_end:])
    
    train_df = full_resampled[full_resampled['session_id'].isin(train_ids)].reset_index(drop=True)
    val_df = full_resampled[full_resampled['session_id'].isin(val_ids)].reset_index(drop=True)
    test_df = full_resampled[full_resampled['session_id'].isin(test_ids)].reset_index(drop=True)
    
    train_df.to_parquet(os.path.join(dirs["splits"], "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(dirs["splits"], "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(dirs["splits"], "test.parquet"), index=False)
    print(f"Exported Parquet splits -> Train ({len(train_ids)} sess), Val ({len(val_ids)} sess), Test ({len(test_ids)} sess)")

    # 8. Preprocessing JSON Report
    report = {
        "raw_total_rows": total_raw_rows,
        "resampled_total_rows": len(full_resampled),
        "active_total_rows": len(active_df),
        "total_sessions": num_sessions,
        "invalid_sanitization_counts": invalid_counts,
        "valid_r_est_samples": int(full_resampled['R_est_Ohm'].notna().sum()),
        "total_energy_delivered_kwh": float(full_resampled['E_cum_kWh'].max()),
        "total_high_current_hours": float(full_resampled['high_current_duration_h'].max()),
        "splits_session_counts": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids)
        }
    }
    
    report_path = os.path.join(dirs["reports"], "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved preprocessing report to: {report_path}")

if __name__ == "__main__":
    main()
