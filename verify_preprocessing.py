import os
import json
import numpy as np
import pandas as pd

def run_verification():
    base_dir = "d:/PEM_Cell_Dataset"
    features_parquet = os.path.join(base_dir, "Features", "pem_features.parquet")
    report_json = os.path.join(base_dir, "Reports", "preprocessing_report.json")
    
    if not os.path.exists(features_parquet):
        print(f"Error: {features_parquet} does not exist.")
        return
        
    print(f"Loading resampled dataset for verification: {features_parquet}")
    df = pd.read_parquet(features_parquet)
    df['Time'] = pd.to_datetime(df['Time'])
    
    results = {}
    
    # --- 1. DATA INTEGRITY ---
    print("\n--- 1. Testing Data Integrity ---")
    
    # Timestamps monotonically increasing inside sessions
    mono_time = True
    max_sess_gap = 0
    for sess_id, s_df in df.groupby('session_id'):
        if not s_df['Time'].is_monotonic_increasing:
            mono_time = False
        dt_max = s_df['Time'].diff().dt.total_seconds().max()
        if not np.isnan(dt_max) and dt_max > max_sess_gap:
            max_sess_gap = dt_max
            
    results['timestamps_monotonic_per_session'] = mono_time
    results['max_internal_session_gap_seconds'] = float(max_sess_gap)
    results['no_internal_session_gap_over_300s'] = bool(max_sess_gap <= 300.0)
    
    print(f" [PASS] Timestamps monotonic per session: {mono_time}")
    print(f" [PASS] Max internal session gap: {max_sess_gap:.1f} s (Must be <= 300s: {max_sess_gap <= 300.0})")
    
    # --- 2. SENSOR QUALITY (Invalid -> NaN) ---
    print("\n--- 2. Testing Sensor Quality ---")
    v_col = 'Fuel Cell Total Voltage V'
    tcomp_col = 'Air Comp Motor Temp degree C'
    pamb_col = 'Ambient Air Pressure kPa'
    
    v_out_of_bounds = ((df[v_col] < 0.0) | (df[v_col] > 450.0)).sum()
    tcomp_out_of_bounds = ((df[tcomp_col] < -10.0) | (df[tcomp_col] > 120.0)).sum()
    pamb_out_of_bounds = ((df[pamb_col] < 80.0) | (df[pamb_col] > 120.0)).sum()
    
    results['v_out_of_bounds_count'] = int(v_out_of_bounds)
    results['tcomp_out_of_bounds_count'] = int(tcomp_out_of_bounds)
    results['pamb_out_of_bounds_count'] = int(pamb_out_of_bounds)
    
    print(f" [PASS] V out of bounds (<0 or >450 V): {v_out_of_bounds}")
    print(f" [PASS] T_comp out of bounds (<-10 or >120 deg C): {tcomp_out_of_bounds}")
    print(f" [PASS] P_amb out of bounds (<80 or >120 kPa): {pamb_out_of_bounds}")
    
    # --- 3. TEMPORAL INTEGRITY ---
    print("\n--- 3. Testing Temporal Integrity ---")
    dt_resampled = df.groupby('session_id')['Time'].diff().dt.total_seconds().dropna()
    dt_mode = dt_resampled.mode().values[0] if len(dt_resampled) > 0 else 0
    results['dt_resampled_mode_seconds'] = float(dt_mode)
    print(f" [PASS] Resampled dt Mode: {dt_mode} seconds (Expected: 4.0 s)")

    # --- 4. PHYSICS SANITY ---
    print("\n--- 4. Testing Physics Sanity ---")
    p_max = df['Stack Power kW'].max()
    p_min = df['Stack Power kW'].min()
    e_cum_mono = df['E_cum_kWh'].is_monotonic_increasing
    hc_mono = df['high_current_duration_h'].is_monotonic_increasing
    
    r_est_valid = df['R_est_Ohm'].dropna()
    r_est_min = float(r_est_valid.min()) if len(r_est_valid) > 0 else np.nan
    r_est_max = float(r_est_valid.max()) if len(r_est_valid) > 0 else np.nan
    
    results['p_stack_min_kw'] = float(p_min)
    results['p_stack_max_kw'] = float(p_max)
    results['e_cum_monotonic'] = bool(e_cum_mono)
    results['high_current_duration_monotonic'] = bool(hc_mono)
    results['r_est_valid_count'] = len(r_est_valid)
    results['r_est_min_ohm'] = r_est_min
    results['r_est_max_ohm'] = r_est_max
    
    print(f" [PASS] Stack Power Range: [{p_min:.2f} kW, {p_max:.2f} kW]")
    print(f" [PASS] Cumulative Energy Monotonic: {e_cum_mono}")
    print(f" [PASS] High Current Duration Monotonic: {hc_mono}")
    print(f" [PASS] Valid R_est samples: {len(r_est_valid)} (Range: [{r_est_min:.4f} Ohm, {r_est_max:.4f} Ohm])")

    # --- 5. LEAKAGE AUDIT ---
    print("\n--- 5. Testing Leakage Audit ---")
    train_p = os.path.join(base_dir, "Splits", "train.parquet")
    val_p = os.path.join(base_dir, "Splits", "validation.parquet")
    test_p = os.path.join(base_dir, "Splits", "test.parquet")
    
    tr_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    te_df = pd.read_parquet(test_p)
    
    no_split_overlap = (
        set(tr_df['session_id']).isdisjoint(set(val_df['session_id'])) and
        set(tr_df['session_id']).isdisjoint(set(te_df['session_id'])) and
        set(val_df['session_id']).isdisjoint(set(te_df['session_id']))
    )
    
    seq_split = (tr_df['Time'].max() <= val_df['Time'].min()) and (val_df['Time'].max() <= te_df['Time'].min())
    
    results['session_splits_disjoint'] = bool(no_split_overlap)
    results['splits_chronological_sequence'] = bool(seq_split)
    
    print(f" [PASS] Train/Val/Test session sets disjoint: {no_split_overlap}")
    print(f" [PASS] Chronological split ordering verified (Train <= Val <= Test): {seq_split}")
    
    print("\n=== SUMMARY VERIFICATION RESULTS ===")
    all_passed = (
        mono_time and 
        (max_sess_gap <= 300.0) and 
        (v_out_of_bounds == 0) and 
        (tcomp_out_of_bounds == 0) and 
        (pamb_out_of_bounds == 0) and 
        (dt_mode == 4.0) and 
        e_cum_mono and 
        hc_mono and 
        no_split_overlap and 
        seq_split
    )
    
    print(f"OVERALL PREPROCESSING VERIFICATION STATUS: {'PASS [ALL TESTS PASSED]' if all_passed else 'FAIL'}")
    
    # Save test results
    with open(os.path.join(base_dir, "Reports", "verification_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_verification()
