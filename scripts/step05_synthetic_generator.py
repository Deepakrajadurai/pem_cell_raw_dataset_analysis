"""
Step 5: Physics-Constrained Synthetic Data Generation & Audit
Implements the 3-Layer Hybrid Generative Strategy:
- Layer 1: Session & Block Bootstrap from 80% Training Sessions (Zero Test Leakage)
- Layer 2: Physics-Constrained & Conditional Perturbation (P(V|I, T, Airflow, Ambient))
- Layer 3: Empirical Residual Sampling V_synthetic = V_expected + Residual_synthetic

Performs full synthetic data audit, polarization comparison, and downstream validation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def run_step_5(data_path, outputs_dir, workspace_dir=None):
    print(f"[Step 5] Initializing Physics-Constrained Synthetic Data Generator from: {data_path}")
    df = pd.read_csv(data_path)
    df['Time_dt'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time_dt').reset_index(drop=True)

    # Clean sensor glitches
    valid_mask = (
        (df['Fuel Cell Total Voltage V'] > 50.0) & 
        (df['Fuel Cell Total Voltage V'] < 420.0) & 
        (df['Stack Current Sensor Value A'] >= 0.0) &
        (df['Stack Coolant Temp Inlet degree C'] > 0.0) &
        (df['Air Flow Sensor kg/h'] >= 0.0)
    )
    df_clean = df[valid_mask].copy()

    # Session segmentation (> 300s gap)
    time_diffs = df_clean['Time_dt'].diff().dt.total_seconds()
    df_clean['session_id'] = ((time_diffs > 300) | (time_diffs.isna())).cumsum()

    session_ids = df_clean['session_id'].unique()
    np.random.seed(42)
    np.random.shuffle(session_ids)

    # 1. Session-Based 80/10/10 Split (Zero Leakage)
    n_total = len(session_ids)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    train_sessions = set(session_ids[:n_train])
    val_sessions = set(session_ids[n_train:n_train+n_val])
    test_sessions = set(session_ids[n_train+n_val:])

    train_df = df_clean[df_clean['session_id'].isin(train_sessions)].copy()
    val_df = df_clean[df_clean['session_id'].isin(val_sessions)].copy()
    test_df = df_clean[df_clean['session_id'].isin(test_sessions)].copy()

    print(f"[Step 5] Session Split -> Train: {len(train_sessions)} sess ({len(train_df)} rows) | Val: {len(val_sessions)} sess | Test: {len(test_sessions)} sess ({len(test_df)} rows)")

    # 2. Fit Baseline Physics Regressor V_expected = f(I, T_coolant, T_air, Airflow, Ambient) on Train Set ONLY
    feature_cols = [
        'Stack Current Sensor Value A',
        'Stack Coolant Temp Inlet degree C',
        'Stack Coolant Temp Outlet degree C',
        'Air Temp Stack Outlet degree C',
        'Air Flow Sensor kg/h',
        'Ambient Air Temp degree C',
        'Ambient Air Pressure kPa'
    ]
    target_col = 'Fuel Cell Total Voltage V'

    model_phys = HistGradientBoostingRegressor(random_state=42, max_iter=250, learning_rate=0.08)
    model_phys.fit(train_df[feature_cols], train_df[target_col])

    train_df['V_expected'] = model_phys.predict(train_df[feature_cols])
    train_df['Residual'] = train_df[target_col] - train_df['V_expected']

    # 3. Hybrid Synthesis Loop (Resample Blocks + Perturb + Residual Injection)
    synthetic_rows = []
    unique_train_sess = list(train_sessions)
    num_synth_sessions = 150

    for synth_id in range(num_synth_sessions):
        src_sess_id = np.random.choice(unique_train_sess)
        sess_block = train_df[train_df['session_id'] == src_sess_id].copy()
        if len(sess_block) < 15:
            continue

        # Controlled local perturbations based on empirical standard deviations
        curr_noise = np.random.normal(0, 1.2, len(sess_block))
        synth_curr = np.clip(sess_block['Stack Current Sensor Value A'].values + curr_noise, 0.0, 255.0)

        t_in_noise = np.random.normal(0, 0.25, len(sess_block))
        synth_t_in = np.clip(sess_block['Stack Coolant Temp Inlet degree C'].values + t_in_noise, 10.0, 80.0)

        t_out_noise = np.random.normal(0, 0.25, len(sess_block))
        synth_t_out = np.clip(sess_block['Stack Coolant Temp Outlet degree C'].values + t_out_noise, 10.0, 85.0)

        t_air_noise = np.random.normal(0, 0.25, len(sess_block))
        synth_t_air = np.clip(sess_block['Air Temp Stack Outlet degree C'].values + t_air_noise, 10.0, 80.0)

        flow_noise = np.random.normal(0, 1.5, len(sess_block))
        synth_flow = np.clip(sess_block['Air Flow Sensor kg/h'].values + flow_noise, 0.0, 407.0)

        amb_t_noise = np.random.normal(0, 0.1, len(sess_block))
        synth_amb_t = sess_block['Ambient Air Temp degree C'].values + amb_t_noise

        amb_p_noise = np.random.normal(0, 0.002, len(sess_block))
        synth_amb_p = np.clip(sess_block['Ambient Air Pressure kPa'].values + amb_p_noise, 0.90, 1.05)

        synth_features = pd.DataFrame({
            'Stack Current Sensor Value A': synth_curr,
            'Stack Coolant Temp Inlet degree C': synth_t_in,
            'Stack Coolant Temp Outlet degree C': synth_t_out,
            'Air Temp Stack Outlet degree C': synth_t_air,
            'Air Flow Sensor kg/h': synth_flow,
            'Ambient Air Temp degree C': synth_amb_t,
            'Ambient Air Pressure kPa': synth_amb_p
        })

        v_exp_synth = model_phys.predict(synth_features)
        residuals_sampled = np.random.choice(train_df['Residual'].values, size=len(sess_block), replace=True)
        v_synth = np.clip(v_exp_synth + residuals_sampled, 100.0, 412.0)

        # Enforce exact power calculation P = V * I / 1000
        p_synth = (v_synth * synth_curr) / 1000.0

        synth_block = synth_features.copy()
        synth_block['Fuel Cell Total Voltage V'] = v_synth
        synth_block['Stack Power kW'] = p_synth
        synth_block['synth_session_id'] = synth_id
        synthetic_rows.append(synth_block)

    synth_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"[Step 5] Generated {len(synth_df)} synthetic samples across {num_synth_sessions} synthetic sessions.")

    # Save synthetic dataset CSV
    os.makedirs(outputs_dir, exist_ok=True)
    synth_csv_path = os.path.join(outputs_dir, "synthetic_pem_dataset.csv")
    synth_df.to_csv(synth_csv_path, index=False)

    # 4. Downstream Validation Experiment (Evaluated on Unseen Real Test Set)
    reg_a = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    reg_a.fit(train_df[feature_cols], train_df[target_col])
    pred_a = reg_a.predict(test_df[feature_cols])
    mae_a = float(mean_absolute_error(test_df[target_col], pred_a))
    r2_a = float(r2_score(test_df[target_col], pred_a))

    combined_train = pd.concat([train_df[feature_cols + [target_col]], synth_df[feature_cols + [target_col]]], ignore_index=True)
    reg_b = HistGradientBoostingRegressor(random_state=42, max_iter=200)
    reg_b.fit(combined_train[feature_cols], combined_train[target_col])
    pred_b = reg_b.predict(test_df[feature_cols])
    mae_b = float(mean_absolute_error(test_df[target_col], pred_b))
    r2_b = float(r2_score(test_df[target_col], pred_b))

    print(f"[Step 5] Model A (Real Train Only)       -> MAE: {mae_a:.4f} V | R^2: {r2_a:.4f}")
    print(f"[Step 5] Model B (Real + Synthetic Train) -> MAE: {mae_b:.4f} V | R^2: {r2_b:.4f}")

    # Audit Metrics
    audit_summary = {
        "step": "step05_synthetic_generator",
        "total_synthetic_rows": len(synth_df),
        "total_synthetic_sessions": num_synth_sessions,
        "session_split": {
            "train_sessions": len(train_sessions),
            "val_sessions": len(val_sessions),
            "test_sessions": len(test_sessions)
        },
        "downstream_validation_experiment": {
            "model_a_real_train_only": {"mae_volts": round(mae_a, 4), "r2_score": round(r2_a, 4)},
            "model_b_real_plus_synthetic": {"mae_volts": round(mae_b, 4), "r2_score": round(r2_b, 4)},
            "delta_mae_volts": round(mae_a - mae_b, 4)
        },
        "synthetic_statistics": {
            "voltage": {"mean": round(float(synth_df['Fuel Cell Total Voltage V'].mean()), 2), "std": round(float(synth_df['Fuel Cell Total Voltage V'].std()), 2)},
            "current": {"mean": round(float(synth_df['Stack Current Sensor Value A'].mean()), 2), "std": round(float(synth_df['Stack Current Sensor Value A'].std()), 2)},
            "power": {"mean": round(float(synth_df['Stack Power kW'].mean()), 2), "std": round(float(synth_df['Stack Power kW'].std()), 2)}
        }
    }

    audit_json_path = os.path.join(outputs_dir, "synthetic_audit_summary.json")
    with open(audit_json_path, 'w') as f:
        json.dump(audit_summary, f, indent=2)

    # 5. Diagnostic Charts
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    dpi = 150

    # Chart 1: Real vs Synthetic Polarization Curve Comparison
    real_iv = train_df[(train_df['Fuel Cell Total Voltage V'] > 0) & (train_df['Stack Current Sensor Value A'] > 0)].copy()
    real_iv['current_bin'] = pd.cut(real_iv['Stack Current Sensor Value A'], bins=np.arange(0, 260, 10))
    real_pol = real_iv.groupby('current_bin', observed=False).agg(
        mean_v=('Fuel Cell Total Voltage V', 'mean'), mean_i=('Stack Current Sensor Value A', 'mean')
    ).dropna().reset_index()

    synth_iv = synth_df[(synth_df['Fuel Cell Total Voltage V'] > 0) & (synth_df['Stack Current Sensor Value A'] > 0)].copy()
    synth_iv['current_bin'] = pd.cut(synth_iv['Stack Current Sensor Value A'], bins=np.arange(0, 260, 10))
    synth_pol = synth_iv.groupby('current_bin', observed=False).agg(
        mean_v=('Fuel Cell Total Voltage V', 'mean'), mean_i=('Stack Current Sensor Value A', 'mean')
    ).dropna().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    ax.plot(real_pol['mean_i'], real_pol['mean_v'], 'o-', color='#1f77b4', linewidth=2.5, label='Real Training Polarization Curve')
    ax.plot(synth_pol['mean_i'], synth_pol['mean_v'], 's--', color='#d62728', linewidth=2.0, label='Synthetic Polarization Curve')
    ax.set_xlabel('Stack Current Sensor Value (A)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fuel Cell Total Voltage (V)', fontsize=11, fontweight='bold')
    ax.set_title('Real vs Synthetic Polarization Curve Comparison (Physics Preservation Audit)', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()

    pol_comp_path = os.path.join(outputs_dir, 'synthetic_vs_real_polarization.png')
    plt.savefig(pol_comp_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'synthetic_vs_real_polarization.png'))
    plt.close()

    # Chart 2: Synthetic Pearson Correlation Matrix Heatmap
    raw_cols = ['Fuel Cell Total Voltage V', 'Stack Current Sensor Value A', 'Stack Coolant Temp Inlet degree C', 'Stack Coolant Temp Outlet degree C', 'Air Temp Stack Outlet degree C', 'Air Flow Sensor kg/h', 'Air Comp Motor Temp degree C', 'Fahrzeuggeschwindigkeit km/h']
    avail_cols = [c for c in raw_cols if c in synth_df.columns]
    num_labels = ['Voltage', 'Current', 'Coolant In', 'Coolant Out', 'Air Out', 'Air Flow']
    corr_synth = synth_df[avail_cols].corr().values

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=dpi)
    cax = ax.matshow(corr_synth, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(avail_cols)))
    ax.set_yticks(range(len(avail_cols)))
    ax.set_xticklabels(num_labels[:len(avail_cols)], rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(num_labels[:len(avail_cols)], fontsize=9)

    for i in range(len(avail_cols)):
        for j in range(len(avail_cols)):
            ax.text(j, i, f"{corr_synth[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr_synth[i, j]) < 0.7 else 'white', fontsize=8, fontweight='bold')

    plt.title('Synthetic Data Pearson Correlation Matrix', fontsize=12, fontweight='bold', pad=25)
    fig.tight_layout()

    corr_synth_path = os.path.join(outputs_dir, 'synthetic_correlation_matrix.png')
    plt.savefig(corr_synth_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'synthetic_correlation_matrix.png'))
    plt.close()

    # Chart 3: Sample Synthetic Drive Session Telemetry
    sample_synth_sess = synth_df[synth_df['synth_session_id'] == 5].copy().reset_index(drop=True)
    sample_synth_sess['Elapsed_samples'] = sample_synth_sess.index * 4 / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=dpi)
    ax1.plot(sample_synth_sess['Elapsed_samples'], sample_synth_sess['Fuel Cell Total Voltage V'], color='#1f77b4', linewidth=1.5, label='Synthetic Voltage (V)')
    ax1.set_ylabel('Voltage (V)', color='#1f77b4', fontweight='bold')
    ax1.set_title('Representative Synthetic Drive Cycle Telemetry (Hybrid Block Resample + Noise)', fontsize=12, fontweight='bold')
    ax1.grid(True)

    ax2.plot(sample_synth_sess['Elapsed_samples'], sample_synth_sess['Stack Current Sensor Value A'], color='#ff7f0e', linewidth=1.5, label='Synthetic Current (A)')
    ax2.plot(sample_synth_sess['Elapsed_samples'], sample_synth_sess['Air Flow Sensor kg/h'], color='#2ca02c', alpha=0.7, linewidth=1.2, label='Synthetic Air Flow (kg/h)')
    ax2.set_ylabel('Current / Flow', fontweight='bold')
    ax2.set_xlabel('Elapsed Time (minutes)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)

    fig.tight_layout()
    ts_synth_path = os.path.join(outputs_dir, 'synthetic_drive_session.png')
    plt.savefig(ts_synth_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'synthetic_drive_session.png'))
    plt.close()

    print(f"[Step 5] Synthetic generation & audit complete! Output saved to {outputs_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    out_dir = os.path.join(base_dir, "outputs")
    run_step_5(data_file, out_dir, workspace_dir=base_dir)
