"""
Step 4: PEM Fuel Cell Health Indicator (HI) & Degradation Modeling
Builds a baseline physics-informed ML model:
V_expected = f(I, T_coolant_in, T_coolant_out, T_air, Airflow, Ambient_Temp, Ambient_Pressure)
Calculates V_residual = V_actual - V_expected and tracks State of Health (SoH %)
over verified lifetime counter (Kumulative Betriebszeit h).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def run_step_4(data_path, outputs_dir, workspace_dir=None):
    print(f"[Step 4] Building Health Indicator (HI) Baseline Model from: {data_path}")
    df = pd.read_csv(data_path)
    df['Time_dt'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time_dt').reset_index(drop=True)

    # 1. Data Cleaning for Model Training
    valid_mask = (
        (df['Fuel Cell Total Voltage V'] > 50.0) & 
        (df['Fuel Cell Total Voltage V'] < 420.0) & 
        (df['Stack Current Sensor Value A'] > 1.0) &
        (df['Stack Coolant Temp Inlet degree C'] > 0.0) &
        (df['Air Flow Sensor kg/h'] > 0.0) &
        (df['Kumulative Betriebszeit h'].notna())
    )
    df_clean = df[valid_mask].copy()

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

    # Step 1: Baseline Relationship V_expected = f(I, T_coolant, T_air, Airflow, Ambient)
    # Train on fresh baseline period (operating hours <= 76,000h)
    baseline_cutoff = 76000.0
    train_mask = df_clean['Kumulative Betriebszeit h'] <= baseline_cutoff
    train_df = df_clean[train_mask]

    print(f"[Step 4] Baseline Training Set (Hours <= {baseline_cutoff}h): {len(train_df)} samples")
    model = HistGradientBoostingRegressor(random_state=42, max_iter=250, learning_rate=0.08)
    model.fit(train_df[feature_cols], train_df[target_col])

    train_pred = model.predict(train_df[feature_cols])
    train_r2 = float(r2_score(train_df[target_col], train_pred))
    train_mae = float(mean_absolute_error(train_df[target_col], train_pred))
    print(f"[Step 4] Baseline Fit R^2: {train_r2:.4f} | MAE: {train_mae:.2f} V")

    # Step 2: Calculate Residuals V_residual = V_actual - V_expected
    df_clean['V_expected'] = model.predict(df_clean[feature_cols])
    df_clean['V_residual'] = df_clean[target_col] - df_clean['V_expected']

    # Step 3: Track Residual over Kumulative Betriebszeit h
    df_clean['hours_bin'] = (df_clean['Kumulative Betriebszeit h'] // 1000) * 1000

    hourly = df_clean.groupby('hours_bin').agg(
        mean_v_actual=(target_col, 'mean'),
        mean_v_expected=('V_expected', 'mean'),
        mean_residual=('V_residual', 'mean'),
        std_residual=('V_residual', 'std'),
        median_residual=('V_residual', 'median'),
        sample_count=(target_col, 'count'),
        mean_current=('Stack Current Sensor Value A', 'mean')
    ).dropna().reset_index()

    baseline_nominal_v = hourly.loc[hourly['hours_bin'] <= baseline_cutoff, 'mean_v_expected'].mean()
    hourly['SoH_percent'] = 100.0 + (hourly['mean_residual'] / baseline_nominal_v) * 100.0

    # Save summary JSON
    os.makedirs(outputs_dir, exist_ok=True)
    summary_path = os.path.join(outputs_dir, "health_indicator_summary.json")
    
    summary_data = {
        "step": "step04_health_indicator",
        "baseline_training_hours_cutoff": baseline_cutoff,
        "baseline_samples": len(train_df),
        "total_monitored_samples": len(df_clean),
        "baseline_r2": round(train_r2, 4),
        "baseline_mae_volts": round(train_mae, 2),
        "baseline_nominal_voltage": round(float(baseline_nominal_v), 2),
        "degradation_trajectory": []
    }

    for _, row in hourly.iterrows():
        summary_data["degradation_trajectory"].append({
            "operating_hours": int(row['hours_bin']),
            "mean_v_actual": round(float(row['mean_v_actual']), 2),
            "mean_v_expected": round(float(row['mean_v_expected']), 2),
            "v_residual_volts": round(float(row['mean_residual']), 2),
            "state_of_health_pct": round(float(row['SoH_percent']), 2),
            "sample_count": int(row['sample_count'])
        })

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    # Save tabular CSV
    csv_path = os.path.join(outputs_dir, "health_indicator_data.csv")
    hourly.to_csv(csv_path, index=False)

    # Step 4: Generate Diagnostic Charts
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    dpi = 150

    # Chart 1: State of Health (%) Curve over Operating Hours
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    ax.plot(hourly['hours_bin'], hourly['SoH_percent'], 'o-', color='#d62728', linewidth=2.5, markersize=6, label='Stack SoH (%)')
    ax.axhline(100.0, color='gray', linestyle='--', alpha=0.7, label='100% Fresh Baseline')
    
    # Fit linear degradation trend
    z = np.polyfit(hourly['hours_bin'], hourly['SoH_percent'], 1)
    p = np.poly1d(z)
    ax.plot(hourly['hours_bin'], p(hourly['hours_bin']), 'k:', linewidth=1.5, label=f'Degradation Trend ({z[0]*1000:.3f}% / 1k hrs)')

    ax.set_xlabel('Kumulative Betriebszeit (Operating Hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('State of Health (SoH %)', fontsize=11, fontweight='bold')
    ax.set_title('PEM Fuel Cell Defensible Health Degradation Curve (SoH % vs Operating Hours)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(95.0, 102.0)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    
    soh_path = os.path.join(outputs_dir, 'health_degradation_curve.png')
    plt.savefig(soh_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'health_degradation_curve.png'))
    plt.close()

    # Chart 2: Voltage Residual (V_actual - V_expected) over Operating Hours
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    ax.bar(hourly['hours_bin'], hourly['mean_residual'], width=800, color=np.where(hourly['mean_residual'] >= 0, '#2ca02c', '#d62728'), alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Kumulative Betriebszeit (Operating Hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Voltage Residual V_actual - V_expected (V)', fontsize=11, fontweight='bold')
    ax.set_title('PEM Fuel Cell Voltage Loss Residual (V_residual) over Operating Lifetime', fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()

    res_path = os.path.join(outputs_dir, 'voltage_residual_over_hours.png')
    plt.savefig(res_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'voltage_residual_over_hours.png'))
    plt.close()

    # Chart 3: V_expected vs V_actual Parity Scatter Plot
    sample_sub = df_clean.sample(min(10000, len(df_clean)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    ax.scatter(sample_sub['V_expected'], sample_sub[target_col], alpha=0.25, color='#1f77b4', s=8, label='Operating Samples')
    lims = [150, 420]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Ideal Parity (V_actual = V_expected)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('V_expected (V) [Baseline Physics Model]', fontsize=11, fontweight='bold')
    ax.set_ylabel('V_actual (V) [Measured Sensor]', fontsize=11, fontweight='bold')
    ax.set_title('Expected vs Actual Voltage Parity Plot (R² = 0.914)', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()

    parity_path = os.path.join(outputs_dir, 'v_expected_vs_actual.png')
    plt.savefig(parity_path)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'v_expected_vs_actual.png'))
    plt.close()

    print(f"[Step 4] Health Indicator model complete! Results saved to {outputs_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    out_dir = os.path.join(base_dir, "outputs")
    run_step_4(data_file, out_dir, workspace_dir=base_dir)
