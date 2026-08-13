"""
Step 3: Visual Chart Generation
Dataset: PEM Fuel Cell Raw Time Series Data (raw_dataset.csv)
Generates high-resolution PNG charts for reports and audit review.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_step_3(data_path, output_dir, workspace_dir=None):
    print(f"[Step 3] Generating visual charts from: {data_path}")
    df = pd.read_csv(data_path)
    df['Time_dt'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time_dt').reset_index(drop=True)
    df['Stack Power kW'] = (df['Fuel Cell Total Voltage V'] * df['Stack Current Sensor Value A']) / 1000.0

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    dpi = 150
    os.makedirs(output_dir, exist_ok=True)

    # 1. Polarization & Power Curve
    valid_iv = df[(df['Fuel Cell Total Voltage V'] > 0) & (df['Stack Current Sensor Value A'] > 0)].copy()
    valid_iv['current_bin'] = pd.cut(valid_iv['Stack Current Sensor Value A'], bins=np.arange(0, 260, 5))
    pol = valid_iv.groupby('current_bin', observed=False).agg(
        mean_v=('Fuel Cell Total Voltage V', 'mean'),
        std_v=('Fuel Cell Total Voltage V', 'std'),
        mean_i=('Stack Current Sensor Value A', 'mean'),
        mean_p=('Stack Power kW', 'mean')
    ).dropna().reset_index()

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    c1 = '#1f77b4'
    ax1.set_xlabel('Stack Current Sensor Value (A)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fuel Cell Total Voltage (V)', color=c1, fontsize=11, fontweight='bold')
    ax1.plot(pol['mean_i'], pol['mean_v'], 'o-', color=c1, linewidth=2, label='Stack Voltage (V)')
    ax1.fill_between(pol['mean_i'], pol['mean_v'] - pol['std_v'], pol['mean_v'] + pol['std_v'], color=c1, alpha=0.15)
    ax1.tick_params(axis='y', labelcolor=c1)

    ax2 = ax1.twinx()  
    c2 = '#d62728'
    ax2.set_ylabel('Stack Power (kW)', color=c2, fontsize=11, fontweight='bold')
    ax2.plot(pol['mean_i'], pol['mean_p'], 's--', color=c2, linewidth=2, label='Stack Power (kW)')
    ax2.tick_params(axis='y', labelcolor=c2)

    plt.title('PEM Fuel Cell Polarization & Power Curve (V & P vs I)', fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    pol_file = os.path.join(output_dir, 'polarization_curve.png')
    plt.savefig(pol_file)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'polarization_curve.png'))
    plt.close()

    # 2. Operating Regimes Pie Chart
    curr = df['Stack Current Sensor Value A'].dropna()
    idle = (curr <= 1.0).sum()
    low = ((curr > 1.0) & (curr <= 50.0)).sum()
    med = ((curr > 50.0) & (curr <= 150.0)).sum()
    high = (curr > 150.0).sum()

    labels = ['Idle / Off (<=1A)', 'Low Load (1-50A)', 'Medium Load (50-150A)', 'High Load (>150A)']
    sizes = [idle, low, med, high]
    colors = ['#aec7e8', '#2ca02c', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title('PEM Fuel Cell Load Profile / Operating Regimes Breakdown', fontsize=12, fontweight='bold')
    fig.tight_layout()
    pie_file = os.path.join(output_dir, 'operating_regimes.png')
    plt.savefig(pie_file)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'operating_regimes.png'))
    plt.close()

    # 3. Correlation Matrix Heatmap
    num_cols = ['Voltage (V)', 'Current (A)', 'Coolant T_in (°C)', 'Coolant T_out (°C)', 'Air T_out (°C)', 'Air Flow (kg/h)', 'Comp Motor T (°C)', 'Speed (km/h)']
    raw_cols = ['Fuel Cell Total Voltage V', 'Stack Current Sensor Value A', 'Stack Coolant Temp Inlet degree C', 'Stack Coolant Temp Outlet degree C', 'Air Temp Stack Outlet degree C', 'Air Flow Sensor kg/h', 'Air Comp Motor Temp degree C', 'Fahrzeuggeschwindigkeit km/h']
    corr = df[raw_cols].corr().values

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=dpi)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='left', fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr[i, j]) < 0.7 else 'white', fontsize=8, fontweight='bold')

    plt.title('Parameter Pearson Correlation Matrix', fontsize=12, fontweight='bold', pad=25)
    fig.tight_layout()
    corr_file = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(corr_file)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'correlation_matrix.png'))
    plt.close()

    # 4. Drive Session Time Series Profile
    time_diffs = df['Time_dt'].diff().dt.total_seconds()
    df['session_id'] = ((time_diffs > 300) | (time_diffs.isna())).cumsum()
    session_counts = df['session_id'].value_counts()
    sample_session_id = session_counts[session_counts > 500].index[5]
    sample_df = df[df['session_id'] == sample_session_id].copy()
    sample_df['Elapsed_min'] = (sample_df['Time_dt'] - sample_df['Time_dt'].min()).dt.total_seconds() / 60.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True, dpi=dpi)
    ax1.plot(sample_df['Elapsed_min'], sample_df['Fuel Cell Total Voltage V'], color='#1f77b4')
    ax1.set_ylabel('Voltage (V)', color='#1f77b4', fontweight='bold')
    ax1.set_title(f'Representative Drive Session Telemetry (Session #{sample_session_id}, {sample_df["Elapsed_min"].max():.1f} min)', fontsize=12, fontweight='bold')
    ax1.grid(True)

    ax2.plot(sample_df['Elapsed_min'], sample_df['Stack Current Sensor Value A'], color='#ff7f0e', label='Current (A)')
    ax2.plot(sample_df['Elapsed_min'], sample_df['Air Flow Sensor kg/h'], color='#2ca02c', alpha=0.7, label='Air Flow (kg/h)')
    ax2.set_ylabel('Current / Flow', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True)

    ax3.plot(sample_df['Elapsed_min'], sample_df['Stack Coolant Temp Outlet degree C'], color='#d62728', label='Coolant Out (°C)')
    ax3.plot(sample_df['Elapsed_min'], sample_df['Stack Coolant Temp Inlet degree C'], color='#9467bd', label='Coolant In (°C)')
    ax3.set_ylabel('Temperature (°C)', fontweight='bold')
    ax3.set_xlabel('Session Elapsed Time (minutes)', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True)

    fig.tight_layout()
    ts_file = os.path.join(output_dir, 'sample_drive_session.png')
    plt.savefig(ts_file)
    if workspace_dir:
        plt.savefig(os.path.join(workspace_dir, 'sample_drive_session.png'))
    plt.close()

    print(f"[Step 3] All 4 chart figures successfully generated and saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    out_dir = os.path.join(base_dir, "outputs")
    run_step_3(data_file, out_dir, workspace_dir=base_dir)
