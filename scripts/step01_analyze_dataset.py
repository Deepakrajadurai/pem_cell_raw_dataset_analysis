"""
Step 1: Dataset Profiling & Summary Statistics
Dataset: PEM Fuel Cell Raw Time Series Data (raw_dataset.csv)
"""

import os
import json
import pandas as pd
import numpy as np

def run_step_1(data_path, output_json_path):
    print(f"[Step 1] Loading raw dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[Step 1] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    df['Time_dt'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time_dt').reset_index(drop=True)

    start_time = df['Time_dt'].min()
    end_time = df['Time_dt'].max()
    duration = end_time - start_time
    time_diffs = df['Time_dt'].diff().dt.total_seconds()

    if 'Fuel Cell Total Voltage V' in df and 'Stack Current Sensor Value A' in df:
        df['Stack Power kW'] = (df['Fuel Cell Total Voltage V'] * df['Stack Current Sensor Value A']) / 1000.0

    if 'Stack Coolant Temp Outlet degree C' in df and 'Stack Coolant Temp Inlet degree C' in df:
        df['Coolant Delta T C'] = df['Stack Coolant Temp Outlet degree C'] - df['Stack Coolant Temp Inlet degree C']

    numeric_cols = [c for c in df.columns if c not in ['Time', 'Time_dt']]
    col_stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        missing_cnt = int(df[col].isna().sum())
        missing_pct = float(missing_cnt / len(df) * 100.0)
        if len(s) > 0:
            col_stats[col] = {
                "count": int(len(s)),
                "missing_count": missing_cnt,
                "missing_percent": round(missing_pct, 4),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "p25": round(float(s.quantile(0.25)), 4),
                "median": round(float(s.median()), 4),
                "p75": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurt": round(float(s.kurtosis()), 4)
            }

    regimes = {}
    if 'Stack Current Sensor Value A' in df:
        curr = df['Stack Current Sensor Value A']
        regimes = {
            "off_idle_current_le_1A": int((curr <= 1.0).sum()),
            "low_load_1A_to_50A": int(((curr > 1.0) & (curr <= 50.0)).sum()),
            "medium_load_50A_to_150A": int(((curr > 50.0) & (curr <= 150.0)).sum()),
            "high_load_gt_150A": int((curr > 150.0).sum())
        }

    energy_kwh = 0.0
    if 'Stack Power kW' in df:
        time_diff_hours = time_diffs / 3600.0
        energy_kwh = float((df['Stack Power kW'] * time_diff_hours).sum())

    summary = {
        "step": "step01_analyze_dataset",
        "total_rows": int(len(df)),
        "total_cols": int(len(df.columns) - 1),
        "start_time": str(start_time),
        "end_time": str(end_time),
        "duration_days": round(float(duration.total_seconds() / 86400.0), 2),
        "duration_hours": round(float(duration.total_seconds() / 3600.0), 2),
        "sampling_interval_sec": {
            "mean": round(float(time_diffs.mean()), 2),
            "median": round(float(time_diffs.median()), 2),
            "min": round(float(time_diffs.min()), 2),
            "max": round(float(time_diffs.max()), 2),
            "mode": round(float(time_diffs.mode()[0]), 2) if len(time_diffs.mode()) > 0 else None
        },
        "operating_regimes": regimes,
        "total_energy_kwh": round(energy_kwh, 2),
        "column_stats": col_stats
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[Step 1] Completed successfully. Summary saved to {output_json_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    out_file = os.path.join(base_dir, "outputs", "analysis_summary.json")
    run_step_1(data_file, out_file)
