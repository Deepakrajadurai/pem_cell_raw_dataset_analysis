"""
Step 2: Deep Time-Series Analysis, Session Segmentation, Correlation & Polarization
Dataset: PEM Fuel Cell Raw Time Series Data (raw_dataset.csv)
"""

import os
import json
import pandas as pd
import numpy as np

def run_step_2(data_path, output_json_path):
    print(f"[Step 2] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df['Time_dt'] = pd.to_datetime(df['Time'], utc=True)
    df = df.sort_values('Time_dt').reset_index(drop=True)

    time_diffs = df['Time_dt'].diff().dt.total_seconds()
    session_starts = (time_diffs > 300) | (time_diffs.isna())
    df['session_id'] = session_starts.cumsum()

    session_stats = df.groupby('session_id').agg(
        start_time=('Time_dt', 'min'),
        end_time=('Time_dt', 'max'),
        sample_count=('Time_dt', 'count'),
        max_current=('Stack Current Sensor Value A', 'max'),
        avg_speed=('Fahrzeuggeschwindigkeit km/h', 'mean'),
        max_speed=('Fahrzeuggeschwindigkeit km/h', 'max')
    )
    session_stats['duration_min'] = (session_stats['end_time'] - session_stats['start_time']).dt.total_seconds() / 60.0
    valid_sessions = session_stats[session_stats['sample_count'] > 10]

    num_cols = [
        'Fuel Cell Total Voltage V',
        'Stack Current Sensor Value A',
        'Stack Coolant Temp Inlet degree C',
        'Stack Coolant Temp Outlet degree C',
        'Air Temp Stack Outlet degree C',
        'Air Flow Sensor kg/h',
        'Air Comp Motor Temp degree C',
        'Ambient Air Temp degree C',
        'Ambient Air Pressure kPa',
        'Fahrzeuggeschwindigkeit km/h'
    ]
    corr_matrix = df[num_cols].corr().round(4).to_dict()

    valid_iv = df[(df['Fuel Cell Total Voltage V'] > 0) & (df['Stack Current Sensor Value A'] >= 0)].copy()
    valid_iv['current_bin'] = pd.cut(valid_iv['Stack Current Sensor Value A'], bins=np.arange(0, 270, 10))
    
    polarization = valid_iv.groupby('current_bin', observed=False).agg(
        mean_v=('Fuel Cell Total Voltage V', 'mean'),
        std_v=('Fuel Cell Total Voltage V', 'std'),
        count=('Fuel Cell Total Voltage V', 'count')
    ).reset_index()

    polarization['current_bin_str'] = polarization['current_bin'].astype(str)
    pol_list = []
    for _, row in polarization.iterrows():
        if row['count'] > 0:
            pol_list.append({
                "current_bin": row['current_bin_str'],
                "mean_voltage_v": round(float(row['mean_v']), 2),
                "std_voltage_v": round(float(row['std_v']), 2),
                "sample_count": int(row['count'])
            })

    anomalies = {
        "voltage_negative_or_zero": int((df['Fuel Cell Total Voltage V'] <= 0).sum()),
        "voltage_gt_400V": int((df['Fuel Cell Total Voltage V'] > 400.0).sum()),
        "current_negative": int((df['Stack Current Sensor Value A'] < 0).sum()),
        "coolant_inlet_lt_0C": int((df['Stack Coolant Temp Inlet degree C'] < 0).sum()),
        "coolant_outlet_gt_80C": int((df['Stack Coolant Temp Outlet degree C'] > 80.0).sum()),
        "comp_motor_temp_lt_minus20C": int((df['Air Comp Motor Temp degree C'] < -20.0).sum()),
        "comp_motor_temp_gt_100C": int((df['Air Comp Motor Temp degree C'] > 100.0).sum()),
    }

    res = {
        "step": "step02_deep_analysis",
        "session_segmentation": {
            "total_sessions": len(session_stats),
            "active_sessions_gt_10_samples": len(valid_sessions),
            "mean_session_duration_min": round(float(valid_sessions['duration_min'].mean()), 2),
            "median_session_duration_min": round(float(valid_sessions['duration_min'].median()), 2),
            "max_session_duration_min": round(float(valid_sessions['duration_min'].max()), 2)
        },
        "correlation_matrix": corr_matrix,
        "polarization_curve": pol_list,
        "anomalies_audit": anomalies
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"[Step 2] Completed successfully. Saved to {output_json_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    out_file = os.path.join(base_dir, "outputs", "deep_analysis.json")
    run_step_2(data_file, out_file)
