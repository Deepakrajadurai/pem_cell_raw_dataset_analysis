import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('d:/PEM_Cell_Dataset/Processed/processed.csv')
print(f"Total rows: {len(df)}")
df['Time'] = pd.to_datetime(df['Time'])

# Filter valid electrical readings
valid_df = df[(df['Fuel Cell Total Voltage V'] > 50) & (df['Stack Current Sensor Value A'] >= 0)].copy()

# Calculate Power
valid_df['Power_kW'] = (valid_df['Fuel Cell Total Voltage V'] * valid_df['Stack Current Sensor Value A']) / 1000.0

print(f"Valid operational rows: {len(valid_df)}")
print(f"Time range: {df['Time'].min()} to {df['Time'].max()}")
print(f"Cumulative time range: {df['Cumulative Time h'].min():.2f} h to {df['Cumulative Time h'].max():.2f} h")
print(f"Operating hours (Kumulative Betriebszeit): {df['Kumulative Betriebszeit h'].min():.1f} h to {df['Kumulative Betriebszeit h'].max():.1f} h")

# Analyze voltage at specific current levels over time
current_bins = [
    ('Low Load (40-60A)', 40, 60),
    ('Medium Load (90-110A)', 90, 110),
    ('High Load (140-160A)', 140, 160)
]

for name, i_min, i_max in current_bins:
    sub = valid_df[(valid_df['Stack Current Sensor Value A'] >= i_min) & (valid_df['Stack Current Sensor Value A'] <= i_max)]
    if len(sub) > 100:
        # Fit linear regression against Cumulative Time h
        valid_sub = sub.dropna(subset=['Cumulative Time h', 'Fuel Cell Total Voltage V'])
        if len(valid_sub) > 50:
            poly = np.polyfit(valid_sub['Cumulative Time h'], valid_sub['Fuel Cell Total Voltage V'], 1)
            decay_rate_per_hour = poly[0] * 1000.0 # mV/hour
            initial_v = poly[1]
            print(f"[{name}] Samples: {len(valid_sub)}, Slope: {decay_rate_per_hour:.3f} mV/h ({decay_rate_per_hour*1000:.1f} uV/h), Intercept: {initial_v:.2f} V")

# Check internal resistance calculation R_est = dV / dI during steady state or dynamic steps
# Calculate degradation rate using energy throughput
valid_df['dt_h'] = valid_df['Time'].diff().dt.total_seconds() / 3600.0
valid_df.loc[valid_df['dt_h'] > (5.0/60.0), 'dt_h'] = 0 # zero out gaps > 5 mins
valid_df['Energy_kWh'] = (valid_df['Power_kW'] * valid_df['dt_h']).cumsum()

print(f"Total calculated energy output: {valid_df['Energy_kWh'].max():.2f} kWh")

# Bin by time quarters to show progressive change in polarization curves
valid_df['Time_Quarter'] = pd.qcut(valid_df['Cumulative Time h'], 4, labels=['Q1 (Early)', 'Q2 (Mid-Early)', 'Q3 (Mid-Late)', 'Q4 (Late)'])

print("\nMean Voltage by Current Range and Time Quarter:")
pv = valid_df.groupby(['Time_Quarter', pd.cut(valid_df['Stack Current Sensor Value A'], bins=[0, 10, 50, 100, 150, 200, 255])])['Fuel Cell Total Voltage V'].agg(['mean', 'count'])
print(pv)
