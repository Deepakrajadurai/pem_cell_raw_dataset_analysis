import pandas as pd
import numpy as np

df = pd.read_csv('d:/PEM_Cell_Dataset/Processed/processed.csv')
df['Time'] = pd.to_datetime(df['Time'])

print("Head of Kumulative Betriebszeit h:")
print(df[['Time', 'Kumulative Betriebszeit h', 'Cumulative Time h', 'session_id']].dropna().head(10))

print("\nTail of Kumulative Betriebszeit h:")
print(df[['Time', 'Kumulative Betriebszeit h', 'Cumulative Time h', 'session_id']].dropna().tail(10))

# Check jumps in Kumulative Betriebszeit h across sessions
session_stats = df.groupby('session_id').agg(
    start_time=('Time', 'min'),
    end_time=('Time', 'max'),
    start_kb=('Kumulative Betriebszeit h', 'min'),
    end_kb=('Kumulative Betriebszeit h', 'max'),
    start_cum_h=('Cumulative Time h', 'min'),
    end_cum_h=('Cumulative Time h', 'max'),
    rows=('session_id', 'count')
)

print("\nFirst 5 sessions:")
print(session_stats.head())

print("\nLast 5 sessions:")
print(session_stats.tail())
