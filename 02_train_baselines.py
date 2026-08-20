import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    models_dir = os.path.join(base_dir, "Models", "baseline")
    
    dirs = {
        "persistence": os.path.join(models_dir, "persistence"),
        "linear": os.path.join(models_dir, "linear_regression"),
        "rf": os.path.join(models_dir, "random_forest"),
        "xgb": os.path.join(models_dir, "xgboost"),
        "lgb": os.path.join(models_dir, "lightgbm"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Loading train and validation splits...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))

    # Filter active states only for SoH modeling
    train_active = train_df[train_df['is_active']].copy().reset_index(drop=True)
    val_active = val_df[val_df['is_active']].copy().reset_index(drop=True)
    
    features = [
        'Stack Current Sensor Value A',
        'Fuel Cell Total Voltage V',
        'Stack Coolant Temp Inlet degree C',
        'Stack Coolant Temp Outlet degree C',
        'Air Temp Stack Outlet degree C',
        'Air Flow Sensor kg/h',
        'Air Comp Motor Temp degree C',
        'Ambient Air Temp degree C',
        'Ambient Air Pressure kPa',
        'Fahrzeuggeschwindigkeit km/h',
        'Stack Power kW',
        'Air_Stoichiometry_Lambda',
        'Delta_V_Nernst',
        'E_cum_kWh',
        'high_current_duration_h',
        'Cumulative Time h'
    ]
    
    # Impute missing feature values with train medians for model stability
    feature_medians = train_active[features].median()
    X_train = train_active[features].fillna(feature_medians)
    y_train = train_active['soh_target']

    X_val = val_active[features].fillna(feature_medians)
    y_val = val_active['soh_target']

    print(f"Feature set size: {len(features)} columns")
    print(f"Train sample size: {len(X_train)} active rows")
    print(f"Val sample size: {len(X_val)} active rows")

    # Save feature list & medians for evaluation pipeline
    joblib.dump({"features": features, "medians": feature_medians}, os.path.join(models_dir, "feature_config.pkl"))

    # --- 1. Persistence Baseline ---
    print("\n[1/5] Training Persistence Baseline...")
    train_mean_soh = y_train.mean()
    joblib.dump({"train_mean_soh": train_mean_soh}, os.path.join(dirs["persistence"], "model.pkl"))

    # --- 2. Linear Regression (Ridge) ---
    print("[2/5] Training Ridge Linear Regression...")
    model_lr = Ridge(alpha=1.0)
    model_lr.fit(X_train, y_train)
    joblib.dump(model_lr, os.path.join(dirs["linear"], "model.pkl"))

    # --- 3. Random Forest Regressor ---
    print("[3/5] Training Random Forest Regressor...")
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model_rf.fit(X_train, y_train)
    joblib.dump(model_rf, os.path.join(dirs["rf"], "model.pkl"))

    # --- 4. XGBoost Regressor ---
    print("[4/5] Training XGBoost Regressor...")
    model_xgb = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model_xgb, os.path.join(dirs["xgb"], "model.pkl"))

    # --- 5. LightGBM Regressor ---
    print("[5/5] Training LightGBM Regressor...")
    model_lgb = LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    joblib.dump(model_lgb, os.path.join(dirs["lgb"], "model.pkl"))

    print("\nAll 5 baseline models trained and saved to Models/baseline/!")

if __name__ == "__main__":
    main()
