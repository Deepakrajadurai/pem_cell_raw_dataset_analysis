import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    models_dir = os.path.join(base_dir, "Models", "baseline")
    reports_dir = os.path.join(base_dir, "Reports")

    config = joblib.load(os.path.join(models_dir, "feature_config.pkl"))
    features = config["features"]
    medians = config["medians"]

    print("Loading Train, Validation, and Test splits for plotting...")
    train_df = pd.read_parquet(os.path.join(splits_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    active_df = full_df[full_df['is_active']].copy().sort_values('Cumulative Time h').reset_index(drop=True)

    X_full = active_df[features].fillna(medians)
    y_full = active_df['soh_target']

    # Load top baseline models
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost", "model.pkl"))
    lgb_model = joblib.load(os.path.join(models_dir, "lightgbm", "model.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest", "model.pkl"))

    active_df['pred_xgb'] = xgb_model.predict(X_full)
    active_df['pred_lgb'] = lgb_model.predict(X_full)
    active_df['pred_rf'] = rf_model.predict(X_full)

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Cumulative Time vs Actual & Predicted SoH Trajectory ---
    ax1 = axes[0, 0]
    ax1.plot(active_df['Cumulative Time h'], active_df['soh_target'], label='Actual SoH Target', color='black', alpha=0.6, linewidth=1.5)
    ax1.plot(active_df['Cumulative Time h'], active_df['pred_xgb'], label='XGBoost Prediction', color='#1f77b4', linestyle='--', linewidth=1.2)
    ax1.plot(active_df['Cumulative Time h'], active_df['pred_lgb'], label='LightGBM Prediction', color='#2ca02c', linestyle=':', linewidth=1.2)

    # Split boundaries vertical lines
    train_max_t = train_df[train_df['is_active']]['Cumulative Time h'].max()
    val_max_t = val_df[val_df['is_active']]['Cumulative Time h'].max()
    
    ax1.axvline(train_max_t, color='gray', linestyle='--', label='Train/Val Split Boundary')
    ax1.axvline(val_max_t, color='red', linestyle='--', label='Val/Test Split Boundary')

    ax1.set_title("PEM Fuel Cell SoH Trajectory & Baseline Forecasts Over Operating Time", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Active Cumulative Operating Time (hours)", fontsize=10)
    ax1.set_ylabel("State of Health (%)", fontsize=10)
    ax1.legend(loc='lower left', frameon=True)

    # --- Plot 2: Test Set Residual Density Distributions ---
    ax2 = axes[0, 1]
    test_active = test_df[test_df['is_active']].copy().reset_index(drop=True)
    X_test = test_active[features].fillna(medians)
    y_test = test_active['soh_target']

    res_xgb = y_test - xgb_model.predict(X_test)
    res_lgb = y_test - lgb_model.predict(X_test)
    res_rf = y_test - rf_model.predict(X_test)

    sns.kdeplot(res_xgb, ax=ax2, label=f'XGBoost Residuals (std={np.std(res_xgb):.2f})', fill=True, alpha=0.3, color='#1f77b4')
    sns.kdeplot(res_lgb, ax=ax2, label=f'LightGBM Residuals (std={np.std(res_lgb):.2f})', fill=True, alpha=0.3, color='#2ca02c')
    sns.kdeplot(res_rf, ax=ax2, label=f'Random Forest Residuals (std={np.std(res_rf):.2f})', fill=True, alpha=0.3, color='#ff7f0e')

    ax2.set_title("Test Set SoH Prediction Residual Distributions", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Residual Error (Actual - Predicted %)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.legend(loc='upper right', frameon=True)

    # --- Plot 3: Predicted vs Actual Scatter Plot (XGBoost Test Set) ---
    ax3 = axes[1, 0]
    ax3.scatter(y_test, xgb_model.predict(X_test), alpha=0.3, color='#1f77b4', s=15, label='Test Observations')
    lims = [min(y_test.min(), xgb_model.predict(X_test).min()), max(y_test.max(), xgb_model.predict(X_test).max())]
    ax3.plot(lims, lims, color='red', linestyle='--', label='1:1 Identity Line')
    
    ax3.set_title("XGBoost Test Set Forecast vs Actual SoH Target", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Actual SoH (%)", fontsize=10)
    ax3.set_ylabel("Predicted SoH (%)", fontsize=10)
    ax3.legend(loc='upper left', frameon=True)

    # --- Plot 4: Feature Importance Top 10 (XGBoost) ---
    ax4 = axes[1, 1]
    importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True)
    importances.tail(10).plot(kind='barh', ax=ax4, color='#1f77b4')

    ax4.set_title("Top 10 Feature Importances (XGBoost Model)", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Relative Feature Importance", fontsize=10)

    plt.tight_layout()
    output_png = os.path.join(reports_dir, "degradation_analysis.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated degradation analysis plot: {output_png}")

if __name__ == "__main__":
    main()
