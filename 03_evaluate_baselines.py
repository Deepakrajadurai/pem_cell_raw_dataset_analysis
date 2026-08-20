import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import skew, kurtosis

def calc_metrics(y_true, y_pred):
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    if len(y_t) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0, "MAPE_%": 0.0}
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}

def calc_residuals(y_true, y_pred):
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    res = y_true[valid_mask] - y_pred[valid_mask]
    if len(res) == 0:
        return {"mean_residual": 0.0, "std_residual": 0.0, "skewness": 0.0, "kurtosis": 0.0, "min_residual": 0.0, "max_residual": 0.0}
    return {
        "mean_residual": float(np.mean(res)),
        "std_residual": float(np.std(res)),
        "skewness": float(skew(res)),
        "kurtosis": float(kurtosis(res)),
        "min_residual": float(np.min(res)),
        "max_residual": float(np.max(res))
    }

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    splits_dir = os.path.join(base_dir, "Splits")
    models_dir = os.path.join(base_dir, "Models", "baseline")
    reports_dir = os.path.join(base_dir, "Reports")
    soh_models_dir = os.path.join(base_dir, "Models", "soh")
    
    os.makedirs(soh_models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    config = joblib.load(os.path.join(models_dir, "feature_config.pkl"))
    features = config["features"]
    medians = config["medians"]

    print("Loading validation and test splits...")
    val_df = pd.read_parquet(os.path.join(splits_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(splits_dir, "test.parquet"))

    val_active = val_df[val_df['is_active'] & val_df['soh_target'].notna()].copy().reset_index(drop=True)
    test_active = test_df[test_df['is_active'] & test_df['soh_target'].notna()].copy().reset_index(drop=True)

    X_val = val_active[features].fillna(medians)
    y_val = val_active['soh_target']

    X_test = test_active[features].fillna(medians)
    y_test = test_active['soh_target']

    # Load models
    models = {
        "Persistence": joblib.load(os.path.join(models_dir, "persistence", "model.pkl")),
        "Linear Regression": joblib.load(os.path.join(models_dir, "linear_regression", "model.pkl")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest", "model.pkl")),
        "XGBoost": joblib.load(os.path.join(models_dir, "xgboost", "model.pkl")),
        "LightGBM": joblib.load(os.path.join(models_dir, "lightgbm", "model.pkl")),
    }

    metrics_rows = []
    residual_rows = []
    results_json = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Predict
        if name == "Persistence":
            mean_val = model["train_mean_soh"]
            pred_val = np.full(len(y_val), mean_val)
            pred_test = np.full(len(y_test), mean_val)
        else:
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            
        val_m = calc_metrics(y_val, pred_val)
        test_m = calc_metrics(y_test, pred_test)
        
        val_res = calc_residuals(y_val, pred_val)
        test_res = calc_residuals(y_test, pred_test)

        # Regimes evaluation on Test set
        bins = [0, 50, 150, 300]
        regime_labels = ['Low Load (0-50A)', 'Medium Load (50-150A)', 'High Load (>150A)']
        test_active['curr_regime'] = pd.cut(test_active['Stack Current Sensor Value A'], bins=bins, labels=regime_labels)
        
        regime_metrics = {}
        for reg in regime_labels:
            mask = test_active['curr_regime'] == reg
            if mask.sum() > 10:
                regime_metrics[reg] = calc_metrics(y_test[mask], pred_test[mask])

        metrics_rows.append({
            "Model": name,
            "Val_MAE": val_m["MAE"], "Val_RMSE": val_m["RMSE"], "Val_R2": val_m["R2"], "Val_MAPE_%": val_m["MAPE_%"],
            "Test_MAE": test_m["MAE"], "Test_RMSE": test_m["RMSE"], "Test_R2": test_m["R2"], "Test_MAPE_%": test_m["MAPE_%"]
        })

        res_entry = {"Model": name}
        res_entry.update({f"Test_{k}": v for k, v in test_res.items()})
        residual_rows.append(res_entry)

        results_json[name] = {
            "validation_metrics": val_m,
            "test_metrics": test_m,
            "test_residuals": test_res,
            "test_regime_metrics": regime_metrics
        }

    # Save outputs
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(reports_dir, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    residual_df = pd.DataFrame(residual_rows)
    residual_csv = os.path.join(reports_dir, "residual_analysis.csv")
    residual_df.to_csv(residual_csv, index=False)

    json_path = os.path.join(soh_models_dir, "baseline_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)

    print("\n--- BASELINE EVALUATION METRICS TABLE ---")
    print(metrics_df.to_string(index=False))
    
    print(f"\nSaved baseline metrics to: {metrics_csv}")
    print(f"Saved residual analysis to: {residual_csv}")
    print(f"Saved baseline results JSON to: {json_path}")

if __name__ == "__main__":
    main()
