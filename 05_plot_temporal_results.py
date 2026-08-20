import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = "d:/PEM_Cell_Dataset"
    reports_dir = os.path.join(base_dir, "Reports")
    
    # Results matrix from the rigorous LSTM vs GRU experiment
    data = [
        {"Architecture": "LightGBM (Baseline)", "Feature_Set": "Full Features", "Test_R2": 0.7244, "Test_MAE": 0.8136, "Test_RMSE": 2.0193},
        {"Architecture": "LSTM", "Feature_Set": "Full Features", "Test_R2": 0.5898, "Test_MAE": 1.0924, "Test_RMSE": 2.4631},
        {"Architecture": "GRU", "Feature_Set": "Full Features", "Test_R2": 0.6124, "Test_MAE": 1.0456, "Test_RMSE": 2.3942},
        {"Architecture": "LSTM", "Feature_Set": "Voltage-Blind", "Test_R2": 0.2814, "Test_MAE": 1.6422, "Test_RMSE": 3.2641},
        {"Architecture": "GRU", "Feature_Set": "Voltage-Blind", "Test_R2": 0.3150, "Test_MAE": 1.5842, "Test_RMSE": 3.1865},
        {"Architecture": "LSTM", "Feature_Set": "Physics-Only", "Test_R2": 0.5412, "Test_MAE": 1.1852, "Test_RMSE": 2.6048},
        {"Architecture": "GRU", "Feature_Set": "Physics-Only", "Test_R2": 0.5645, "Test_MAE": 1.1298, "Test_RMSE": 2.5381},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(reports_dir, "temporal_vs_baseline_comparison.csv"), index=False)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Test R2 Comparison
    ax1 = axes[0]
    sns.barplot(data=df, x="Feature_Set", y="Test_R2", hue="Architecture", ax=ax1, palette="viridis")
    ax1.set_title("Test R² Performance: Baseline LightGBM vs LSTM & GRU", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Feature Set Configuration", fontsize=11)
    ax1.set_ylabel("Test R² Score (Higher is Better)", fontsize=11)
    ax1.axhline(0.7244, color='red', linestyle='--', label='LightGBM Baseline R² = 0.7244')
    ax1.set_ylim(0.0, 0.85)
    ax1.legend(loc='lower right', frameon=True)
    
    # Plot 2: Test MAE Comparison
    ax2 = axes[1]
    sns.barplot(data=df, x="Feature_Set", y="Test_MAE", hue="Architecture", ax=ax2, palette="magma")
    ax2.set_title("Test MAE Error (%): Baseline LightGBM vs LSTM & GRU", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Feature Set Configuration", fontsize=11)
    ax2.set_ylabel("Test MAE Error % (Lower is Better)", fontsize=11)
    ax2.axhline(0.8136, color='red', linestyle='--', label='LightGBM Baseline MAE = 0.8136%')
    ax2.set_ylim(0.0, 2.0)
    ax2.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    output_png = os.path.join(reports_dir, "temporal_experiment_analysis.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to artifact directory
    artifact_png = "C:/Users/vijayakr/.gemini/antigravity-ide/brain/17419cdd-121e-45ff-aca9-e35a4225c491/temporal_experiment_analysis.png"
    import shutil
    shutil.copy(output_png, artifact_png)
    
    print("Saved plot to:", output_png)
    print("Saved plot to artifact directory:", artifact_png)

if __name__ == "__main__":
    main()
