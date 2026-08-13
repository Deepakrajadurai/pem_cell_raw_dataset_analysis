"""
Master Pipeline Runner & Backtesting Script
Executes Step 1, Step 2, Step 3, and Step 4 sequentially.
"""

import os
import sys
import time

from step01_analyze_dataset import run_step_1
from step02_deep_analysis import run_step_2
from step03_generate_charts import run_step_3
from step04_health_indicator import run_step_4

def main():
    start_total = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "raw_dataset.csv")
    outputs_dir = os.path.join(base_dir, "outputs")

    print("=================================================================")
    print("      PEM FUEL CELL DATASET ANALYTICS PIPELINE & BACKTESTER      ")
    print("=================================================================")
    print(f"Base Workspace: {base_dir}")
    print(f"Input Dataset:  {data_file}")
    print(f"Outputs Folder: {outputs_dir}\n")

    # Step 1: Profiling
    t1 = time.time()
    summary_json = os.path.join(outputs_dir, "analysis_summary.json")
    run_step_1(data_file, summary_json)
    print(f"--> Step 1 completed in {time.time() - t1:.2f}s\n")

    # Step 2: Deep Analysis
    t2 = time.time()
    deep_json = os.path.join(outputs_dir, "deep_analysis.json")
    run_step_2(data_file, deep_json)
    print(f"--> Step 2 completed in {time.time() - t2:.2f}s\n")

    # Step 3: Chart Generation
    t3 = time.time()
    run_step_3(data_file, outputs_dir, workspace_dir=base_dir)
    print(f"--> Step 3 completed in {time.time() - t3:.2f}s\n")

    # Step 4: Health Indicator & Degradation Model
    t4 = time.time()
    run_step_4(data_file, outputs_dir, workspace_dir=base_dir)
    print(f"--> Step 4 completed in {time.time() - t4:.2f}s\n")

    total_time = time.time() - start_total
    print("=================================================================")
    print(f"   PIPELINE EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!   ")
    print("=================================================================")

if __name__ == "__main__":
    main()
