import pandas as pd
import numpy as np
import subprocess
import sys
from scipy.stats import ks_2samp

def detect_and_handle_drift(reference_path: str, current_path: str, p_value_threshold: float = 0.05):
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    numerical_cols = ref_df.select_dtypes(include=[np.number]).columns

    print(f"--- Running Data Drift Analysis ---")
    drift_detected = False

    for col in numerical_cols:
        ks_result = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
        p_val = ks_result.pvalue

        if p_val < p_value_threshold:
            print(f"🚨 DRIFT DETECTED in feature '{col}': p-value = {p_val:.5f}")
            drift_detected = True
        else:
            print(f"✅ Feature '{col}': No drift detected (p-value = {p_val:.5f})")

    # AUTOMATED TRIGGER
    if drift_detected:
        print("\n⚠️ WARNING: Data drift detected! Triggering automated retraining pipeline...")
        trigger_retraining(current_path)
    else:
        print("\n🎉 All features are operating within normal baseline limits. No retraining needed.")

def trigger_retraining(new_data_path: str):
    """Executes train.py as a subprocess passing the drifted dataset."""
    try:
        print("\nLaunching train.py via subprocess...")
        # Added encoding="utf-8" below
        result = subprocess.run(
            [sys.executable, "train.py"], 
            check=True, 
            capture_output=True, 
            text=True,
            encoding="utf-8"
        )
        print("=== RETRAINING STDOUT ===")
        print(result.stdout)
        print("========================")
        print("Retraining triggered and executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Retraining failed with error:\n{e.stderr}")

if __name__ == "__main__":
    # Test against drifted production batch
    detect_and_handle_drift("sensor_data.csv", "new_production_batch.csv")