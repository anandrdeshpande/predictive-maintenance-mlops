import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def detect_drift(reference_path: str, current_path: str, p_value_threshold: float = 0.05):
    """
    Compares baseline (training) data against current (production) data 
    to detect statistical feature drift.
    """
    # 1. Load Datasets
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    # Filter numerical columns (sensor features)
    numerical_cols = ref_df.select_dtypes(include=[np.number]).columns

    print(f"--- Running Data Drift Analysis ---")
    drift_detected = False

    for col in numerical_cols:
        # Perform 2-sample Kolmogorov-Smirnov test
        ks_result = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
        p_val = ks_result.pvalue

        if p_val < p_value_threshold:
            print(f"🚨 DRIFT DETECTED in feature '{col}': p-value = {p_val:.5f} (Threshold < {p_value_threshold})")
            drift_detected = True
        else:
            print(f"✅ Feature '{col}': No drift detected (p-value = {p_val:.5f})")

    if drift_detected:
        print("\n⚠️ WARNING: Incoming production data has drifted. Model retraining recommended!")
    else:
        print("\n🎉 All features are operating within normal baseline limits.")

if __name__ == "__main__":
    # Test baseline vs. baseline (Should show NO drift)
    detect_drift("sensor_data.csv", "sensor_data.csv")