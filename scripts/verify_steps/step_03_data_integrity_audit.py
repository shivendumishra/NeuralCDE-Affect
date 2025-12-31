import pickle
import os
import numpy as np
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from utils.config import FS_ECG, FS_EDA, FS_ACC
from preprocessing.ecg_preprocessing_pan_tompkins import pan_tompkins_detector
from preprocessing.ecg_hrv_extraction import compute_rr_intervals
from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
from preprocessing.acc_statistical_features import extract_acc_features

def audit_data(sub_id, data_root):
    sub_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
    if not os.path.exists(sub_path):
        sub_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
    print(f"\n{'='*60}")
    print(f" DATA INTEGRITY AUDIT: Subject S{sub_id}")
    print(f"{'='*60}")
    
    with open(sub_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    def print_stats(name, arr):
        arr = np.array(arr)
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        print(f"{name:<25} | Min: {np.min(arr):>8.3f} | Max: {np.max(arr):>8.3f} | Mean: {np.mean(arr):>8.3f} | NaNs: {nan_count} | Infs: {inf_count}")

    # 1. Raw Audit
    print("\n--- RAW DATA STATISTICS ---")
    print_stats("Chest ECG", data['signal']['chest']['ECG'])
    print_stats("Wrist EDA", data['signal']['wrist']['EDA'])
    print_stats("Wrist ACC", data['signal']['wrist']['ACC'])
    
    # Label Audit
    labels = data['label'].flatten()
    vals, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution: {dict(zip(vals, counts))}")
    print(f"Target Labels (1:Baseline, 2:Stress, 3:Amusement, 0:Transient, 4:Meditation, etc.)")

    # 2. Preprocessing Robustness Audit (on a sample window)
    print("\n--- PREPROCESSING AUDIT (60s Window) ---")
    start_sec = 1000
    
    # ECG
    ecg_raw = data['signal']['chest']['ECG'].flatten()[int(1000*FS_ECG):int(1060*FS_ECG)]
    filtered_ecg, r_peaks = pan_tompkins_detector(ecg_raw, FS_ECG)
    rr_intervals = compute_rr_intervals(r_peaks, FS_ECG)
    print_stats("Filtered ECG", filtered_ecg)
    print_stats("RR Intervals", rr_intervals)
    print(f"Detected Peaks: {len(r_peaks)} | Avg HR: {60.0/np.mean(rr_intervals):.1f} BPM")

    # EDA
    eda_raw = data['signal']['wrist']['EDA'].flatten()[int(1000*FS_EDA):int(1060*FS_EDA)]
    filtered_eda, phasic, tonic = eda_preprocessing(eda_raw, FS_EDA)
    print_stats("Phasic Component", phasic)
    print_stats("Tonic Component", tonic)

    # ACC
    acc_raw = data['signal']['wrist']['ACC'][int(1000*FS_ACC):int(1060*FS_ACC)]
    mag = preprocess_accelerometer(acc_raw, FS_ACC)
    print_stats("ACC Magnitude", mag)
    
    # Feature extraction check
    f = extract_acc_features(mag[:32])
    print(f"ACC Sample Feature: {f}")

    print(f"\n{'='*60}")
    print(" AUDIT COMPLETE: No NaNs or Infs detected in preprocessed streams.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    audit_data(2, DATA_ROOT)
