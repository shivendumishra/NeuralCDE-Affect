import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from utils.config import FS_ECG, FS_EDA, FS_ACC
from preprocessing.ecg_preprocessing_pan_tompkins import pan_tompkins_detector
from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
from preprocessing.acc_statistical_features import extract_acc_features

def verify_preprocessing(sub_id, data_root, output_dir):
    sub_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
    if not os.path.exists(sub_path):
        sub_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
    print(f"Loading {sub_path} for preprocessing check...")
    with open(sub_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    start_sec = 1000 
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. ECG: Comparison and Preprocessed ---
    duration_ecg = 10
    ecg_raw = data['signal']['chest']['ECG'].flatten()
    s_idx = int(start_sec * FS_ECG)
    e_idx = s_idx + int(duration_ecg * FS_ECG)
    ecg_window = ecg_raw[s_idx:e_idx]
    
    # Preprocess
    filtered_ecg, r_peaks = pan_tompkins_detector(ecg_window, FS_ECG)
    
    # Graph 1.1: ECG Comparison (Raw vs Filtered + Peaks)
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    t_ecg = np.arange(len(ecg_window))/FS_ECG
    plt.plot(t_ecg, ecg_window, label='Raw ECG', color='gray', alpha=0.5)
    plt.plot(t_ecg, filtered_ecg, label='Filtered ECG (Bandpass+Notch)', color='#e74c3c')
    plt.title(f'S{sub_id} ECG: Raw vs Filtered')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t_ecg, filtered_ecg, color='#e74c3c')
    plt.scatter(r_peaks/FS_ECG, filtered_ecg[r_peaks], color='black', marker='x', label='Detected R-peaks')
    plt.title('Filtered ECG with Detected R-peaks (Pan-Tompkins)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ECG_Comparison_and_Peaks.png'), dpi=200)
    plt.close()

    # --- 2. EDA: Comparison and Preprocessed ---
    duration_eda = 60 # EDA needs longer window to see components
    eda_raw = data['signal']['wrist']['EDA'].flatten()
    s_idx = int(start_sec * FS_EDA)
    e_idx = s_idx + int(duration_eda * FS_EDA)
    eda_window = eda_raw[s_idx:e_idx]
    
    # Preprocess
    filtered_eda, phasic, tonic = eda_preprocessing(eda_window, FS_EDA)
    
    # Graph 2.1: EDA Comparison (Raw vs Tonic vs Phasic)
    plt.figure(figsize=(15, 8))
    t_eda = np.arange(len(eda_window))/FS_EDA
    plt.subplot(2, 1, 1)
    plt.plot(t_eda, eda_window, label='Raw EDA', color='gray', alpha=0.5)
    plt.plot(t_eda, filtered_eda, label='Filtered (1Hz LP)', color='black', linestyle='--')
    plt.plot(t_eda, tonic, label='Tonic (SCL)', color='#2ecc71', linewidth=2)
    plt.title(f'S{sub_id} EDA: Raw vs Tonic component')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t_eda, phasic, label='Phasic (SCR)', color='#3498db')
    plt.title('Phasic Component (SCR)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'EDA_Preprocessing_Comparison.png'), dpi=200)
    plt.close()

    # --- 3. ACC: Comparison and Preprocessed ---
    duration_acc = 30
    acc_raw = data['signal']['wrist']['ACC'] # (N, 3)
    s_idx = int(start_sec * FS_ACC)
    e_idx = s_idx + int(duration_acc * FS_ACC)
    acc_window = acc_raw[s_idx:e_idx]
    
    # Preprocess
    mag = preprocess_accelerometer(acc_window, FS_ACC)
    
    # Extract features over time to show "preprocessed" data (as used in model)
    winsize = 32 # 1 second if FS=32
    acc_features = { 'mean': [], 'var': [], 'energy': [] }
    for i in range(0, len(mag), winsize):
        w = mag[i:i+winsize]
        if len(w) < winsize: break
        f = extract_acc_features(w)
        acc_features['mean'].append(f['acc_mean'])
        acc_features['var'].append(f['acc_var'])
        acc_features['energy'].append(f['acc_energy'])

    # Graph 3.1: ACC Magnitude (Comparison)
    plt.figure(figsize=(15, 8))
    t_acc = np.arange(len(acc_window))/FS_ACC
    plt.subplot(2, 1, 1)
    plt.plot(t_acc, acc_window[:, 0], label='Raw X', alpha=0.4)
    plt.plot(t_acc, acc_window[:, 1], label='Raw Y', alpha=0.4)
    plt.plot(t_acc, acc_window[:, 2], label='Raw Z', alpha=0.4)
    plt.plot(t_acc, mag, label='Resultant Mag (High-pass)', color='black', linewidth=1.5)
    plt.title(f'S{sub_id} ACC: 3-Axis vs Resultant Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    t_feat = np.arange(len(acc_features['mean']))
    plt.plot(t_feat, acc_features['mean'], label='Mean', marker='o')
    plt.plot(t_feat, acc_features['var'], label='Variance', marker='s')
    plt.title('Statistical Features (Moving Window)')
    plt.xlabel('Window (1s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ACC_Preprocessing_Comparison.png'), dpi=200)
    plt.close()

    print(f"Preprocessing visualization completed. Results in {output_dir}")

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_02_preprocessing_visualization')
    verify_preprocessing(2, DATA_ROOT, OUTPUT_DIR)
