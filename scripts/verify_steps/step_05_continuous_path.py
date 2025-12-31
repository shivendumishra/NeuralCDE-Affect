import torch
import torchcde
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
from preprocessing.ecg_hrv_extraction import compute_rr_intervals
from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
from continuous_path.cubic_spline_construction import build_spline
from normalization.intensity_channel import add_intensity_channel

def verify_continuous_path(sub_id, data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and Preprocess sample window
    sub_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
    if not os.path.exists(sub_path):
        sub_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
    with open(sub_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    start_sec = 1000 
    duration = 20
    
    # ECG -> RR
    ecg_raw = data['signal']['chest']['ECG'].flatten()[int(start_sec*FS_ECG):int((start_sec+duration)*FS_ECG)]
    _, r_peaks = pan_tompkins_detector(ecg_raw, FS_ECG)
    rr_intervals = compute_rr_intervals(r_peaks, FS_ECG)
    r_times = r_peaks[1:] / FS_ECG 
    
    # EDA
    eda_raw = data['signal']['wrist']['EDA'].flatten()[int(start_sec*FS_EDA):int((start_sec+duration)*FS_EDA)]
    _, phasic, tonic = eda_preprocessing(eda_raw, FS_EDA)
    t_eda = np.arange(len(eda_raw))/FS_EDA

    # --- SIMULATE TRAINING PIPELINE ---
    # Convert to tensors
    rr_tensor = torch.tensor(rr_intervals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # (1, L, 1)
    rr_times_tensor = torch.tensor(r_times, dtype=torch.float32).unsqueeze(0) # (1, L)
    
    # Build Spline for ECG
    # We must add intensity channel first as in train_loso.py
    rr_augmented = add_intensity_channel(rr_tensor, rr_times_tensor)
    rr_path = build_spline(rr_augmented)
    
    # Evaluate at high resolution
    t_eval = torch.linspace(0, duration, 500)
    # Note: build_spline assumes the control points are defined over indices 0 to L-1 
    # if times are not handled inside build_spline's natural_cubic_coeffs.
    # In cubic_spline_construction.py: coeffs = torchcde.natural_cubic_coeffs(data)
    # It treats the indices of augmented_data as the time-like grid.
    
    L_rr = rr_tensor.shape[1]
    t_index_eval = torch.linspace(0, L_rr - 1, 500)
    rr_continuous = rr_path.evaluate(t_index_eval)

    # --- PLOT 1: ECG Continuous Path ---
    plt.figure(figsize=(15, 6))
    plt.scatter(np.arange(L_rr), rr_intervals, color='red', label='Discrete RR Points', zorder=5)
    plt.plot(t_index_eval.numpy(), rr_continuous[0, :, 0].detach().numpy(), color='black', label='Cubic Spline Path')
    plt.title(f'Subject S{sub_id} - ECG Continuous Path (Spline Interpolation)')
    plt.xlabel('Point Index')
    plt.ylabel('RR Interval (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'Step_05_ECG_Continuous_Path.png'), dpi=200)
    plt.close()

    # --- PLOT 2: EDA Continuous Path ---
    eda_tensor = torch.tensor(phasic, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    eda_times_tensor = torch.tensor(t_eda, dtype=torch.float32).unsqueeze(0)
    eda_augmented = add_intensity_channel(eda_tensor, eda_times_tensor)
    eda_path = build_spline(eda_augmented)
    
    L_eda = eda_tensor.shape[1]
    t_index_eval_eda = torch.linspace(0, L_eda - 1, 1000)
    eda_continuous = eda_path.evaluate(t_index_eval_eda)
    
    plt.figure(figsize=(15, 6))
    plt.scatter(np.arange(L_eda), phasic, color='#3498db', s=10, label='Discrete Phasic Dots', alpha=0.5)
    plt.plot(t_index_eval_eda.numpy(), eda_continuous[0, :, 0].detach().numpy(), color='black', linewidth=1, label='Smooth Spline Path')
    plt.title(f'Subject S{sub_id} - EDA Phasic Continuous Path')
    plt.xlabel('Sample Index (4Hz)')
    plt.ylabel('Phasic Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'Step_05_EDA_Continuous_Path.png'), dpi=200)
    plt.close()

    # Audit for Gradient Stability
    # dX/d_index
    rr_grad = np.max(np.abs(np.diff(rr_continuous[0, :, 0].detach().numpy())))
    eda_grad = np.max(np.abs(np.diff(eda_continuous[0, :, 0].detach().numpy())))
    
    print("\n--- CONTINUOUS PATH AUDIT ---")
    print(f"Max Gradient in RR Path: {rr_grad:.4f}")
    print(f"Max Gradient in EDA Path: {eda_grad:.4f}")
    print(f"Path verification completed. Results in {output_dir}")

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_05_continuous_path')
    verify_continuous_path(2, DATA_ROOT, OUTPUT_DIR)
