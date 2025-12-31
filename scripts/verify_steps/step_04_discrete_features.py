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

def visualize_discrete_points(sub_id, data_root, output_dir):
    sub_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
    if not os.path.exists(sub_path):
        sub_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
    with open(sub_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    start_sec = 1000 
    duration = 15 # 15 seconds to see enough RR intervals
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. ECG -> Discrete RR Intervals ---
    ecg_raw = data['signal']['chest']['ECG'].flatten()[int(start_sec*FS_ECG):int((start_sec+duration)*FS_ECG)]
    _, r_peaks = pan_tompkins_detector(ecg_raw, FS_ECG)
    rr_intervals = compute_rr_intervals(r_peaks, FS_ECG)
    # Times for RR intervals are the peak times
    r_times = r_peaks[1:] / FS_ECG # Time of the end of each interval

    plt.figure(figsize=(15, 5))
    plt.stem(r_times, rr_intervals, basefmt=" ", label='RR-Interval Points')
    plt.scatter(r_times, rr_intervals, color='red', zorder=3)
    plt.title(f'Subject S{sub_id} - Discrete RR-Interval Points (Pre-Spline)')
    plt.xlabel('Time (s)')
    plt.ylabel('Interval (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Step_04_ECG_Discrete_Points.png'), dpi=200)
    plt.close()

    # --- 2. EDA -> Discrete Phasic/Tonic ---
    eda_raw = data['signal']['wrist']['EDA'].flatten()[int(start_sec*FS_EDA):int((start_sec+duration)*FS_EDA)]
    _, phasic, tonic = eda_preprocessing(eda_raw, FS_EDA)
    t_eda = np.arange(len(eda_raw))/FS_EDA

    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_eda, tonic, color='black', alpha=0.2, linestyle='--')
    plt.scatter(t_eda, tonic, color='#2ecc71', s=20, label='Discrete Tonic Points')
    plt.title('Discrete Tonic Points (4Hz Sampling)')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.subplot(2, 1, 2)
    plt.scatter(t_eda, phasic, color='#3498db', s=20, label='Discrete Phasic Points')
    plt.title('Discrete Phasic Points (4Hz Sampling)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_04_EDA_Discrete_Points.png'), dpi=200)
    plt.close()

    # --- 3. ACC -> Discrete Magnitude ---
    acc_raw = data['signal']['wrist']['ACC'][int(start_sec*FS_ACC):int((start_sec+duration)*FS_ACC)]
    mag = preprocess_accelerometer(acc_raw, FS_ACC)
    t_acc = np.arange(len(mag))/FS_ACC

    plt.figure(figsize=(15, 5))
    # Plotting only first 2 seconds of the 15s window to see ACC points clearly (32Hz is dense)
    zoom_idx = int(2 * FS_ACC)
    plt.plot(t_acc[:zoom_idx], mag[:zoom_idx], color='black', alpha=0.1)
    plt.scatter(t_acc[:zoom_idx], mag[:zoom_idx], color='#9b59b6', s=15, label='Discrete ACC Magnitude Points')
    plt.title('Discrete ACC Points (32Hz Sampling - 2s Zoom)')
    plt.xlabel('Time (s)')
    plt.ylabel('m/s^2')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'Step_04_ACC_Discrete_Points.png'), dpi=200)
    plt.close()

    print(f"Discrete point visualization saved to {output_dir}")

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_04_discrete_features')
    visualize_discrete_points(2, DATA_ROOT, OUTPUT_DIR)
