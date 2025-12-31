import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add project root to path for relative imports if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from utils.config import FS_ECG, FS_EDA, FS_ACC

def visualize_raw_subject(sub_id, data_root, output_dir):
    sub_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
    if not os.path.exists(sub_path):
        sub_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
    print(f"Loading {sub_path}...")
    with open(sub_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Selection of a 10 second window in the middle for 'amplified' detail
    start_sec = 1000 
    duration_sec = 10
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ECG (Chest) - FS=700
    ecg = data['signal']['chest']['ECG'].flatten()
    s_idx = int(start_sec * FS_ECG)
    e_idx = s_idx + int(duration_sec * FS_ECG)
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(e_idx-s_idx)/FS_ECG, ecg[s_idx:e_idx], linewidth=1.5, color='#e74c3c')
    plt.title(f'Subject S{sub_id} - Raw Chest ECG (Amplified 10s window)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Voltage', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'S{sub_id}_Chest_ECG.png'), dpi=200)
    plt.close()
    print(f"Saved Amplified ECG plot to {output_dir}")

    # 2. EDA (Wrist) - FS=4
    # EDA is slow, so 60s was okay, but let's do 30s for better visibility of trends
    eda_duration = 30
    eda = data['signal']['wrist']['EDA'].flatten()
    s_idx = int(start_sec * FS_EDA)
    e_idx = s_idx + int(eda_duration * FS_EDA)
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(e_idx-s_idx)/FS_EDA, eda[s_idx:e_idx], linewidth=2, color='#2ecc71')
    plt.title(f'Subject S{sub_id} - Raw Wrist EDA (30s window)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Conductance (uS)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'S{sub_id}_Wrist_EDA.png'), dpi=200)
    plt.close()
    print(f"Saved EDA plot to {output_dir}")

    # 3. ACC (Wrist) - FS=32
    # 10s window for ACC to see tremors/movements clearly
    acc = data['signal']['wrist']['ACC'] # (N, 3)
    s_idx = int(start_sec * FS_ACC)
    e_idx = s_idx + int(duration_sec * FS_ACC)
    plt.figure(figsize=(15, 6))
    time_acc = np.arange(e_idx-s_idx)/FS_ACC
    plt.plot(time_acc, acc[s_idx:e_idx, 0], label='X', alpha=0.8)
    plt.plot(time_acc, acc[s_idx:e_idx, 1], label='Y', alpha=0.8)
    plt.plot(time_acc, acc[s_idx:e_idx, 2], label='Z', alpha=0.8)
    plt.title(f'Subject S{sub_id} - Raw Wrist ACC (Amplified 10s window)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('m/s^2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'S{sub_id}_Wrist_ACC.png'), dpi=200)
    plt.close()
    print(f"Saved Amplified ACC plot to {output_dir}")

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_01_raw_visualization')
    visualize_raw_subject(2, DATA_ROOT, OUTPUT_DIR)
