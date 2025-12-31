import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from normalization.zscore_normalization import zscore_normalize

def verify_normalization_impact(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- MOCK DATA SIMULATING REAL WESAD VALUES ---
    # ECG (RR-Intervals): ~800ms
    # EDA: ~0.5 uS
    # ACC: ~20 m/s^2 (Mean)
    
    ecg = 800 + 50 * torch.randn(1, 100, 1)
    eda = 0.5 + 0.1 * torch.randn(1, 100, 1)
    acc = 20 + 5 * torch.randn(1, 100, 1)
    
    # --- BEFORE NORMALIZATION ---
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(ecg[0, :, 0].numpy(), label='ECG (RR ~800)', color='#e74c3c', linewidth=2)
    plt.plot(eda[0, :, 0].numpy(), label='EDA (~0.5)', color='#2ecc71', linewidth=2)
    plt.plot(acc[0, :, 0].numpy(), label='ACC (~20)', color='#3498db', linewidth=2)
    plt.title('BEFORE Z-Score Normalization: Massive Scale Disparity', fontsize=14)
    plt.ylabel('Raw Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- APPLY NORMALIZATION ---
    ecg_norm, mean_ecg, std_ecg = zscore_normalize(ecg)
    eda_norm, mean_eda, std_eda = zscore_normalize(eda)
    acc_norm, mean_acc, std_acc = zscore_normalize(acc)
    
    # --- AFTER NORMALIZATION ---
    plt.subplot(2, 1, 2)
    plt.plot(ecg_norm[0, :, 0].numpy(), label='Normalized ECG', color='#e74c3c', linewidth=2)
    plt.plot(eda_norm[0, :, 0].numpy(), label='Normalized EDA', color='#2ecc71', linewidth=2, alpha=0.8)
    plt.plot(acc_norm[0, :, 0].numpy(), label='Normalized ACC', color='#3498db', linewidth=2, alpha=0.6)
    plt.title('AFTER Z-Score Normalization: Balanced & Zero-Centered', fontsize=14)
    plt.ylabel('Standard Deviations (σ)')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_09_Normalization_Comparison.png'), dpi=200)
    plt.close()
    
    print("\n" + "="*60)
    print("        Z-SCORE NORMALIZATION AUDIT")
    print("="*60)
    print(f"Signal | Raw Mean | Norm Mean | Norm Std")
    print("-" * 50)
    print(f"ECG    | {mean_ecg.mean():>8.2f} | {ecg_norm.mean():>9.2f} | {ecg_norm.std():>8.2f}")
    print(f"EDA    | {mean_eda.mean():>8.2f} | {eda_norm.mean():>9.2f} | {eda_norm.std():>8.2f}")
    print(f"ACC    | {mean_acc.mean():>8.2f} | {acc_norm.mean():>9.2f} | {acc_norm.std():>8.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_09_normalization')
    verify_normalization_impact(OUTPUT_DIR)
