import torch
import torchcde
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from continuous_path.cubic_spline_construction import build_spline
from normalization.intensity_channel import add_intensity_channel

def verify_path_integrity(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mock batch with unequal lengths
    # Sample 1: length 20, Sample 2: length 10
    max_len = 20
    n_feats = 1
    
    s1 = torch.sin(torch.linspace(0, 1, 20)).unsqueeze(-1)
    s2 = torch.sin(torch.linspace(0, 0.5, 10)).unsqueeze(-1)
    
    t1 = torch.linspace(0, 5, 20)
    t2 = torch.linspace(0, 2.5, 10)
    
    # Current padding logic (from train_loso.py)
    data_padded = torch.zeros(2, max_len, n_feats)
    times_padded = torch.zeros(2, max_len)
    
    data_padded[0, :20, :] = s1
    times_padded[0, :20] = t1
    
    data_padded[1, :10, :] = s2
    times_padded[1, :10] = t2
    
    # Add intensity channel
    # This matches train_loso.py: ecg_in = add_intensity_channel(ecg_data, ecg_t)
    # Note: intensity_channel.py appends to the end.
    data_augmented = add_intensity_channel(data_padded, times_padded)
    
    print(f"Augmented Data Shape: {data_augmented.shape}")
    print(f"Sample 2 (padded) last few rows:\n{data_augmented[1, 8:12, :]}")

    # Build Spline
    path = build_spline(data_augmented)
    
    # Evaluate at the transition point for Sample 2 (index 9 to 10)
    # Spline is built over t_idx [0, 19]
    t_eval = torch.linspace(0, 19, 200)
    vals = path.evaluate(t_eval) # (2, 200, 2)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # 1. Sample 2 Signal
    plt.subplot(2, 1, 1)
    plt.plot(t_eval.numpy(), vals[1, :, 0].detach().numpy(), label='Interpolated Signal')
    plt.axvline(x=9, color='r', linestyle='--', label='End of real data')
    plt.title('Sample 2: Signal Path with Zero Padding')
    plt.legend()
    plt.grid(True)

    # 2. Sample 2 Time Channel (The 'Intensity' channel)
    plt.subplot(2, 1, 2)
    plt.plot(t_eval.numpy(), vals[1, :, 1].detach().numpy(), label='Interpolated Time Channel')
    plt.axvline(x=9, color='r', linestyle='--', label='End of real data')
    plt.title('Sample 2: Time Channel Path with Zero Padding')
    plt.ylabel('Physiological Time (s)')
    plt.xlabel('Interpolation Index')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Spline_Padding_Audit.png'), dpi=150)
    plt.close()
    
    # Audit for NaNs or huge derivatives
    # Derivative is dX/dt_idx
    derivs = (vals[:, 1:, :] - vals[:, :-1, :]) / (t_eval[1] - t_eval[0])
    max_deriv = torch.max(torch.abs(derivs))
    print(f"Max Derivative in Path: {max_deriv.item()}")
    
    if max_deriv > 100:
        print("\n[WARNING] DETECTED EXPLOSIVE DERIVATIVES DUE TO PADDING!")
        print("This is the likely cause of NaN loss on GPU.")
    else:
        print("\nPath derivatives look stable.")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_03_path_verification')
    verify_path_integrity(OUTPUT_DIR)
