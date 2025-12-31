import torch
import torchcde
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from continuous_path.cubic_spline_construction import build_spline
from normalization.intensity_channel import add_intensity_channel

def visualize_cliff_problem(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SIMULATE A BATCH WITH DIFFERENT LENGTHS ---
    # Sample 1: 50 points (Full)
    # Sample 2: 30 points (Shorter - needing padding)
    max_len = 50
    n_feats = 1
    
    # Biological-looking RR data (around 800ms)
    t = torch.linspace(0, 1, max_len)
    s1 = (0.8 + 0.05 * torch.sin(2 * 3.14 * t)).unsqueeze(-1) # (50, 1)
    
    t_short = torch.linspace(0, 0.6, 30)
    s2_real = (0.8 + 0.05 * torch.sin(2 * 3.14 * t_short)).unsqueeze(-1) # (30, 1)
    
    # 1. Zero Padding (This is what train_loso.py currently does)
    data_padded = torch.zeros(2, max_len, n_feats)
    data_padded[0, :, :] = s1
    data_padded[1, :30, :] = s2_real
    # data_padded[1, 30:, :] remains 0.0 <--- THE PROBLEM
    
    # 2. Add Intensity/Time Channel
    # Assuming uniform indices for this demonstration
    times = torch.arange(max_len).float().unsqueeze(0).repeat(2, 1)
    data_augmented = add_intensity_channel(data_padded, times)
    
    # 3. Build the Continuous Path (Spline)
    # The spline will try to connect the last real point (idx 29) to the first zero (idx 30)
    path = build_spline(data_augmented)
    
    # Evaluate at high resolution around the transition (idx 25 to 35)
    t_eval = torch.linspace(25, 35, 200)
    vals = path.evaluate(t_eval) # (2, 200, 2)
    
    # --- PLOT THE CLIFF ---
    plt.figure(figsize=(15, 8))
    
    # Plot Sample 1 (Steady)
    plt.subplot(2, 1, 1)
    plt.plot(t_eval.numpy(), vals[0, :, 0].detach().numpy(), label='Sample 1 (Full Data)', color='green', linewidth=2)
    plt.title('Sample 1: Continuous and Stable')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot Sample 2 (The Cliff)
    plt.subplot(2, 1, 2)
    # Original data points for reference
    plt.scatter(np.arange(25, 30), s2_real[25:30, 0].numpy(), color='blue', label='Real Data Points', zorder=5)
    plt.scatter(np.arange(30, 35), np.zeros(5), color='red', label='Zero Padding', zorder=5)
    
    # The Spline Path
    plt.plot(t_eval.numpy(), vals[1, :, 0].detach().numpy(), label='The "Cliff" Path', color='red', linewidth=3)
    
    plt.axvline(x=29, color='black', linestyle='--', alpha=0.5)
    plt.annotate('SUDDEN CRASH TO ZERO', xy=(29.5, 0.4), xytext=(31, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')
    
    plt.title('Sample 2: The "Cliff" Phenomenon (Padding Failure)')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_06_The_Cliff_Problem.png'), dpi=200)
    plt.close()
    
    # --- AUDIT THE GRADIENTS ---
    # Derivative = ΔX / Δt
    dt = (t_eval[1] - t_eval[0]).item()
    grads = np.abs(np.diff(vals[1, :, 0].detach().numpy())) / dt
    max_grad = np.max(grads)
    
    print("\n" + "="*50)
    print("        CLIFF AUDIT RESULTS")
    print("="*50)
    print(f"Normal Gradient (Sample 1): {np.max(np.abs(np.diff(vals[0, :, 0].detach().numpy())) / dt):.4f}")
    print(f"Exploding Gradient (Sample 2): {max_grad:.4f}")
    print(f"\nCRITICAL: The gradient at the cliff is {max_grad/4:.0f} times larger than normal!")
    print("This is the mathematical reason for NaN loss on GPU.")
    print("="*50 + "\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_06_cliff_audit')
    visualize_cliff_problem(OUTPUT_DIR)
