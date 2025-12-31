import torch
import torchcde
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from continuous_path.cubic_spline_construction import build_spline
from normalization.intensity_channel import add_intensity_channel

def verify_robust_padding(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SIMULATE A BATCH WITH DIFFERENT LENGTHS ---
    max_len = 50
    n_feats = 1
    L_real = 30 # Real data only goes to 30
    
    t_full = torch.linspace(0, 1, max_len)
    s_real = (0.8 + 0.05 * torch.sin(4 * 3.14 * t_full[:L_real])).unsqueeze(-1) # First 30 points
    t_real = torch.linspace(0, 6, L_real) # Real time up to 6s
    
    # 1. OLD LOGIC: Zero Padding
    zero_pad = torch.zeros(max_len, n_feats)
    zero_pad[:L_real, :] = s_real
    
    zero_time = torch.zeros(max_len)
    zero_time[:L_real] = t_real
    
    # 2. NEW LOGIC: LOCF + Linear Time (From train_loso.py)
    locf_pad = torch.zeros(max_len, n_feats)
    locf_time = torch.zeros(max_len)
    
    locf_pad[:L_real, :] = s_real
    locf_time[:L_real] = t_real
    
    # Carry forward
    last_val = s_real[-1, :]
    locf_pad[L_real:, :] = last_val
    
    last_t = t_real[-1]
    dt = (t_real[-1] - t_real[-2])
    locf_time[L_real:] = last_t + torch.arange(1, max_len - L_real + 1).float() * dt
    
    # 3. Build Splines for both
    # We'll batch them to build at once
    batch_data = torch.stack([zero_pad, locf_pad]) # (2, 50, 1)
    batch_time = torch.stack([zero_time, locf_time]) # (2, 50)
    
    # Build Spline
    # Note: Using indices for spline base (0 to 49)
    augmented = add_intensity_channel(batch_data, batch_time)
    path = build_spline(augmented)
    
    # Evaluate at high resolution around transition (idx 20 to 50)
    t_eval = torch.linspace(20, 49, 300)
    vals = path.evaluate(t_eval) # (2, 300, 2)
    
    # --- PLOT COMPARISON ---
    plt.figure(figsize=(15, 10))
    
    # Top: The Spline Paths
    plt.subplot(2, 1, 1)
    plt.plot(t_eval.numpy(), vals[0, :, 0].detach().numpy(), color='red', linestyle='--', label='BEFORE: Zero Padding (The Cliff)', linewidth=2)
    plt.plot(t_eval.numpy(), vals[1, :, 0].detach().numpy(), color='green', label='AFTER: LOCF Padding (Stable)', linewidth=3)
    plt.axvline(x=29, color='black', alpha=0.5, label='End of Real Data')
    plt.title('Spline Interpolation: Zero Padding vs. LOCF Padding')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Bottom: The Intensity/Time Channels
    plt.subplot(2, 1, 2)
    plt.plot(t_eval.numpy(), vals[0, :, 1].detach().numpy(), color='red', linestyle='--', label='BEFORE: Zero Time', linewidth=2)
    plt.plot(t_eval.numpy(), vals[1, :, 1].detach().numpy(), color='green', label='AFTER: Linear Time Extension', linewidth=3)
    plt.title('Intensity/Time Channel: Discontinuity vs. Continuous Growth')
    plt.ylabel('Time (s)')
    plt.xlabel('Point Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_07_Robust_Padding_Verification.png'), dpi=200)
    plt.close()
    
    # Gradient Audit
    dt_eval = (t_eval[1] - t_eval[0]).item()
    grad_before = np.max(np.abs(np.diff(vals[0, :, 0].detach().numpy())) / dt_eval)
    grad_after = np.max(np.abs(np.diff(vals[1, :, 0].detach().numpy())) / dt_eval)
    
    print("\n" + "="*50)
    print("        STABILITY VERIFICATION RESULTS")
    print("="*50)
    print(f"Max Gradient BEFORE (Zero Pad): {grad_before:.4f}")
    print(f"Max Gradient AFTER (LOCF Pad): {grad_after:.4f}")
    print(f"Improvement Factor: {grad_before / grad_after:.1f}x more stable")
    print("="*50 + "\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_07_robustness')
    verify_robust_padding(OUTPUT_DIR)
