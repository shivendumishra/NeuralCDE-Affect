import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from continuous_path.cubic_spline_construction import build_spline
from latent_discretization.temporal_sampling import generate_fixed_timeline, sample_latent_trajectory
from neural_cde.neural_cde_model import NeuralCDE
from normalization.intensity_channel import add_intensity_channel

def verify_discretization(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SETUP ---
    # 1. Create a mock continuous signal (60 seconds)
    duration = 60.0
    fs = 4.0 # 4Hz discrete points
    t_discrete = torch.arange(0, duration, 1.0/fs)
    # A simple sine wave representing EDA
    eda_values = (2.0 + 0.5 * torch.sin(2 * 3.14 * 0.05 * t_discrete)).unsqueeze(0).unsqueeze(-1) # (1, L, 1)
    
    # 2. Build the Path (Step 5 logic)
    eda_augmented = add_intensity_channel(eda_values, t_discrete.unsqueeze(0))
    path = build_spline(eda_augmented)
    
    # 3. Setup Neural CDE (Step 6/7 logic)
    # Latent Dim = 4 for this demo
    model = NeuralCDE(input_channels=2, hidden_dim=4)
    model.eval()
    
    # 4. DISCRETIZATION (The Snapshots)
    # Sampling rate for latent space = 1.0 Hz (1 snapshot per second)
    sampling_rate_hz = 1.0 
    timeline = generate_fixed_timeline(0, duration, sampling_rate_hz)
    
    # Take the snapshots
    with torch.no_grad():
        z_snapshots = sample_latent_trajectory(model, path, timeline) # (batch, steps, hidden_dim)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    # Plot 1: The Input and the Grid
    plt.subplot(2, 1, 1)
    t_fine = torch.linspace(0, duration, 500)
    input_path_eval = path.evaluate(torch.linspace(0, len(t_discrete)-1, 500))
    
    plt.plot(t_fine.numpy(), input_path_eval[0, :, 0].numpy(), color='blue', label='Continuous Physical Signal (EDA)')
    
    # Draw vertical lines for the "Sampling Clock"
    for t in timeline[::2]: # Every 2nd line to avoid clutter
        plt.axvline(x=t, color='gray', alpha=0.2, linestyle=':')
    plt.axvline(x=timeline[0], color='gray', alpha=0.5, linestyle=':', label='Transformer Sampling Grid')
    
    plt.title('Stage 1: The Continuous Input with Sampling Grid')
    plt.ylabel('Conductance')
    plt.legend()
    plt.grid(True, alpha=0.1)

    # Plot 2: The Latent Snapshots
    plt.subplot(2, 1, 2)
    # We plot the first latent dimension as an example
    z_example = z_snapshots[0, :, 0].numpy()
    
    # Draw the "Hidden Intelligence" line (what the NCDE thinks)
    # For visualization, we evaluate it at higher res too
    t_latent_fine = torch.linspace(0, duration, 200)
    with torch.no_grad():
        z_fine = sample_latent_trajectory(model, path, t_latent_fine)
    
    plt.plot(t_latent_fine.numpy(), z_fine[0, :, 0].numpy(), color='purple', alpha=0.3, label='Continuous Hidden State z(t)')
    
    # Draw the discrete snapshots that go to the Transformer
    plt.scatter(timeline.numpy(), z_snapshots[0, :, 0].numpy(), color='purple', s=40, label='Discrete Latent Tokens for Transformer')
    
    # Add arrows to show the "taking a snapshot" action
    for i in range(0, len(timeline), 10):
        plt.annotate('', xy=(timeline[i], z_snapshots[0, i, 0]), xytext=(timeline[i], z_snapshots[0, i, 0]+0.2),
                     arrowprops=dict(arrowstyle='->', color='black'))

    plt.title('Stage 2: Discrete Latent Snapshots (Discretization)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Latent Value (z_0)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_08_Latent_Discretization.png'), dpi=200)
    plt.close()
    
    print(f"\nDiscretization Audit:")
    print(f"Total Window: {duration}s")
    print(f"Discrete Data Points: {len(t_discrete)} (Input)")
    print(f"Latent Snapshots: {len(timeline)} (Created for Transformer)")
    print(f"Snapshot Resolution: {1.0/sampling_rate_hz} second per token")
    print(f"Visualization saved to {output_dir}\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_08_discretization')
    verify_discretization(OUTPUT_DIR)
