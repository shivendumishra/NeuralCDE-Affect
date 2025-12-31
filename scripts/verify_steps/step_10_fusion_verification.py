import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from fusion_transformer.multimodal_transformer import MultimodalTransformer

def verify_fusion_logic(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SETUP ---
    batch_size = 1
    seq_len = 60 # 60 snapshots (1 per second)
    hidden_dim = 16
    num_heads = 4
    num_layers = 1
    
    model = MultimodalTransformer(input_dim=hidden_dim, 
                                  embed_dim=hidden_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers)
    model.eval()
    
    # --- CREATE "EVENT" PULSES ---
    # We create a pulse in ECG at time 10, EDA at time 20, and ACC at time 30
    # To see if the fusion mixes them
    z_ecg = torch.zeros(batch_size, seq_len, hidden_dim)
    z_eda = torch.zeros(batch_size, seq_len, hidden_dim)
    z_acc = torch.zeros(batch_size, seq_len, hidden_dim)
    
    z_ecg[0, 10, :] = 1.0 # ECG heartbeat event
    z_eda[0, 20, :] = 1.0 # EDA sweat burst
    z_acc[0, 30, :] = 1.0 # ACC movement
    
    # --- FORWARD PASS ---
    with torch.no_grad():
        # The model pools the sequence at the end, but we want to see the merged sequence
        # We'll temporarily reach into the model to see the 'fused' tensor before pooling
        
        # Project first
        h_ecg = model.proj_ecg(z_ecg)
        h_eda = model.proj_eda(z_eda)
        h_acc = model.proj_acc(z_acc)
        
        # Cross-modal mix
        ecg_mix = h_ecg + model.cma_ecg_eda(h_ecg, h_eda, h_eda) + model.cma_ecg_acc(h_ecg, h_acc, h_acc)
        eda_mix = h_eda + model.cma_eda_ecg(h_eda, h_ecg, h_ecg) + model.cma_eda_acc(h_eda, h_acc, h_acc)
        acc_mix = h_acc + model.cma_acc_ecg(h_acc, h_ecg, h_ecg) + model.cma_acc_eda(h_acc, h_eda, h_eda)
        
        fused_seq = torch.cat([ecg_mix, eda_mix, acc_mix], dim=-1) # (1, 60, 48)
        
        # Final embedding (pooled)
        final_embedding = model(z_ecg, z_eda, z_acc)
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 12))
    
    # 1. Input Modalities
    plt.subplot(3, 1, 1)
    plt.plot(z_ecg[0, :, 0].numpy(), label='ECG Latent Activity', color='#e74c3c')
    plt.plot(z_eda[0, :, 0].numpy(), label='EDA Latent Activity', color='#2ecc71')
    plt.plot(z_acc[0, :, 0].numpy(), label='ACC Latent Activity', color='#3498db')
    plt.title('Stage 1: Individual Latent Trajectories (Before Fusion)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.2)

    # 2. Fused Sequence Heatmap
    # This shows how the activities are now present in the combined vector
    plt.subplot(3, 1, 2)
    # Mapping the 3*hidden_dim vector over time
    im = plt.imshow(fused_seq[0].T.numpy(), aspect='auto', cmap='magma')
    plt.title('Stage 2: Cross-Modal Fused Sequence (Inter-modality dependency learned)')
    plt.ylabel('Feature Dimension (3 * Embed)')
    plt.colorbar(im, label='Activation')
    # Label the pulse times
    plt.axvline(x=10, color='white', linestyle='--', alpha=0.5)
    plt.axvline(x=20, color='white', linestyle='--', alpha=0.5)
    plt.axvline(x=30, color='white', linestyle='--', alpha=0.5)
    
    # 3. Decision Vector (The Pooled Vector)
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(final_embedding.shape[1]), final_embedding[0].numpy(), color='purple')
    plt.title('Stage 3: Global Emotion Embedding (Input for Classifier)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Weight')
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Step_10_Fusion_Verification.png'), dpi=200)
    plt.close()
    
    print(f"Fusion verification completed.")
    print(f"Fused Sequence Shape: {fused_seq.shape}")
    print(f"Final Embedding Size: {final_embedding.shape[1]}")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_10_fusion')
    verify_fusion_logic(OUTPUT_DIR)
