import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from classifier.emotion_classifier import EmotionClassifier

def verify_classifier_head(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- SETUP ---
    batch_size = 5
    input_dim = 48 # Output dim from our MultimodalTransformer (3 * embed_dim)
    num_classes = 3 # Baseline, Stress, Amusement
    
    model = EmotionClassifier(input_dim=input_dim, num_classes=num_classes)
    model.eval()
    
    # --- CREATE MOCK EMBEDDINGS ---
    # We'll create 5 different embeddings representing 5 test samples
    embeddings = torch.randn(batch_size, input_dim)
    
    # --- FORWARD PASS ---
    with torch.no_grad():
        logits, probs = model(embeddings)
    
    # --- VERIFICATION CHECKS ---
    print("\n" + "="*50)
    print("        CLASSIFIER AUDIT RESULTS")
    print("="*50)
    print(f"Input Shape: {embeddings.shape}")
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Output Probs Shape: {probs.shape}")
    
    # Check if probabilities sum to 1
    prob_sums = probs.sum(dim=1).numpy()
    print(f"Probability Sums (should be 1.0): {prob_sums}")
    
    # Check for NaNs
    if torch.isnan(logits).any():
        print("[ERROR] NaNs detected in classifier logits!")
    else:
        print("[SUCCESS] No NaNs in classifier output.")

    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 8))
    
    # 1. Visualization of Probabilities for 5 different windows
    x = np.arange(num_classes)
    classes = ['Baseline (0)', 'Stress (1)', 'Amusement (2)']
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for i in range(batch_size):
        plt.subplot(1, 5, i+1)
        plt.bar(x, probs[i].numpy(), color=colors)
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.title(f'Sample {i+1}')
        if i == 0:
            plt.ylabel('Confidence Score')
            
    plt.suptitle('Stage 4: Final Emotion Classification (3-Class Probabilities)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Step_11_Classifier_Head.png'), dpi=200)
    plt.close()
    
    # Summary of Mapping
    print("\nLabel Mapping Verification:")
    print("---------------------------")
    print("Raw WESAD | Model Value | Meaning")
    print("   1      |      0      | Baseline")
    print("   2      |      1      | Stress")
    print("   3      |      2      | Amusement")
    print("---------------------------")
    print(f"Results saved to {output_dir}\n")

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step_11_classifier')
    verify_classifier_head(OUTPUT_DIR)
