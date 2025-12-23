import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel, collate_paths
from training.wesad_dataset import WESADDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Demo on {device}...")
    
    # 1. Load Demo Model (Trained on all subjects except S2)
    config = {'hidden_dim': 16, 'num_heads': 4, 'num_layers': 2}
    model = FullPipelineModel(config).to(device)
    
    weights_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'wesad_model.pth')
    if not os.path.exists(weights_path):
        print(f"Error: Demo model not found at {weights_path}. Run training/train_demo_model.py first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Load Unseen Subject Data (S2)
    data_root = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    cache_root = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WESAD_Coeffs')
    
    print("Loading unseen data (Subject S4)...")
    dataset = WESADDataset([4], data_root, use_cache=True, cache_root=cache_root)
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_paths)
    
    timeline = generate_fixed_timeline(0, 60, 1.0).to(device)
    
    print("\n" + "="*50)
    print("      DEMO: INFERENCE ON UNSEEN DATA (S4)")
    print("="*50)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Windows"):
            ecg_p = batch['ecg_path'].to(device)
            eda_p = batch['eda_path'].to(device)
            acc_p = batch['acc_path'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(ecg_p, eda_p, acc_p, timeline)
            _, predicted = torch.max(logits, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = 100 * (y_true == y_pred).sum() / len(y_true)
    
    print(f"\nResults for Unseen Subject S4:")
    print(f"Total Windows: {len(y_true)}")
    print(f"Accuracy:      {accuracy:.2f}%")
    print("-" * 30)
    
    target_names = ["Baseline", "Stress", "Amusement"]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Optional: Save a confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - Unseen Subject S2 (Acc: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'demo')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'demo_confusion_matrix.png'))
    print(f"\nConfusion matrix saved to {os.path.join(results_dir, 'demo_confusion_matrix.png')}")
    print("="*50)

if __name__ == "__main__":
    run_demo()
