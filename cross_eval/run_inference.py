import torch
import os
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel
from cross_eval.affective_road_loader import AffectiveRoadDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline
from normalization.intensity_channel import add_intensity_channel
import torchcde

def run_cross_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")
    
    # 1. Load Model
    config = {'hidden_dim': 16, 'num_heads': 4, 'num_layers': 2}
    model = FullPipelineModel(config).to(device)
    
    weights_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'wesad_model.pth')
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}. Run train_and_save.py first.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Load AffectiveRoad Data
    data_root = os.path.join(PROJECT_ROOT, 'data', 'AffectiveROAD_Data')
    dataset = AffectiveRoadDataset(data_root)
    
    if len(dataset) == 0:
        print("No samples found in AffectiveRoad.")
        return
        
    timeline = generate_fixed_timeline(0, 60, 1.0).to(device)
    labels_map = {0: "Baseline", 1: "Stress", 2: "Amusement"}
    
    print("\n--- Cross-Dataset Inference Results ---")
    print(f"{'Sample':<10} | {'Predicted Emotion':<20}")
    print("-" * 35)
    
    # Test first 10 samples
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        
        # Build Splines with Intensity Channel
        ecg_t = torch.tensor(sample['ecg_seq']).unsqueeze(0)
        eda_t = torch.tensor(sample['eda_seq']).unsqueeze(0)
        acc_t = torch.tensor(sample['acc_seq']).unsqueeze(0)
        
        ecg_time = torch.tensor(sample['ecg_time'])
        eda_time = torch.tensor(sample['eda_time'])
        acc_time = torch.tensor(sample['acc_time'])
        
        ecg_in = add_intensity_channel(ecg_t, ecg_time.unsqueeze(0))
        eda_in = add_intensity_channel(eda_t, eda_time.unsqueeze(0))
        acc_in = add_intensity_channel(acc_t, acc_time.unsqueeze(0))
        
        ecg_p = torchcde.CubicSpline(torchcde.hermite_cubic_coefficients_with_backward_differences(ecg_in, ecg_time)).to(device)
        eda_p = torchcde.CubicSpline(torchcde.hermite_cubic_coefficients_with_backward_differences(eda_in, eda_time)).to(device)
        acc_p = torchcde.CubicSpline(torchcde.hermite_cubic_coefficients_with_backward_differences(acc_in, acc_time)).to(device)
        
        with torch.no_grad():
            logits = model(ecg_p, eda_p, acc_p, timeline)
            _, predicted = torch.max(logits, 1)
            
        print(f"Sample {i+1:<3} | {labels_map[predicted.item()]:<20}")

if __name__ == "__main__":
    run_cross_inference()
