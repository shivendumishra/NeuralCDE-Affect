import os
import sys
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel
from training.wesad_dataset import WESADDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline
import torchcde
from normalization.intensity_channel import add_intensity_channel

def test_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    config = {'hidden_dim': 16, 'num_heads': 4, 'num_layers': 2}
    model = FullPipelineModel(config).to(device)
    
    weights_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'wesad_model.pth')
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("Model loaded.")
    
    data_root = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    dataset = WESADDataset(subject_ids=[2], data_root=data_root)
    
    sample = dataset[0]
    print("Sample preprocessed.")
    
    def build_coeffs(data, times):
        d_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
        d_in = add_intensity_channel(d_tensor, t_tensor)
        return torchcde.natural_cubic_coeffs(d_in)

    ecg_coeffs = build_coeffs(sample['ecg_seq'], sample['ecg_time'])
    eda_coeffs = build_coeffs(sample['eda_seq'], sample['eda_time'])
    acc_coeffs = build_coeffs(sample['acc_seq'], sample['acc_time'])
    
    ecg_p = torchcde.CubicSpline(ecg_coeffs).to(device)
    eda_p = torchcde.CubicSpline(eda_coeffs).to(device)
    acc_p = torchcde.CubicSpline(acc_coeffs).to(device)
    
    timeline = generate_fixed_timeline(0, 60, 1.0).to(device)
    
    with torch.no_grad():
        logits = model(ecg_p, eda_p, acc_p, timeline)
        print(f"Logits: {logits}")
        print("Inference successful!")

if __name__ == "__main__":
    test_inference()
