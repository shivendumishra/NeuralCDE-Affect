import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel, collate_paths
from training.wesad_dataset import WESADDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline

def train_demo_model(epochs=5):
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    # Exclude S2 to use it as unseen data for the demo
    TRAIN_SUBJECTS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    CONFIG = {
        'hidden_dim': 16,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 16,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'epochs': epochs,
        'snapshot_hz': 1.0
    }
    
    cache_root = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WESAD_Coeffs')
    train_ds = WESADDataset(TRAIN_SUBJECTS, DATA_ROOT, use_cache=True, cache_root=cache_root)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_paths)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullPipelineModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.CrossEntropyLoss()
    timeline = generate_fixed_timeline(0, 60, CONFIG['snapshot_hz']).to(device)
    
    print(f"Training demo model on subjects: {TRAIN_SUBJECTS}")
    print(f"Target unseen subject: S2")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        correct = 0
        total = 0
        for batch in pbar:
            optimizer.zero_grad()
            ecg_p = batch['ecg_path'].to(device)
            eda_p = batch['eda_path'].to(device)
            acc_p = batch['acc_path'].to(device)
            y = batch['label'].to(device)
            
            logits = model(ecg_p, eda_p, acc_p, timeline)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.1f}%"})
        
        scheduler.step()
            
    # Save the model
    save_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'demo_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Demo model saved to {save_path}")

if __name__ == "__main__":
    train_demo_model(epochs=5)
