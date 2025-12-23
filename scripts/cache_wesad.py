import os
import sys
import torch
import torchcde
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from training.wesad_dataset import WESADDataset
from normalization.intensity_channel import add_intensity_channel
from utils.config import PROJECT_ROOT

def cache_coefficients(subject_ids, data_root, cache_root):
    """
    Advanced caching: Performs normalization, intensity augmentation, and 
    natural cubic spline coefficient computation.
    Saves 'coeffs' tensors which can be directly used by the model.
    """
    os.makedirs(cache_root, exist_ok=True)
    
    for sub_id in subject_ids:
        print(f"\nProcessing Subject S{sub_id} for Coefficient Caching...")
        sub_cache_dir = os.path.join(cache_root, f'S{sub_id}')
        os.makedirs(sub_cache_dir, exist_ok=True)
        
        # 1. Load subject data
        ds = WESADDataset([sub_id], data_root, use_cache=False)
        
        # 2. Extract and Preprocess all windows to compute stats on final features
        print(f"  Preprocessing all windows to get stats for S{sub_id}...")
        processed_samples = []
        for i in tqdm(range(len(ds)), desc="Preprocessing"):
            processed_samples.append(ds[i])
            
        def get_stats(key):
            # concatenate all sequences for this key
            all_data = [s[key] for s in processed_samples]
            flat = np.concatenate(all_data, axis=0)
            return np.mean(flat, axis=0), np.std(flat, axis=0)
            
        print(f"  Calculating stats for S{sub_id}...")
        m_ecg, s_ecg = get_stats('ecg_seq')
        m_eda, s_eda = get_stats('eda_seq')
        m_acc, s_acc = get_stats('acc_seq')
        
        print(f"  Computing and saving coefficients...")
        # 3. Build splines and save
        for i, ps in enumerate(tqdm(processed_samples, desc="CDE Splines")):
            def normalize(data, mean, std):
                eps = 1e-8
                return (data - mean) / (std + eps)
                
            ecg_norm = normalize(ps['ecg_seq'], m_ecg, s_ecg)
            eda_norm = normalize(ps['eda_seq'], m_eda, s_eda)
            acc_norm = normalize(ps['acc_seq'], m_acc, s_acc)
            
            def build_coeffs(data, times):
                d_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                t_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
                d_in = add_intensity_channel(d_tensor, t_tensor)
                return torchcde.natural_cubic_coeffs(d_in).squeeze(0)
                
            save_data = {
                'ecg_coeffs': build_coeffs(ecg_norm, ps['ecg_time']),
                'eda_coeffs': build_coeffs(eda_norm, ps['eda_time']),
                'acc_coeffs': build_coeffs(acc_norm, ps['acc_time']),
                'label': ps['label']
            }
            
            save_path = os.path.join(sub_cache_dir, f'sample_{i:04d}.pt')
            torch.save(save_data, save_path)

if __name__ == "__main__":
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
    CACHE_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed', 'WESAD_Coeffs')
    
    ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    cache_coefficients(ALL_SUBJECTS, DATA_ROOT, CACHE_ROOT)
    print("\nCoefficient Caching Complete!")
