import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from utils.config import FS_EDA, FS_ACC

class AffectiveRoadDataset(Dataset):
    def __init__(self, data_root, window_size_sec=60, step_size_sec=10):
        self.samples = []
        self.window_size_sec = window_size_sec
        
        e4_root = os.path.join(data_root, 'Database', 'E4')
        annot_path = os.path.join(e4_root, 'Annot_E4_Left.csv')
        
        if not os.path.exists(annot_path):
            print(f"Error: Annotation file not found at {annot_path}")
            return

        annots = pd.read_csv(annot_path)
        
        # For the fast demo, we only process the first drive (Drv1)
        for _, row in annots.iloc[:1].iterrows():
            drive_id = row['Drive-id']
            drive_folder = None
            for d in os.listdir(e4_root):
                if d.endswith(drive_id):
                    drive_folder = os.path.join(e4_root, d, 'Left')
                    break
            
            if not drive_folder or not os.path.exists(drive_folder):
                continue
                
            print(f"Pre-loading {drive_id} for inference...")
            
            try:
                # Load Signals
                eda_df = pd.read_csv(os.path.join(drive_folder, 'EDA.csv'), skiprows=2, header=None)
                acc_df = pd.read_csv(os.path.join(drive_folder, 'ACC.csv'), skiprows=2, header=None)
                ibi_df = pd.read_csv(os.path.join(drive_folder, 'IBI.csv'), skiprows=1, header=None)
                
                eda_raw = eda_df[0].values.astype(np.float32)
                acc_raw = acc_df.values.astype(np.float32)
                ibi_offsets = ibi_df[0].values.astype(np.float32)
                ibi_values = ibi_df[1].values.astype(np.float32)
                
                duration_sec = len(eda_raw) / 4.0
                
                # Windowing with Ground Truth Labels
                current_time = 0.0
                
                # Create a list of (start, end, label) from the row
                # Mapping: Rest -> 0 (Baseline), City/Hwy -> 1 (Stress)
                intervals = []
                cols = annots.columns[1:]
                for i in range(0, len(cols), 2):
                    label_str = cols[i].split('_')[0]
                    start = row[cols[i]]
                    end = row[cols[i+1]]
                    
                    if label_str == 'Rest':
                        l = 0
                    elif label_str in ['City1', 'City2', 'Hwy']:
                        l = 1
                    else:
                        continue # Skip 'Z' or others
                    intervals.append((start, end, l))

                while current_time + window_size_sec <= duration_sec:
                    # Find label for this window
                    # A window is valid if it's entirely within one interval
                    win_label = None
                    for start, end, l in intervals:
                        if current_time >= start and (current_time + window_size_sec) <= end:
                            win_label = l
                            break
                    
                    if win_label is None:
                        current_time += step_size_sec
                        continue

                    start_idx_4hz = int(current_time * 4.0)
                    end_idx_4hz = start_idx_4hz + int(window_size_sec * 4.0)
                    start_idx_32hz = int(current_time * 32.0)
                    end_idx_32hz = start_idx_32hz + int(window_size_sec * 32.0)
                    
                    if end_idx_4hz > len(eda_raw) or end_idx_32hz > len(acc_raw): break
                        
                    mask = (ibi_offsets >= current_time) & (ibi_offsets < current_time + window_size_sec)
                    win_ibi = ibi_values[mask]
                    win_ibi_times = ibi_offsets[mask] - current_time
                    
                    if len(win_ibi) >= 2:
                        self.samples.append({
                            'ibi': win_ibi,
                            'ibi_times': win_ibi_times,
                            'eda': eda_raw[start_idx_4hz:end_idx_4hz],
                            'acc': acc_raw[start_idx_32hz:end_idx_32hz],
                            'label': win_label
                        })
                    current_time += step_size_sec
            except Exception as e:
                print(f"Error: {e}")

        # Drive-wide Normalization Stats
        if self.samples:
            print("Calculating normalization stats for AffectiveRoad...")
            all_ibi = np.concatenate([s['ibi'] for s in self.samples])
            all_eda = np.concatenate([s['eda'] for s in self.samples]) # This is raw EDA, but we need stats for phasic/tonic
            # Actually, it's better to compute stats on the final features
            
            # We'll compute stats on the fly for the first 50 samples to get a good estimate
            # or just process all since it's a small dataset
            all_ecg_feats = []
            all_eda_feats = []
            all_acc_feats = []
            
            from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
            from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
            from preprocessing.acc_statistical_features import extract_acc_features
            
            for s in self.samples[:100]: # Sample 100 for speed
                _, phasic, tonic = eda_preprocessing(s['eda'], 4.0)
                mag = preprocess_accelerometer(s['acc'], 32.0)
                acc_f = []
                for i in range(0, len(mag), 32):
                    w = mag[i:i+32]
                    if len(w) < 32: break
                    f = extract_acc_features(w)
                    acc_f.append([f['acc_mean'], f['acc_var'], f['acc_energy'], f['acc_entropy']])
                
                all_ecg_feats.append(s['ibi'].reshape(-1, 1))
                all_eda_feats.append(np.stack([phasic, tonic], axis=1))
                if acc_f: all_acc_feats.append(np.array(acc_f))
                
            self.stats = {
                'ecg_m': np.mean(np.concatenate(all_ecg_feats), axis=0),
                'ecg_s': np.std(np.concatenate(all_ecg_feats), axis=0) + 1e-8,
                'eda_m': np.mean(np.concatenate(all_eda_feats), axis=0),
                'eda_s': np.std(np.concatenate(all_eda_feats), axis=0) + 1e-8,
                'acc_m': np.mean(np.concatenate(all_acc_feats), axis=0),
                'acc_s': np.std(np.concatenate(all_acc_feats), axis=0) + 1e-8
            }
            print("Normalization stats computed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
        from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
        from preprocessing.acc_statistical_features import extract_acc_features
        
        _, phasic, tonic = eda_preprocessing(sample['eda'], 4.0)
        mag = preprocess_accelerometer(sample['acc'], 32.0)
        
        acc_feats = []
        for i in range(0, len(mag), 32):
            w = mag[i:i+32]
            if len(w) < 32: break
            f = extract_acc_features(w)
            acc_feats.append([f['acc_mean'], f['acc_var'], f['acc_energy'], f['acc_entropy']])
            
        # Apply Normalization
        ecg_norm = (sample['ibi'].reshape(-1, 1) - self.stats['ecg_m']) / self.stats['ecg_s']
        eda_norm = (np.stack([phasic, tonic], axis=1) - self.stats['eda_m']) / self.stats['eda_s']
        acc_norm = (np.array(acc_feats) - self.stats['acc_m']) / self.stats['acc_s']

        # Compute Coefficients for CDE
        import torchcde
        from normalization.intensity_channel import add_intensity_channel
        
        def build_coeffs(data, times):
            d_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
            d_in = add_intensity_channel(d_tensor, t_tensor)
            return torchcde.natural_cubic_coeffs(d_in).squeeze(0)
            
        return {
            'ecg_coeffs': build_coeffs(ecg_norm, sample['ibi_times']),
            'eda_coeffs': build_coeffs(eda_norm, np.arange(len(phasic))/4.0),
            'acc_coeffs': build_coeffs(acc_norm, np.arange(len(acc_norm))),
            'label': sample['label']
        }
