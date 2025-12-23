import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from utils.config import FS_ECG, FS_EDA, FS_ACC
from preprocessing.ecg_preprocessing_pan_tompkins import pan_tompkins_detector
from preprocessing.ecg_hrv_extraction import compute_rr_intervals
from preprocessing.eda_preprocessing_cvxeda import eda_preprocessing
from preprocessing.acc_preprocessing_filtering import preprocess_accelerometer
from preprocessing.acc_statistical_features import extract_acc_features

class WESADDataset(Dataset):
    """
    PyTorch Dataset for WESAD (Wearable Stress and Affect Detection) dataset.
    """
    
    def __init__(self, subject_ids, data_root, window_size_sec=60, step_size_sec=10, valid_labels=[1, 2, 3], use_cache=False, cache_root=None):
        self.samples = []
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.window_size_sec = window_size_sec
        self.win_ecg = int(window_size_sec * FS_ECG)
        self.win_eda = int(window_size_sec * FS_EDA)
        self.win_acc = int(window_size_sec * FS_ACC)
        
        print(f"Loading WESAD Dataset for subjects: {subject_ids}", flush=True)
        
        if use_cache and cache_root:
            for sub_id in subject_ids:
                sub_cache_dir = os.path.join(cache_root, f'S{sub_id}')
                if os.path.exists(sub_cache_dir):
                    files = sorted([os.path.join(sub_cache_dir, f) for f in os.listdir(sub_cache_dir) if f.endswith('.pt')])
                    self.samples.extend(files)
                    print(f"Subject S{sub_id}: Loaded {len(files)} cached samples.")
                else:
                    print(f"Warning: Cache for S{sub_id} not found at {sub_cache_dir}. Falling back to raw processing.")
                    self._load_subject_raw(sub_id, data_root, window_size_sec, step_size_sec, valid_labels)
        else:
            for sub_id in subject_ids:
                self._load_subject_raw(sub_id, data_root, window_size_sec, step_size_sec, valid_labels)

    def _load_subject_raw(self, sub_id, data_root, window_size_sec, step_size_sec, valid_labels):
        # Handle both nested and flat structure
        nested_path = os.path.join(data_root, f'S{sub_id}', f'S{sub_id}.pkl')
        flat_path = os.path.join(data_root, f'S{sub_id}.pkl')
        
        if os.path.exists(nested_path):
            sub_path = nested_path
        elif os.path.exists(flat_path):
            sub_path = flat_path
        else:
            print(f"Warning: Data for S{sub_id} not found at {nested_path} or {flat_path}. Skipping.")
            return
        
        print(f"Loading {sub_path} ...", flush=True)
        try:
            with open(sub_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            print(f"Loaded {sub_path}. Structuring...", flush=True)
        except Exception as e:
            print(f"Error loading {sub_path}: {e}")
            return
            
        # Extract Signals
        ecg_raw = data['signal']['chest']['ECG'].flatten()
        eda_raw = data['signal']['wrist']['EDA'].flatten()
        acc_raw = data['signal']['wrist']['ACC'] # (N, 3)
        labels = data['label'].flatten()
        
        len_ecg = len(ecg_raw)
        len_eda = len(eda_raw)
        len_acc = len(acc_raw)
        duration_sec = len_ecg / FS_ECG
        
        current_time = 0.0
        count_wins = 0
        
        while current_time + window_size_sec <= duration_sec:
            start_ecg = int(current_time * FS_ECG)
            end_ecg = start_ecg + self.win_ecg
            
            start_eda = int(current_time * FS_EDA)
            end_eda = start_eda + self.win_eda
            
            start_acc = int(current_time * FS_ACC)
            end_acc = start_acc + self.win_acc
            
            if end_ecg > len_ecg or end_eda > len_eda or end_acc > len_acc:
                break
                
            label_window = labels[start_ecg:end_ecg]
            if len(label_window) == 0:
                break
                
            vals, counts = np.unique(label_window, return_counts=True)
            maj_label = vals[np.argmax(counts)]
            
            if maj_label in valid_labels:
                self.samples.append({
                    'ecg': ecg_raw[start_ecg:end_ecg].astype(np.float32),
                    'eda': eda_raw[start_eda:end_eda].astype(np.float32),
                    'acc': acc_raw[start_acc:end_acc].astype(np.float32),
                    'label': int(maj_label)
                })
                count_wins += 1
            
            current_time += step_size_sec
        
        print(f"Subject S{sub_id}: Extracted {count_wins} windows.", flush=True)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        if self.use_cache and isinstance(self.samples[idx], str):
            return torch.load(self.samples[idx], weights_only=False)
        return self.preprocess_sample(self.samples[idx])

    def preprocess_sample(self, sample):
        # 1. ECG
        _, r_peaks = pan_tompkins_detector(sample['ecg'], FS_ECG)
        rr_intervals = compute_rr_intervals(r_peaks, FS_ECG) 
        if len(rr_intervals) < 2:
             # Fallback if no peaks or single peak (TorchCDE needs >=2)
             rr_intervals = np.array([0.0, 0.0])
             r_times = np.array([0.0, 1.0])
        else:
             r_times = r_peaks.astype(np.float32) / FS_ECG
             # Truncate and align
             if len(r_times) > len(rr_intervals):
                 r_times = r_times[1:]
             r_times = r_times - r_times[0]
        
        # 2. EDA
        _, phasic, tonic = eda_preprocessing(sample['eda'], FS_EDA)
        
        # 3. ACC
        mag = preprocess_accelerometer(sample['acc'], FS_ACC)
        winsize = 32
        acc_feats_list = []
        for i in range(0, len(mag), winsize):
            w = mag[i:i+winsize]
            if len(w) < winsize: break
            f = extract_acc_features(w)
            vec = [f['acc_mean'], f['acc_var'], f['acc_energy'], f['acc_entropy']]
            acc_feats_list.append(vec)
        
        if len(acc_feats_list) < 2:
            acc_seq = np.zeros((2, 4))
            acc_time = np.array([0.0, 1.0])
        else:
            acc_seq = np.array(acc_feats_list)
            acc_time = np.arange(len(acc_seq)) * 1.0
        
        return {
            'ecg_seq': rr_intervals.reshape(-1, 1),
            'ecg_time': r_times,
            'eda_seq': np.stack([phasic, tonic], axis=1),
            'eda_time': np.arange(len(phasic))/FS_EDA,
            'acc_seq': acc_seq,
            'acc_time': acc_time,
            'label': int(sample['label']) - 1 # Remap 1-3 to 0-2
        }
