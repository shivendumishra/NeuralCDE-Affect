import os
import sys
import torch
import numpy as np
import pickle
import traceback
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from training.train_loso import FullPipelineModel, collate_paths
from training.wesad_dataset import WESADDataset
from latent_discretization.temporal_sampling import generate_fixed_timeline
import torchcde
from normalization.intensity_channel import add_intensity_channel

app = Flask(__name__)
CORS(app)

# Global variables for model and dataset
model = None
dataset = None
stats = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timeline = generate_fixed_timeline(0, 60, 1.0).to(device)

def compute_dataset_stats(ds):
    print("Computing normalization stats for WESAD samples...")
    all_ecg = []
    all_eda = []
    all_acc = []
    
    # Sample a subset to get representative stats quickly
    indices = np.linspace(0, len(ds)-1, 50, dtype=int)
    for idx in indices:
        p = ds[idx]
        all_ecg.append(p['ecg_seq'])
        all_eda.append(p['eda_seq'])
        all_acc.append(p['acc_seq'])
    
    stats = {
        'ecg_m': np.mean(np.concatenate(all_ecg), axis=0),
        'ecg_s': np.std(np.concatenate(all_ecg), axis=0) + 1e-8,
        'eda_m': np.mean(np.concatenate(all_eda), axis=0),
        'eda_s': np.std(np.concatenate(all_eda), axis=0) + 1e-8,
        'acc_m': np.mean(np.concatenate(all_acc), axis=0),
        'acc_s': np.std(np.concatenate(all_acc), axis=0) + 1e-8
    }
    print("Stats computed.")
    return stats

def init_model():
    global model, dataset, stats
    print(f"Initializing model and WESAD dataset on {device}...")
    
    try:
        # Load Model
        config = {'hidden_dim': 16, 'num_heads': 4, 'num_layers': 2}
        model = FullPipelineModel(config).to(device)
        
        weights_path = os.path.join(PROJECT_ROOT, 'cross_eval', 'wesad_model.pth')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.eval()
            print("Model loaded successfully.")
        else:
            print(f"Warning: Weights not found at {weights_path}")

        # Load WESAD Data (Subject S2 for demo)
        data_root = os.path.join(PROJECT_ROOT, 'data', 'raw', 'WESAD')
        if os.path.exists(data_root):
            dataset = WESADDataset(subject_ids=[2], data_root=data_root)
            print(f"WESAD Dataset loaded with {len(dataset)} samples from S2.")
            stats = compute_dataset_stats(dataset)
        else:
            print(f"Warning: WESAD Data root not found at {data_root}")
    except Exception as e:
        print(f"Error during initialization: {e}")
        traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def get_samples():
    if dataset is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    try:
        # Explicitly find indices for each WESAD label (1: Baseline, 2: Stress, 3: Amusement)
        b_indices = [i for i, s in enumerate(dataset.samples) if int(s['label']) == 1]
        s_indices = [i for i, s in enumerate(dataset.samples) if int(s['label']) == 2]
        a_indices = [i for i, s in enumerate(dataset.samples) if int(s['label']) == 3]
        
        print(f"DEBUG: Found B:{len(b_indices)}, S:{len(s_indices)}, A:{len(a_indices)}")
        
        # Take 5 samples from each category, evenly spaced
        def get_subset(indices, count=5):
            if not indices: return []
            step = max(1, len(indices) // count)
            return indices[::step][:count]
            
        b_sel = get_subset(b_indices)
        s_sel = get_subset(s_indices)
        a_sel = get_subset(a_indices)
        
        # Interleave them: B1, S1, A1, B2, S2, A2...
        interleaved = []
        for i in range(5):
            if i < len(b_sel): interleaved.append(b_sel[i])
            if i < len(s_sel): interleaved.append(s_sel[i])
            if i < len(a_sel): interleaved.append(a_sel[i])
            
        samples_info = []
        for i in interleaved:
            label_val = int(dataset.samples[i]['label'])
            label_name = ["", "Baseline", "Stress", "Amusement"][label_val]
                
            samples_info.append({
                "id": i,
                "label": label_val - 1,
                "label_name": label_name
            })
            
        print(f"DEBUG: Returning {len(samples_info)} interleaved samples.")
        return jsonify(samples_info)
    except Exception as e:
        print(f"Error in get_samples: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/<int:sample_id>')
def predict(sample_id):
    if model is None or dataset is None:
        return jsonify({"error": "Model or Dataset not loaded"}), 500
    
    try:
        processed = dataset[sample_id]
        
        # Apply Normalization
        ecg_norm = (processed['ecg_seq'] - stats['ecg_m']) / stats['ecg_s']
        eda_norm = (processed['eda_seq'] - stats['eda_m']) / stats['eda_s']
        acc_norm = (processed['acc_seq'] - stats['acc_m']) / stats['acc_s']

        # Build coefficients
        def build_coeffs(data, times):
            d_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(0)
            d_in = add_intensity_channel(d_tensor, t_tensor)
            return torchcde.natural_cubic_coeffs(d_in)

        ecg_coeffs = build_coeffs(ecg_norm, processed['ecg_time'])
        eda_coeffs = build_coeffs(eda_norm, processed['eda_time'])
        acc_coeffs = build_coeffs(acc_norm, processed['acc_time'])
        
        ecg_p = torchcde.CubicSpline(ecg_coeffs).to(device)
        eda_p = torchcde.CubicSpline(eda_coeffs).to(device)
        acc_p = torchcde.CubicSpline(acc_coeffs).to(device)
        
        with torch.no_grad():
            logits = model(ecg_p, eda_p, acc_p, timeline)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = int(torch.argmax(logits, dim=1).cpu().item())
        
        raw_sample = dataset.samples[sample_id]
        
        return jsonify({
            "prediction": prediction,
            "prediction_name": ["Baseline", "Stress", "Amusement"][prediction],
            "probabilities": probs.tolist(),
            "signals": {
                "eda": raw_sample['eda'][::4].tolist(),
                "acc": np.linalg.norm(raw_sample['acc'], axis=1)[::32].tolist(),
                "ibi": processed['ecg_seq'].flatten().tolist()
            }
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    app.run(debug=False, port=5000, threaded=False)
