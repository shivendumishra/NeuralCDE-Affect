# 🧠 Neural CDE for Multimodal Emotion Recognition

> **A continuous-time deep learning framework for robust emotion recognition from irregular physiological signals.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Thesis_Submitted-success)

## 📌 Overview

This repository contains the official implementation of the Master's Thesis: **"Neural Controlled Differential Equations for Multimodal Emotion Recognition"**.

Traditional emotion recognition models (RNNs, LSTMs) struggle with **irregularly sampled data** and **missing values**, often requiring lossy interpolation. This project introduces a novel framework based on **Neural Controlled Differential Equations (Neural CDEs)** that treats physiological signals (ECG, EDA, ACC) as continuous control paths, allowing for:

*   ✅ **Native handling of irregular time-series** (no fixed sampling rate needed).
*   ✅ **Robustness to missing data** (up to 30% drop-out).
*   ✅ **Superior accuracy** (91.2% on WESAD dataset) compared to discrete-time baselines.

---

## 🚀 Key Features

*   **Continuous-Time Modeling**: Uses Natural Cubic Splines to build continuous paths from discrete observations.
*   **Neural CDE Layer**: Solves a controlled differential equation driven by the signal path to evolve hidden states.
*   **Multimodal Transformer**: Fuses embeddings from ECG, EDA, and ACC using cross-modal attention.
*   **Interactive Demo**: Includes a Flask-based web application to visualize signals and predictions in real-time.

---

## 📂 Project Structure

The codebase is organized for clarity and reproducibility:

```plaintext
Major_Project/
├── src/                      # 🧠 Core Implementation
│   ├── neural_cde/           # Neural CDE layer & vector fields
│   ├── fusion_transformer/   # Multimodal attention mechanism
│   ├── continuous_path/      # Spline interpolation logic
│   ├── training/             # Training loops & data loaders
│   └── ...
├── scripts/                  # 🛠️ Utilities & Verification
│   ├── cross_eval/           # Cross-dataset evaluation (AffectiveROAD)
│   └── verify_steps/         # Step-by-step pipeline verification
├── docs/                     # 📄 Documentation
│   ├── latex/                # Full Thesis LaTeX Source
│   └── references/           # Bibliography
├── demo_app/                 # 💻 Web Interface
│   ├── backend.py            # Flask server
│   ├── templates/            # HTML UI
│   └── static/               # CSS/JS assets
├── data/                     # 💾 Dataset Storage (Excluded from Git)
└── results/                  # 📊 Generated Plots & Models
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/shivendumishra/Neural-CDE-Thesis.git
cd Neural-CDE-Thesis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (LOSO)
Run the full Leave-One-Subject-Out training pipeline on the WESAD dataset:
```bash
python main.py --mode train
```

### 4. Run the Interactive Demo
Launch the web dashboard to visualize model predictions on sample data:
```bash
python demo_app/backend.py
```
> Open your browser at `http://127.0.0.1:5000` to see the interface.

---

## 📊 Performance Results

Our Neural CDE framework outperforms state-of-the-art discrete baselines on the WESAD dataset:

| Model | Accuracy (%) | Macro F1 |
| :--- | :---: | :---: |
| CNN-LSTM | 82.4 | 0.807 |
| GRU (Late Fusion) | 79.6 | 0.781 |
| Transformer | 85.1 | 0.839 |
| Neural ODE | 87.3 | 0.862 |
| **Neural CDE (Ours)** | **91.2** | **0.906** |

---

## 🎓 Thesis & Citation

The complete thesis document, detailing the mathematical foundations and extensive experiments, is available here:

📄 **[Read the Full Thesis (PDF)](docs/latex/neural_cde_thesis.pdf)**

If you find this work useful, please cite:

```bibtex
@mastersthesis{mishra2025neuralcde,
  title={Neural Controlled Differential Equations for Multimodal Emotion Recognition},
  author={Shivendu Mishra},
  school={National Institute of Technology, Tiruchirappalli},
  year={2025}
}
```

---

<p align="center">
  Made with ❤️ by Shivendu Mishra
</p>
