# Stress Detection Model Demo

This is a small web application designed to demonstrate the performance of your stress detection model to your teacher.

## Features
- **Real-time Inference**: Select a sample from the Affective Road dataset and see the model's prediction instantly.
- **Multimodal Visualization**: View the EDA (Electrodermal Activity) and ACC (Accelerometer) signals for each sample.
- **Confidence Scores**: See how confident the model is in its prediction.
- **Ground Truth Comparison**: Compare the model's prediction with the actual label.

## How to Run
1. Ensure you have the required dependencies:
   ```bash
   pip install flask flask-cors torch torchcde numpy pandas tqdm matplotlib scikit-learn
   ```
2. Run the backend:
   ```bash
   python demo_app/backend.py
   ```
3. Open your browser and navigate to:
   `http://127.0.0.1:5000`

## Project Structure
- `backend.py`: Flask server that handles model loading and inference.
- `templates/index.html`: The main UI structure.
- `static/style.css`: Premium styling with glassmorphism and animations.
- `static/script.js`: Frontend logic for interactivity and charting.
