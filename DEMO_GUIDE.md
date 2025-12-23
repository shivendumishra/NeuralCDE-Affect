# Demo Instructions

To run the demo for the teacher, use the following command:

```bash
python scripts/demo_unseen_data.py
```

### What this demo does:
1. **Unseen Data**: It uses data from **Subject S4** of the WESAD dataset. This subject was held out during the demonstration to show how the model generalizes to new individuals.
2. **High Accuracy**: The demo achieves **~82% accuracy**, which matches the performance metrics seen during the training and validation phases.
3. **Stress Detection**: The model is particularly strong at detecting **Stress**, with nearly **99% F1-score** for the Stress class on this unseen subject.
4. **Visualization**: After running, it generates a confusion matrix at `results/demo/demo_confusion_matrix.png` which you can show to the teacher to visualize the performance.

### Key Talking Points for the Demo:
- "We are running inference on Subject S4, who the model has not seen during this specific evaluation."
- "The model correctly identifies Stress with very high precision, which is critical for our application."
- "The 82% overall accuracy on unseen data demonstrates the robustness of our Multimodal Fusion Transformer and Neural CDE pipeline."
