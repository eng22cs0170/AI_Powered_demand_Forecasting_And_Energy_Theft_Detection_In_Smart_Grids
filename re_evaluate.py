import os
import json
import numpy as np
import torch
import joblib
import sys

# Change to project dir
project_dir = r"c:\Users\Rajshekar\Desktop\major-project"
os.chdir(project_dir)
sys.path.insert(0, project_dir)

from src.utils import load_config, get_device, plot_confusion_matrix, plot_theft_probabilities
from src.model import create_model
from src.evaluate import evaluate_theft_detection

def main():
    print("Loading config and device...")
    config = load_config()
    device = get_device(config)

    print("Loading theft sequences...")
    t_data = np.load("data/sequences/theft_sequences.npz", allow_pickle=True)
    X_test_t, y_test_t = t_data['X_test'], t_data['y_test']
    X_val_t, y_val_t = t_data['X_val'], t_data['y_val']

    print("Loading model...")
    model = create_model(config, device)
    # Trigger lazy initialization
    dummy = torch.randn(1, X_test_t.shape[1], X_test_t.shape[2]).to(device)
    _ = model(dummy, task='theft')
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device, weights_only=True))
    model.eval()

    print("Evaluating theft detection...")
    threshold = config['evaluation']['theft_probability_threshold']
    theft_metrics, theft_probs = evaluate_theft_detection(
        model, X_test_t, y_test_t, threshold, device, X_val=X_val_t, y_val=y_val_t
    )

    opt_t = theft_metrics.get('Threshold', threshold)

    print(f"New Accuracy: {theft_metrics.get('Accuracy')}")

    print("Saving metrics and plots...")
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/theft_metrics.json', 'w') as f:
        json.dump(theft_metrics, f, indent=4)

    os.makedirs('results/plots', exist_ok=True)
    from sklearn.metrics import confusion_matrix
    theft_preds = (theft_probs >= opt_t).astype(int)
    cm = confusion_matrix(y_test_t.astype(int), theft_preds)

    plot_confusion_matrix(cm, save_path='results/plots/theft_confusion_matrix.png')
    plot_theft_probabilities(theft_probs, y_test_t, threshold=opt_t, save_path='results/plots/theft_probabilities.png')

    print("Done! Check dashboard.")

if __name__ == "__main__":
    main()
