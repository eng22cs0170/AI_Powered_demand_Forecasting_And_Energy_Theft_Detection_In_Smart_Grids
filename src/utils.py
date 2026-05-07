"""
Utility functions for the Smart Grid AI project.
Handles configuration loading, plotting, device setup, and common helpers.
"""

import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file, no GUI popup)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

def load_config(config_path="config/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config=None):
    """Get the appropriate compute device (CPU/CUDA)."""
    if config and config.get('training', {}).get('device', 'auto') != 'auto':
        return torch.device(config['training']['device'])
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_directories(config):
    """Create all necessary output directories."""
    dirs = [
        "data/processed",
        "data/sequences",
        "models",
        "results/plots",
        "results/metrics",
        "results/reports",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[✓] All directories created/verified.")


# ============================================================
# Plotting Helpers
# ============================================================

def setup_plot_style():
    """Set up consistent matplotlib style for all plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 100,
    })


def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', color='#FF5722', linewidth=2)
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Forecasting loss
    axes[1].plot(history['train_forecast_loss'], label='Train', color='#4CAF50', linewidth=2)
    axes[1].plot(history['val_forecast_loss'], label='Val', color='#FF9800', linewidth=2)
    axes[1].set_title('Forecasting Loss (MSE)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Theft detection loss
    axes[2].plot(history['train_theft_loss'], label='Train', color='#9C27B0', linewidth=2)
    axes[2].plot(history['val_theft_loss'], label='Val', color='#E91E63', linewidth=2)
    axes[2].set_title('Theft Detection Loss (BCE)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[✓] Training history plot saved to {save_path}")
    plt.close()


def plot_predictions_vs_actual(actual, predicted, title="Predictions vs Actual", save_path=None):
    """Plot predicted vs actual values for demand forecasting."""
    setup_plot_style()
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Moving-average overlays help compare trend without hiding raw values
    win = min(12, max(3, len(actual) // 20))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    actual_smooth = np.convolve(actual, kernel, mode='same') if len(actual) >= win else actual
    predicted_smooth = np.convolve(predicted, kernel, mode='same') if len(predicted) >= win else predicted

    # Time series comparison
    axes[0].plot(actual, label='Actual', color='#2196F3', linewidth=1.4, alpha=0.8)
    axes[0].plot(predicted, label='Predicted', color='#FF5722', linewidth=1.4, alpha=0.8)
    axes[0].plot(actual_smooth, label=f'Actual (MA-{win})', color='#2196F3', linewidth=2.0, alpha=0.35)
    axes[0].plot(predicted_smooth, label=f'Predicted (MA-{win})', color='#FF5722', linewidth=2.0, alpha=0.35)
    axes[0].set_title(f'{title} - Time Series')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('MW')
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(actual, predicted, alpha=0.55, color='#4CAF50', s=22, edgecolors='none')
    min_val = float(min(np.min(actual), np.min(predicted)))
    max_val = float(max(np.max(actual), np.max(predicted)))
    pad = 0.05 * (max_val - min_val + 1e-8)
    lo, hi = min_val - pad, max_val + pad
    axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_title(f'{title} - Scatter Plot')
    axes[1].set_xlabel('Actual MW')
    axes[1].set_ylabel('Predicted MW')
    axes[1].grid(alpha=0.25)

    # Display core regression quality directly on the scatter panel
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    axes[1].text(
        0.98, 0.02,
        f'RMSE: {rmse:.2f}\nR²: {r2:.3f}',
        transform=axes[1].transAxes,
        ha='right', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9)
    )
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[✓] Prediction plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix for theft detection."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Theft'],
                yticklabels=['Normal', 'Theft'], ax=ax)
    ax.set_title('Theft Detection - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[✓] Confusion matrix saved to {save_path}")
    plt.close()


def plot_theft_probabilities(probabilities, labels, threshold=0.5, save_path=None):
    """Plot theft probability distribution."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(probabilities[labels == 0], bins=50, alpha=0.7, color='#4CAF50', label='Normal')
    axes[0].hist(probabilities[labels == 1], bins=50, alpha=0.7, color='#F44336', label='Theft')
    axes[0].axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    axes[0].set_title('Theft Probability Distribution')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Sorted probabilities
    sorted_probs = np.sort(probabilities)[::-1]
    colors = ['#F44336' if p >= threshold else '#4CAF50' for p in sorted_probs]
    axes[1].bar(range(len(sorted_probs)), sorted_probs, color=colors, width=1.0)
    axes[1].axhline(y=threshold, color='black', linestyle='--', linewidth=2)
    axes[1].set_title('Sorted Theft Probabilities')
    axes[1].set_xlabel('Meter Index (sorted)')
    axes[1].set_ylabel('Theft Probability')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[✓] Theft probability plot saved to {save_path}")
    plt.close()


# ============================================================
# Metrics Saving
# ============================================================

def save_metrics(metrics, filepath):
    """Save evaluation metrics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert numpy types to native Python types
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v

    clean['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"[✓] Metrics saved to {filepath}")


def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        elif isinstance(v, (list, np.ndarray)):
            continue  # Skip arrays in pretty print
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'='*50}\n")
