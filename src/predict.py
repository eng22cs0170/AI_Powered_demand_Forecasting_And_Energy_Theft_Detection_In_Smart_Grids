"""
Prediction Module
Handles inference for both demand forecasting and theft detection.
Provides functions for single predictions and batch predictions.
"""

import torch
import numpy as np
import joblib
import os


def load_trained_model(model, model_path, device=None):
    """Load trained model weights."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"[✓] Model loaded from {model_path}")
    else:
        print(f"[!] Model file not found: {model_path}")

    return model


def predict_demand(model, input_sequence, scaler=None, target_idx=0, device=None):
    """
    Predict electricity demand (MW) for the next time step.

    Args:
        model: trained SmartGridModel
        input_sequence: numpy array (seq_len, features) or (batch, seq_len, features)
        scaler: fitted scaler for inverse transform
        target_idx: target column index

    Returns:
        predicted MW value(s)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    if input_sequence.ndim == 2:
        input_sequence = input_sequence[np.newaxis, :]  # Add batch dim

    X_tensor = torch.FloatTensor(input_sequence).to(device)

    with torch.no_grad():
        output = model(X_tensor, task='forecast')
        prediction = output['forecast'].cpu().numpy()

    # Inverse transform
    if scaler is not None:
        n_features = scaler.n_features_in_
        pred_full = np.zeros((len(prediction), n_features))
        pred_full[:, target_idx] = prediction
        prediction = scaler.inverse_transform(pred_full)[:, target_idx]

    return prediction


def predict_theft(model, input_sequence, threshold=0.5, device=None):
    """
    Predict theft probability for given consumption sequence.

    Args:
        model: trained SmartGridModel
        input_sequence: numpy array (seq_len, features) or (batch, seq_len, features)
        threshold: classification threshold

    Returns:
        dict with 'probability', 'is_theft', 'alert_level'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    if input_sequence.ndim == 2:
        input_sequence = input_sequence[np.newaxis, :]

    X_tensor = torch.FloatTensor(input_sequence).to(device)

    with torch.no_grad():
        output = model(X_tensor, task='theft')
        probabilities = output['theft'].cpu().numpy()

    results = []
    for prob in probabilities:
        if prob >= 0.8:
            alert = 'CRITICAL'
        elif prob >= threshold:
            alert = 'WARNING'
        elif prob >= 0.3:
            alert = 'WATCH'
        else:
            alert = 'NORMAL'

        results.append({
            'probability': float(prob),
            'is_theft': bool(prob >= threshold),
            'alert_level': alert
        })

    return results


def batch_predict(model, X_data, task='both', scaler=None, target_idx=0,
                  threshold=0.5, device=None):
    """
    Run batch predictions for demand and/or theft.

    Args:
        model: trained SmartGridModel
        X_data: numpy array (batch, seq_len, features)
        task: 'forecast', 'theft', or 'both'
        scaler: demand scaler for inverse transform
        target_idx: target column index
        threshold: theft threshold

    Returns:
        dict with predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    X_tensor = torch.FloatTensor(X_data).to(device)

    results = {}

    with torch.no_grad():
        output = model(X_tensor, task=task)

        if 'forecast' in output:
            preds = output['forecast'].cpu().numpy()
            if scaler is not None:
                n_features = scaler.n_features_in_
                pred_full = np.zeros((len(preds), n_features))
                pred_full[:, target_idx] = preds
                preds = scaler.inverse_transform(pred_full)[:, target_idx]
            results['demand_predictions'] = preds

        if 'theft' in output:
            probs = output['theft'].cpu().numpy()
            results['theft_probabilities'] = probs
            results['theft_labels'] = (probs >= threshold).astype(int)
            results['alert_levels'] = []
            for p in probs:
                if p >= 0.8:
                    results['alert_levels'].append('CRITICAL')
                elif p >= threshold:
                    results['alert_levels'].append('WARNING')
                elif p >= 0.3:
                    results['alert_levels'].append('WATCH')
                else:
                    results['alert_levels'].append('NORMAL')

    return results
