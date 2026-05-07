"""
Main Entry Point
=================
AI-Powered Demand Forecasting and Energy Theft Detection in Smart Grids

Runs the complete pipeline:
  1. Data Ingestion
  2. Feature Engineering
  3. Data Preprocessing
  4. Model Creation
  5. Training
  6. Evaluation
  7. Visualization
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_config, create_directories, get_device,
    plot_training_history, plot_predictions_vs_actual,
    plot_confusion_matrix, plot_theft_probabilities,
    save_metrics, print_metrics
)
from src.data_ingestion import run_ingestion
from src.feature_engineering import engineer_demand_features, engineer_theft_features
from src.data_preprocessing import (
    preprocess_demand, preprocess_theft, save_processed_csv
)
from src.model import create_model
from src.train import train_model
from src.evaluate import run_full_evaluation
import numpy as np


def main():
    print("╔" + "═" * 58 + "╗")
    print("║  AI-Powered Demand Forecasting & Energy Theft Detection ║")
    print("║              Smart Grid Analytics System                ║")
    print("║           Hybrid GRU + TCN Architecture                 ║")
    print("╚" + "═" * 58 + "╝\n")

    # ============================================================
    # STEP 0: Load Configuration
    # ============================================================
    config = load_config()
    create_directories(config)
    device = get_device(config)
    print(f"[✓] Configuration loaded. Device: {device}\n")

    # ============================================================
    # STEP 1: Data Ingestion
    # ============================================================
    demand_features, theft_features = run_ingestion(config)

    # ============================================================
    # STEP 2: Feature Engineering
    # ============================================================
    demand_features = engineer_demand_features(demand_features, config)
    theft_features = engineer_theft_features(theft_features, config)

    # Save processed CSVs
    save_processed_csv(demand_features, theft_features, config)

    # ============================================================
    # STEP 3: Data Preprocessing (Normalization + Sequences)
    # ============================================================
    demand_result = preprocess_demand(demand_features, config)
    (X_train_d, y_train_d), (X_val_d, y_val_d), (X_test_d, y_test_d), demand_feat_cols, demand_scaler = demand_result

    theft_result = preprocess_theft(theft_features, config)
    (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t), theft_feat_cols, theft_scaler = theft_result

    print(f"\n{'='*60}")
    print(f" DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Demand - Train: {X_train_d.shape}, Val: {X_val_d.shape}, Test: {X_test_d.shape}")
    print(f"  Demand - Features: {len(demand_feat_cols)}, Seq length: {X_train_d.shape[1]}")
    print(f"  Theft  - Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
    print(f"  Theft  - Features: {len(theft_feat_cols)}, Seq length: {X_train_t.shape[1]}")
    print(f"  Theft  - Train label dist: {np.mean(y_train_t)*100:.1f}% theft")
    print()

    # ============================================================
    # STEP 4: Create Model
    # ============================================================
    model = create_model(config, device)

    # ============================================================
    # STEP 5: Training
    # ============================================================
    history = train_model(
        model, config,
        demand_data=((X_train_d, y_train_d), (X_val_d, y_val_d)),
        theft_data=((X_train_t, y_train_t), (X_val_t, y_val_t)),
        device=device
    )

    # Plot training history
    plot_training_history(history, save_path='results/plots/training_history.png')

    # ============================================================
    # STEP 6: Evaluation
    # ============================================================
    # Find target index
    target_col = config['preprocessing']['demand']['target_column']
    target_idx = demand_feat_cols.index(target_col)

    threshold = config['evaluation']['theft_probability_threshold']

    all_metrics, predictions = run_full_evaluation(
        model,
        demand_test_data=(X_test_d, y_test_d),
        theft_test_data=(X_test_t, y_test_t),
        demand_scaler=demand_scaler,
        target_idx=target_idx,
        threshold=threshold,
        device=device,
        theft_val_data=(X_val_t, y_val_t)
    )

    # Print and save metrics
    if 'forecasting' in all_metrics:
        print_metrics(all_metrics['forecasting'], "Demand Forecasting Metrics")
        save_metrics(all_metrics['forecasting'], 'results/metrics/forecast_metrics.json')

    if 'theft_detection' in all_metrics:
        print_metrics(all_metrics['theft_detection'], "Theft Detection Metrics")
        save_metrics(all_metrics['theft_detection'], 'results/metrics/theft_metrics.json')

    # ============================================================
    # STEP 7: Visualization
    # ============================================================

    # Forecasting plots
    if 'forecast_predictions' in predictions:
        plot_predictions_vs_actual(
            predictions['forecast_actuals'],
            predictions['forecast_predictions'],
            title='Demand Forecasting',
            save_path='results/plots/forecast_predictions.png'
        )

    # Theft detection plots
    if 'theft_probabilities' in predictions:
        from sklearn.metrics import confusion_matrix as cm_func

        # Use optimal threshold from evaluation
        optimal_threshold = all_metrics.get('theft_detection', {}).get('Threshold', threshold)

        # Confusion matrix
        theft_preds = (predictions['theft_probabilities'] >= optimal_threshold).astype(int)
        cm = cm_func(predictions['theft_actuals'].astype(int), theft_preds)
        plot_confusion_matrix(cm, save_path='results/plots/theft_confusion_matrix.png')

        # Probability distributions
        plot_theft_probabilities(
            predictions['theft_probabilities'],
            predictions['theft_actuals'],
            threshold=optimal_threshold,
            save_path='results/plots/theft_probabilities.png'
        )

    # ============================================================
    # DONE!
    # ============================================================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║                  PIPELINE COMPLETE!                     ║")
    print("╠" + "═" * 58 + "╣")
    print("║  Outputs:                                               ║")
    print("║    📁 data/processed/   - Cleaned datasets              ║")
    print("║    📁 data/sequences/   - Training sequences            ║")
    print("║    📁 models/           - Trained model & scalers       ║")
    print("║    📁 results/plots/    - Visualizations                ║")
    print("║    📁 results/metrics/  - Evaluation metrics            ║")
    print("║                                                         ║")
    print("║  Next: Run the dashboard with:                          ║")
    print("║    streamlit run dashboard/app.py                       ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
