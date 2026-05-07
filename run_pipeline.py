"""
Fast Pipeline Runner — with output to file
"""

import sys
import os
import io
import warnings
import time

# Redirect all output to file
log_file = open('pipeline_output.txt', 'w', encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import joblib

from src.utils import (
    load_config, create_directories, get_device,
    plot_training_history, plot_predictions_vs_actual,
    plot_confusion_matrix, plot_theft_probabilities,
    save_metrics, print_metrics
)
from src.data_preprocessing import (
    create_scaler, normalize_features, create_sequences,
    create_meter_sequences, label_theft, split_data
)
from src.model import create_model
from src.train import train_model
from src.evaluate import run_full_evaluation


def main():
    pipeline_start = time.time()

    print("=" * 62)
    print("  AI-Powered Demand Forecasting & Energy Theft Detection")
    print("  Smart Grid Analytics - Fast Pipeline")
    print("  Architecture: Hybrid GRU + TCN (Multitask)")
    print("=" * 62)
    print()

    # STEP 0: Configuration
    config = load_config()
    create_directories(config)
    device = get_device(config)
    print(f"[OK] Configuration loaded. Device: {device}\n")

    # STEP 1: Load Preprocessed CSVs
    print("=" * 60)
    print(" STEP 1: LOADING PREPROCESSED DATA")
    print("=" * 60)

    demand_path = config['data']['demand_processed']
    theft_path = 'data/processed/theft_processed_fixed.csv'

    print(f"[->] Loading demand: {demand_path}")
    demand_df = pd.read_csv(demand_path)
    print(f"    Shape: {demand_df.shape}")

    print(f"[->] Loading theft: {theft_path}")
    theft_df = pd.read_csv(theft_path)
    print(f"    Shape: {theft_df.shape}\n")

    # STEP 2a: DEMAND PREPROCESSING
    print("=" * 60)
    print(" STEP 2a: DEMAND PREPROCESSING")
    print("=" * 60)

    prep_demand = config['preprocessing']['demand']
    demand_feature_cols = [c for c in demand_df.columns
                           if c not in ['DateTime', 'Year']
                           and demand_df[c].dtype in ['float64', 'int64', 'int32', 'float32']]
    print(f"[->] Features ({len(demand_feature_cols)}): {demand_feature_cols}")

    demand_df[demand_feature_cols] = (demand_df[demand_feature_cols]
                                       .fillna(method='ffill')
                                       .fillna(method='bfill')
                                       .fillna(0))

    demand_scaler = create_scaler(prep_demand.get('scaler', 'minmax'))
    demand_normalized, demand_scaler = normalize_features(
        demand_df, demand_feature_cols, demand_scaler, fit=True
    )
    os.makedirs('models', exist_ok=True)
    joblib.dump(demand_scaler, 'models/demand_scaler.pkl')

    target_col = prep_demand['target_column']
    target_idx = demand_feature_cols.index(target_col)
    print(f"    Target: '{target_col}' at index {target_idx}")

    seq_length_d = prep_demand['sequence_length']
    X_d, y_d = create_sequences(demand_normalized, target_idx, seq_length_d)
    print(f"    Sequences: X={X_d.shape}, y={y_d.shape}")

    (X_train_d, y_train_d), (X_val_d, y_val_d), (X_test_d, y_test_d) = split_data(
        X_d, y_d, prep_demand['train_split'], prep_demand['val_split']
    )

    seq_path_d = config['data']['demand_sequences']
    os.makedirs(os.path.dirname(seq_path_d), exist_ok=True)
    np.savez(seq_path_d, X_train=X_train_d, y_train=y_train_d,
             X_val=X_val_d, y_val=y_val_d, X_test=X_test_d, y_test=y_test_d,
             feature_cols=demand_feature_cols, target_idx=target_idx)
    print(f"[OK] Demand sequences saved\n")

    # STEP 2b: THEFT PREPROCESSING
    print("=" * 60)
    print(" STEP 2b: THEFT PREPROCESSING")
    print("=" * 60)

    prep_theft = config['preprocessing']['theft']
    theft_df = label_theft(theft_df, threshold_std=prep_theft.get('theft_threshold_std', 2.0))

    theft_feature_cols = ['Present_Reading', 'Previous_Reading', 'Difference',
                          'Meter_Constant', 'Net_Energy_MU', 'Consumption_Ratio',
                          'Meter_Status_OK', 'Month_Num']
    theft_feature_cols = [c for c in theft_feature_cols if c in theft_df.columns]
    print(f"[->] Features ({len(theft_feature_cols)}): {theft_feature_cols}")

    # Impute NaN
    discrete_cols = ['Month_Num']
    for col in theft_feature_cols:
        if col in discrete_cols:
            meter_mode = theft_df.groupby('Energy_Meter_ID')[col].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            theft_df[col] = theft_df[col].fillna(meter_mode)
            global_mode = theft_df[col].mode().iloc[0] if not theft_df[col].mode().empty else 0
            theft_df[col] = theft_df[col].fillna(global_mode)
        else:
            meter_med = theft_df.groupby('Energy_Meter_ID')[col].transform('median')
            theft_df[col] = theft_df[col].fillna(meter_med)
            theft_df[col] = theft_df[col].fillna(theft_df[col].median())
    theft_df[theft_feature_cols] = theft_df[theft_feature_cols].fillna(0)

    theft_scaler = create_scaler(prep_theft.get('scaler', 'minmax'))
    theft_df[theft_feature_cols] = theft_scaler.fit_transform(theft_df[theft_feature_cols])
    joblib.dump(theft_scaler, 'models/theft_scaler.pkl')

    seq_length_t = prep_theft['sequence_length']
    X_t, y_t, meter_ids = create_meter_sequences(
        theft_df, theft_feature_cols, seq_length_t, label_col='Theft_Label'
    )
    print(f"    Sequences: X={X_t.shape}, y={y_t.shape}")
    print(f"    Theft ratio: {y_t.mean()*100:.1f}%")

    (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t) = split_data(
        X_t, y_t, prep_theft['train_split'], prep_theft['val_split']
    )

    # SMOTE
    theft_count = int(y_train_t.sum())
    normal_count = len(y_train_t) - theft_count
    print(f"    Before SMOTE: Normal={normal_count}, Theft={theft_count}")

    if theft_count > 0 and theft_count < normal_count:
        try:
            from imblearn.over_sampling import SMOTE
            n_s, s_l, n_f = X_train_t.shape
            smote = SMOTE(sampling_strategy=0.3,
                          k_neighbors=min(5, theft_count - 1),
                          random_state=42)
            X_flat_res, y_train_t = smote.fit_resample(X_train_t.reshape(n_s, -1), y_train_t)
            X_train_t = X_flat_res.reshape(-1, s_l, n_f).astype(np.float32)
            y_train_t = y_train_t.astype(np.float32)
            print(f"    After SMOTE:  Normal={len(y_train_t)-int(y_train_t.sum())}, "
                  f"Theft={int(y_train_t.sum())}")
        except ImportError:
            print("    [!] imbalanced-learn not installed. Skipping SMOTE.")
        except Exception as e:
            print(f"    [!] SMOTE failed: {e}")

    seq_path_t = config['data']['theft_sequences']
    os.makedirs(os.path.dirname(seq_path_t), exist_ok=True)
    np.savez(seq_path_t, X_train=X_train_t, y_train=y_train_t,
             X_val=X_val_t, y_val=y_val_t, X_test=X_test_t, y_test=y_test_t,
             feature_cols=theft_feature_cols)
    print(f"[OK] Theft sequences saved\n")

    # DATA SUMMARY
    print("=" * 60)
    print(" DATA SUMMARY")
    print("=" * 60)
    print(f"  Demand - Train: {X_train_d.shape}, Val: {X_val_d.shape}, Test: {X_test_d.shape}")
    print(f"  Theft  - Train: {X_train_t.shape}, Val: {X_val_t.shape}, Test: {X_test_t.shape}")
    print(f"  Theft train dist: {np.mean(y_train_t)*100:.1f}% theft\n")

    # STEP 3: MODEL
    print("=" * 60)
    print(" STEP 3: MODEL CREATION")
    print("=" * 60)
    model = create_model(config, device)
    dummy = torch.randn(1, X_train_d.shape[1], X_train_d.shape[2]).to(device)
    _ = model(dummy, task='forecast')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}\n")

    # STEP 4: TRAINING
    history = train_model(
        model, config,
        demand_data=((X_train_d, y_train_d), (X_val_d, y_val_d)),
        theft_data=((X_train_t, y_train_t), (X_val_t, y_val_t)),
        device=device
    )
    plot_training_history(history, save_path='results/plots/training_history.png')

    # STEP 5: EVALUATION
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

    if 'forecasting' in all_metrics:
        print_metrics(all_metrics['forecasting'], "DEMAND FORECASTING METRICS")
        save_metrics(all_metrics['forecasting'], 'results/metrics/forecast_metrics.json')

    if 'theft_detection' in all_metrics:
        print_metrics(all_metrics['theft_detection'], "THEFT DETECTION METRICS")
        save_metrics(all_metrics['theft_detection'], 'results/metrics/theft_metrics.json')

    # STEP 6: PLOTS
    if 'forecast_predictions' in predictions:
        plot_predictions_vs_actual(
            predictions['forecast_actuals'], predictions['forecast_predictions'],
            title='Demand Forecasting', save_path='results/plots/forecast_predictions.png'
        )
    if 'theft_probabilities' in predictions:
        from sklearn.metrics import confusion_matrix as cm_func
        opt_t = all_metrics.get('theft_detection', {}).get('Threshold', threshold)
        theft_preds = (predictions['theft_probabilities'] >= opt_t).astype(int)
        cm = cm_func(predictions['theft_actuals'].astype(int), theft_preds)
        plot_confusion_matrix(cm, save_path='results/plots/theft_confusion_matrix.png')
        plot_theft_probabilities(
            predictions['theft_probabilities'], predictions['theft_actuals'],
            threshold=opt_t, save_path='results/plots/theft_probabilities.png'
        )

    # FINAL SUMMARY
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 62)
    print("  PIPELINE COMPLETE!")
    print("=" * 62)
    print(f"  Total time: {total_time:.1f}s\n")
    print("  RESULTS:")
    print("  " + "-" * 58)

    if 'forecasting' in all_metrics:
        fm = all_metrics['forecasting']
        print(f"  DEMAND FORECASTING:")
        print(f"    RMSE:            {fm['RMSE']:.4f}")
        print(f"    MAE:             {fm['MAE']:.4f}")
        print(f"    R2 Score:        {fm['R2_Score']:.4f}")
        print(f"    MAPE:            {fm['MAPE']:.2f}%")
        print(f"    RMSE Accuracy:   {fm['RMSE_Accuracy_Pct']:.2f}%")
        print(f"    MAE Accuracy:    {fm['MAE_Accuracy_Pct']:.2f}%")
        print(f"    Within +/-5 MW:  {fm['Within_5MW']:.1f}%")
        print(f"    Within +/-10 MW: {fm['Within_10MW']:.1f}%")
        print()

    if 'theft_detection' in all_metrics:
        tm = all_metrics['theft_detection']
        print(f"  THEFT DETECTION:")
        print(f"    Accuracy:   {tm['Accuracy']*100:.2f}%")
        print(f"    Precision:  {tm['Precision']:.4f}")
        print(f"    Recall:     {tm['Recall']:.4f}")
        print(f"    F1 Score:   {tm['F1_Score']:.4f}")
        print(f"    AUC-ROC:    {tm['AUC_ROC']:.4f}")
        print(f"    Threshold:  {tm['Threshold']:.2f}")
        print()

    print("  SAVED FILES:")
    print("    models/best_model.pth       - Trained model")
    print("    results/plots/              - Visualizations")
    print("    results/metrics/            - JSON metrics")
    print("=" * 62)

    log_file.flush()


if __name__ == "__main__":
    main()
    log_file.close()
    # Print to real stdout that we're done
    real_stdout = open('CON', 'w')
    real_stdout.write("Pipeline complete! Results in pipeline_output.txt\n")
    real_stdout.close()
