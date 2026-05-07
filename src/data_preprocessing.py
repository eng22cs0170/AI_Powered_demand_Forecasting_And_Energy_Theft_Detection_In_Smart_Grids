"""
Data Preprocessing Module
Handles cleaning, normalization, sliding window creation, and labeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import joblib


# ============================================================
# Normalization
# ============================================================

def create_scaler(scaler_type='minmax'):
    """Create a scaler instance."""
    if scaler_type == 'standard':
        return StandardScaler()
    return MinMaxScaler(feature_range=(0, 1))


def normalize_features(df, feature_cols, scaler=None, fit=True):
    """
    Normalize selected feature columns.
    Returns normalized data and fitted scaler.
    """
    data = df[feature_cols].values.astype(np.float32)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    if fit:
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)

    return normalized, scaler


# ============================================================
# Sliding Window Sequences
# ============================================================

def create_sequences(data, target_idx, seq_length, timestamps=None, max_gap_hours=None):
    """
    Create sliding window sequences for time-series forecasting.
    Optionally skips windows that span large time gaps.

    Args:
        data: normalized feature array (n_samples, n_features)
        target_idx: column index of the target variable
        seq_length: number of time steps per input sequence
        timestamps: optional array of datetime values for gap detection
        max_gap_hours: maximum allowed gap between consecutive timestamps
                       in a window (windows with larger gaps are discarded)

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)
    """
    X, y = [], []
    skipped = 0

    for i in range(len(data) - seq_length):
        # Check for time gaps within this window
        if timestamps is not None and max_gap_hours is not None:
            window_times = timestamps[i : i + seq_length + 1]
            time_diffs = pd.to_timedelta(pd.Series(window_times).diff().dropna())
            max_diff_hours = time_diffs.max().total_seconds() / 3600
            if max_diff_hours > max_gap_hours:
                skipped += 1
                continue  # Skip this window — it bridges a large gap

        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])

    if skipped > 0:
        print(f"    Skipped {skipped} windows due to time gaps > {max_gap_hours}h")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_meter_sequences(df, feature_cols, seq_length, label_col='Theft_Label'):
    """
    Create sequences grouped by meter for theft detection.
    Each meter's history is broken into sliding windows.

    Args:
        df: DataFrame with meter data
        feature_cols: columns to use as features
        seq_length: sliding window length
        label_col: theft label column name

    Returns:
        X: (n_sequences, seq_length, n_features)
        y: (n_sequences,)
        meter_ids: corresponding meter IDs for each sequence
    """
    X, y, meter_ids = [], [], []

    for meter_id, group in df.groupby('Energy_Meter_ID'):
        group = group.sort_values(['Year', 'Month_Num']).reset_index(drop=True)
        values = group[feature_cols].values.astype(np.float32)
        labels = group[label_col].values.astype(np.float32)

        if len(values) < seq_length:
            # Pad short sequences
            pad_len = seq_length - len(values)
            values = np.vstack([np.zeros((pad_len, values.shape[1]), dtype=np.float32), values])
            labels = np.concatenate([np.zeros(pad_len, dtype=np.float32), labels])

        for i in range(len(values) - seq_length):
            X.append(values[i : i + seq_length])
            y.append(labels[i + seq_length - 1])  # Classify the current window's final state
            meter_ids.append(meter_id)

        # Also add the last window
        if len(values) >= seq_length:
            X.append(values[-seq_length:])
            y.append(labels[-1])
            meter_ids.append(meter_id)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), meter_ids


# ============================================================
# Theft Labeling
# ============================================================

def label_theft(df, threshold_std=2.0):
    """
    Generate theft labels based on statistical anomaly detection.
    If 'Is_Theft' column explicitly exists (from synthetic data), use it directly.
    """
    print("[→] Generating theft labels...")
    df = df.copy()
    
    # If the ground-truth synthetic labels exist, use them directly instead of guessing!
    if 'Is_Theft' in df.columns:
        print("    [!] Found 'Is_Theft' ground-truth column! Overriding heuristic labeling.")
        df['Theft_Label'] = df['Is_Theft']
        theft_count = df['Theft_Label'].sum()
        total = len(df)
        print(f"    Total records: {total}")
        print(f"    Normal: {total - theft_count} ({(1 - theft_count/total)*100:.1f}%)")
        print(f"    Suspicious: {theft_count} ({theft_count/total*100:.1f}%)")
        print(f"[✓] Theft labeling complete.\n")
        return df

    df['Theft_Label'] = 0  # Default: Normal

    # --- Method 1: Z-score based anomaly detection per meter ---
    for meter_id, group in df.groupby('Energy_Meter_ID'):
        if len(group) < 3:
            continue

        diff = group['Difference'].values
        valid_mask = ~np.isnan(diff)
        if valid_mask.sum() < 3:
            continue

        mean_diff = np.nanmean(diff)
        std_diff = np.nanstd(diff)

        if std_diff > 0:
            z_scores = np.abs((diff - mean_diff) / std_diff)
            # Flag records with unusually LOW consumption (potential theft)
            low_consumption = (diff - mean_diff) / std_diff  # negative z = low consumption
            anomaly_mask = (low_consumption < -threshold_std) | (z_scores > threshold_std * 1.5)
            df.loc[group.index[anomaly_mask & valid_mask], 'Theft_Label'] = 1

    # --- Method 2: Negative or zero differences (impossible in normal operation) ---
    negative_mask = df['Difference'] < 0
    df.loc[negative_mask, 'Theft_Label'] = 1

    # --- Method 3: Extremely low consumption ratio ---
    if 'Consumption_Ratio' in df.columns:
        low_ratio = df['Consumption_Ratio'] < -0.01
        df.loc[low_ratio, 'Theft_Label'] = 1

    # --- Method 4: Meter status issues ---
    if 'Meter_Status_OK' in df.columns:
        df.loc[df['Meter_Status_OK'] == 0, 'Theft_Label'] = 1

    theft_count = df['Theft_Label'].sum()
    total = len(df)
    print(f"    Total records: {total}")
    print(f"    Normal: {total - theft_count} ({(1 - theft_count/total)*100:.1f}%)")
    print(f"    Suspicious: {theft_count} ({theft_count/total*100:.1f}%)")
    print(f"[✓] Theft labeling complete.\n")

    return df


# ============================================================
# Train/Val/Test Split
# ============================================================

def split_data(X, y, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into train, validation, and test sets.
    Maintains temporal ordering (no shuffling).
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Val:   {X_val.shape[0]} samples")
    print(f"    Test:  {X_test.shape[0]} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

def preprocess_demand(demand_features, config):
    """
    Complete preprocessing pipeline for demand forecasting data.
    """
    print("=" * 60)
    print(" STEP 2a: DEMAND DATA PREPROCESSING")
    print("=" * 60)

    prep_cfg = config['preprocessing']['demand']

    # Select numeric feature columns (exclude DateTime)
    feature_cols = [c for c in demand_features.columns
                    if c not in ['DateTime', 'Year'] and demand_features[c].dtype in ['float64', 'int64', 'int32', 'float32']]

    print(f"[→] Feature columns: {feature_cols}")

    # Handle any remaining NaN values
    demand_features[feature_cols] = demand_features[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Normalize
    print("[→] Normalizing features...")
    scaler = create_scaler(prep_cfg.get('scaler', 'minmax'))
    normalized_data, scaler = normalize_features(demand_features, feature_cols, scaler, fit=True)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/demand_scaler.pkl')
    print(f"    Scaler saved to models/demand_scaler.pkl")

    # Find target index
    target_col = prep_cfg['target_column']
    target_idx = feature_cols.index(target_col)
    print(f"    Target column '{target_col}' is at index {target_idx}")

    # Create sequences
    seq_length = prep_cfg['sequence_length']
    print(f"[→] Creating sliding windows (length={seq_length})...")

    # Analyze time gaps for data quality awareness (but don't discard data,
    # since this dataset has sparse readings — discarding would leave too few samples)
    if 'DateTime' in demand_features.columns:
        time_diffs = pd.to_datetime(demand_features['DateTime']).diff().dropna()
        if len(time_diffs) > 0:
            median_gap = time_diffs.median()
            max_gap = time_diffs.max()
            pct_large_gaps = (time_diffs > pd.Timedelta(hours=48)).mean() * 100
            print(f"    Time gap analysis: median={median_gap}, max={max_gap}")
            if pct_large_gaps > 10:
                print(f"    ⚠️  {pct_large_gaps:.0f}% of steps have gaps > 48h (sparse data)")
                print(f"    → Lag/rolling features help model handle irregular spacing")

    X, y = create_sequences(normalized_data, target_idx, seq_length)
    print(f"    Sequences shape: X={X.shape}, y={y.shape}")

    # Split
    print("[→] Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X, y, prep_cfg['train_split'], prep_cfg['val_split']
    )

    # Save processed data
    save_path = config['data']['demand_sequences']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_cols=feature_cols,
             target_idx=target_idx)
    print(f"[✓] Demand sequences saved to {save_path}\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, scaler


def preprocess_theft(theft_features, config):
    """
    Complete preprocessing pipeline for theft detection data.
    Includes SMOTE oversampling to balance theft/normal classes.
    """
    print("=" * 60)
    print(" STEP 2b: THEFT DATA PREPROCESSING")
    print("=" * 60)

    prep_cfg = config['preprocessing']['theft']

    # Generate theft labels
    theft_features = label_theft(
        theft_features,
        threshold_std=prep_cfg.get('theft_threshold_std', 2.0)
    )

    # Select strictly stationary numeric feature columns
    feature_cols = ['Difference', 'Meter_Constant', 'Consumption_Ratio', 
                    'Meter_Status_OK', 'Month_Num']

    # Ensure all feature cols exist
    feature_cols = [c for c in feature_cols if c in theft_features.columns]
    print(f"[→] Feature columns: {feature_cols}")

    # --- Impute NaN with meter-group medians (instead of 0) ---
    # Step 1: Fill with per-meter median/mode based on column type
    print("[→] Imputing NaN values with meter-group medians...")
    nan_before = theft_features[feature_cols].isna().sum().sum()

    # Discrete columns use mode (most frequent), continuous use median
    discrete_cols = ['Month_Num']

    for col in feature_cols:
        if col in discrete_cols:
            # Use mode for discrete integer columns (e.g. Month_Num = 1-12)
            meter_mode = theft_features.groupby('Energy_Meter_ID')[col].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            theft_features[col] = theft_features[col].fillna(meter_mode)
            global_mode = theft_features[col].mode().iloc[0] if not theft_features[col].mode().empty else 0
            theft_features[col] = theft_features[col].fillna(global_mode).astype(int)
        else:
            # Use median for continuous columns
            meter_med = theft_features.groupby('Energy_Meter_ID')[col].transform('median')
            theft_features[col] = theft_features[col].fillna(meter_med)
            theft_features[col] = theft_features[col].fillna(theft_features[col].median())

    # Last resort — fill any remaining NaN with 0
    theft_features[feature_cols] = theft_features[feature_cols].fillna(0)

    nan_after = theft_features[feature_cols].isna().sum().sum()
    print(f"    NaN values: {nan_before} → {nan_after} (imputed {nan_before - nan_after})")

    # Normalize features
    print("[→] Applying log1p transformation & Normalizing features...")
    if 'Difference' in feature_cols and 'Difference' in theft_features.columns:
        # Heavily skewed log-normal feature; log-transform separates small drops perfectly
        theft_features['Difference'] = np.log1p(np.clip(theft_features['Difference'], a_min=0, a_max=None))

    scaler = create_scaler(prep_cfg.get('scaler', 'minmax'))
    theft_features[feature_cols] = scaler.fit_transform(theft_features[feature_cols])

    # Save scaler
    joblib.dump(scaler, 'models/theft_scaler.pkl')
    print(f"    Scaler saved to models/theft_scaler.pkl")

    # Create sequences per meter
    seq_length = prep_cfg['sequence_length']
    print(f"[→] Creating meter-level sequences (length={seq_length})...")
    X, y, meter_ids = create_meter_sequences(
        theft_features, feature_cols, seq_length, label_col='Theft_Label'
    )
    print(f"    Sequences shape: X={X.shape}, y={y.shape}")
    print(f"    Theft ratio in sequences: {y.mean()*100:.1f}%")

    # Split
    print("[→] Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
        X, y, prep_cfg['train_split'], prep_cfg['val_split']
    )

    # ============================================================
    # CLASS BALANCING: SMOTE oversampling on training data ONLY
    # Val/Test remain untouched for honest evaluation
    # ============================================================
    print("[→] Balancing classes with SMOTE...")
    theft_count_train = int(y_train.sum())
    normal_count_train = len(y_train) - theft_count_train
    print(f"    Before SMOTE: Normal={normal_count_train}, Theft={theft_count_train} "
          f"({theft_count_train/len(y_train)*100:.1f}% theft)")

    if theft_count_train > 0 and theft_count_train < normal_count_train:
        try:
            from imblearn.over_sampling import SMOTE

            # Flatten 3D sequences to 2D for SMOTE: (n_samples, seq_len * n_features)
            n_samples, seq_len, n_features = X_train.shape
            X_train_flat = X_train.reshape(n_samples, -1)

            # Apply SMOTE — generate synthetic theft samples
            # ratio=0.3 means theft will be 30% of normal count (not 50/50)
            # This balances learning without making the model too aggressive
            smote = SMOTE(
                sampling_strategy=0.3,  # 0.3 = theft count = 30% of normal count
                k_neighbors=min(5, theft_count_train - 1),  # adapt to available samples
                random_state=42
            )
            X_train_flat_resampled, y_train_resampled = smote.fit_resample(
                X_train_flat, y_train
            )

            # Reshape back to 3D: (n_new_samples, seq_len, n_features)
            X_train = X_train_flat_resampled.reshape(-1, seq_len, n_features).astype(np.float32)
            y_train = y_train_resampled.astype(np.float32)

            new_theft = int(y_train.sum())
            new_normal = len(y_train) - new_theft
            print(f"    After SMOTE:  Normal={new_normal}, Theft={new_theft} "
                  f"({new_theft/len(y_train)*100:.1f}% theft)")
            print(f"    Generated {new_theft - theft_count_train} synthetic theft samples")

        except ImportError:
            print("    ⚠️ imbalanced-learn not installed. Skipping SMOTE.")
            print("    Install with: pip install imbalanced-learn")
        except Exception as e:
            print(f"    ⚠️ SMOTE failed: {e}. Using original data.")
    else:
        print("    Classes already balanced or no theft samples found.")

    print(f"    Final train set: X={X_train.shape}, y={y_train.shape}")

    # --- Save SMOTE-balanced dataset as CSV (original scale) ---
    balanced_path = 'data/processed/theft_balanced.csv'
    os.makedirs(os.path.dirname(balanced_path), exist_ok=True)
    n_samples, seq_len, n_feats = X_train.shape

    # Inverse transform to get original scale values
    rows = []
    for i in range(n_samples):
        # Inverse transform the normalized sequence back to original values
        orig_values = scaler.inverse_transform(X_train[i])  # (seq_len, n_features)
        for t in range(seq_len):
            row = {'Sequence_ID': i, 'TimeStep': t, 'Theft_Label': int(y_train[i])}
            for f_idx, f_name in enumerate(feature_cols):
                row[f_name] = round(orig_values[t, f_idx], 2)
            rows.append(row)
    balanced_df = pd.DataFrame(rows)
    balanced_df.to_csv(balanced_path, index=False)
    print(f"[✓] SMOTE-balanced dataset saved to {balanced_path} (original scale)")
    print(f"    Balanced samples: {n_samples} sequences "
          f"(Normal={int((y_train==0).sum())}, Theft={int((y_train==1).sum())})")

    # Save processed data
    save_path = config['data']['theft_sequences']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test,
             feature_cols=feature_cols)
    print(f"[✓] Theft sequences saved to {save_path}\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, scaler


# ============================================================
# Save processed CSV (for EDA / dashboard)
# ============================================================

def save_processed_csv(demand_features, theft_features, config):
    """Save processed features as CSV for later use."""
    os.makedirs('data/processed', exist_ok=True)

    demand_path = config['data']['demand_processed']
    theft_path = config['data']['theft_processed']

    demand_features.to_csv(demand_path, index=False)
    theft_features.to_csv(theft_path, index=False)

    print(f"[✓] Processed demand data saved to {demand_path}")
    print(f"[✓] Processed theft data saved to {theft_path}")


if __name__ == "__main__":
    from data_ingestion import run_ingestion
    from utils import load_config, create_directories

    config = load_config()
    create_directories(config)

    demand_features, theft_features = run_ingestion(config)
    demand_data, theft_data = preprocess_demand(demand_features, config), preprocess_theft(theft_features, config)
