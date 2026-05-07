"""
Feature Engineering Module
Creates advanced temporal and statistical features for improved model performance.
"""

import pandas as pd
import numpy as np


def add_lag_features(df, target_col='MW', lags=[1, 2, 3, 6, 12, 24]):
    """
    Add lag features for time-series forecasting.

    Args:
        df: DataFrame sorted by time
        target_col: column to create lags for
        lags: list of lag periods

    Returns:
        DataFrame with lag features added
    """
    print(f"[→] Adding lag features for '{target_col}': {lags}")
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    return df


def add_rolling_features(df, target_col='MW', windows=[3, 6, 12, 24]):
    """
    Add rolling window statistics (mean, std, min, max).
    """
    print(f"[→] Adding rolling features for '{target_col}': windows={windows}")
    for w in windows:
        df[f'{target_col}_rolling_mean_{w}'] = df[target_col].rolling(window=w).mean()
        df[f'{target_col}_rolling_std_{w}'] = df[target_col].rolling(window=w).std()
        df[f'{target_col}_rolling_min_{w}'] = df[target_col].rolling(window=w).min()
        df[f'{target_col}_rolling_max_{w}'] = df[target_col].rolling(window=w).max()

    return df


def add_diff_features(df, target_col='MW', periods=[1, 2, 3]):
    """
    Add difference features (change from previous values).
    """
    print(f"[→] Adding difference features for '{target_col}': periods={periods}")
    for p in periods:
        df[f'{target_col}_diff_{p}'] = df[target_col].diff(periods=p)

    return df


def add_theft_statistical_features(df):
    """
    Add per-meter statistical features for theft detection.
    These features capture deviation from a meter's normal behavior.
    """
    print("[→] Adding theft statistical features per meter...")

    # Per-meter aggregation features
    meter_stats = df.groupby('Energy_Meter_ID').agg(
        meter_mean_diff=('Difference', 'mean'),
        meter_std_diff=('Difference', 'std'),
        meter_mean_net=('Net_Energy_MU', 'mean'),
        meter_std_net=('Net_Energy_MU', 'std'),
        meter_record_count=('Energy_Meter_ID', 'count')
    ).reset_index()

    df = df.merge(meter_stats, on='Energy_Meter_ID', how='left')

    # Z-score of current reading relative to meter's history
    df['Diff_Zscore'] = np.where(
        df['meter_std_diff'] > 0,
        (df['Difference'] - df['meter_mean_diff']) / df['meter_std_diff'],
        0
    )

    df['Net_Energy_Zscore'] = np.where(
        df['meter_std_net'] > 0,
        (df['Net_Energy_MU'] - df['meter_mean_net']) / df['meter_std_net'],
        0
    )

    # Deviation from expected (ratio of current to mean)
    df['Diff_Deviation_Ratio'] = np.where(
        df['meter_mean_diff'] != 0,
        df['Difference'] / df['meter_mean_diff'],
        1.0
    )

    print(f"    Added features: Diff_Zscore, Net_Energy_Zscore, Diff_Deviation_Ratio, etc.")
    return df


def engineer_demand_features(demand_features, config):
    """
    Complete feature engineering for demand forecasting.
    Adds lag, rolling, and difference features.

    Note: Feature count is kept moderate to avoid overfitting
    with small datasets (train/feature ratio should be > 3x).
    """
    print("\n" + "=" * 60)
    print(" FEATURE ENGINEERING: DEMAND")
    print("=" * 60)

    df = demand_features.copy()

    # Add lag features (keep only most important lags)
    df = add_lag_features(df, 'MW', lags=[1, 2, 3, 6, 12])

    # Add rolling statistics (reduced from [3,6,12] to [3,6] to reduce features)
    df = add_rolling_features(df, 'MW', windows=[3, 6])

    # Add difference features
    df = add_diff_features(df, 'MW', periods=[1, 2])

    # Fill NaN from lag/rolling features
    df = df.fillna(method='bfill').fillna(0)

    # --- Overfitting risk check ---
    n_rows = len(df)
    n_features = len([c for c in df.columns if c not in ['DateTime', 'Year']])
    seq_len = config['preprocessing']['demand']['sequence_length']
    input_dim = seq_len * n_features
    train_samples = int((n_rows - seq_len) * config['preprocessing']['demand']['train_split'])
    ratio = train_samples / n_features  # simplified ratio
    print(f"    Dataset size: {n_rows} rows, {n_features} features")
    print(f"    Estimated train samples: {train_samples}")
    print(f"    Samples-to-features ratio: {ratio:.1f}x", end="")
    if ratio < 30:
        print(" ⚠️ (low — regularization critical)")
    else:
        print(" ✅")

    print(f"[✓] Demand feature engineering complete. Shape: {df.shape}\n")
    return df


def engineer_theft_features(theft_features, config):
    """
    Complete feature engineering for theft detection.
    Adds per-meter statistical features.
    """
    print("\n" + "=" * 60)
    print(" FEATURE ENGINEERING: THEFT")
    print("=" * 60)

    df = theft_features.copy()

    # Add per-meter statistical features
    df = add_theft_statistical_features(df)

    # --- Impute NaN with meter-group medians (instead of 0) ---
    # This ensures faulty/tampered meter rows get realistic values
    # based on that meter's historical behavior, not arbitrary zeros.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [c for c in numeric_cols if c != 'Energy_Meter_ID']

    # Discrete columns use mode (most frequent), continuous use median
    discrete_cols = ['Month_Num']

    print("[→] Imputing NaN values with meter-group medians...")
    nan_before = df[cols_to_impute].isna().sum().sum()

    for col in cols_to_impute:
        if col in discrete_cols:
            # Use mode for discrete integer columns (e.g. Month_Num = 1-12)
            meter_mode = df.groupby('Energy_Meter_ID')[col].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            df[col] = df[col].fillna(meter_mode)
            global_mode = df[col].mode().iloc[0] if not df[col].mode().empty else 0
            df[col] = df[col].fillna(global_mode).astype(int)
        else:
            # Use median for continuous columns
            meter_med = df.groupby('Energy_Meter_ID')[col].transform('median')
            df[col] = df[col].fillna(meter_med)
            df[col] = df[col].fillna(df[col].median())

    # Last resort — fill any remaining NaN with 0
    df[cols_to_impute] = df[cols_to_impute].fillna(0)

    nan_after = df[cols_to_impute].isna().sum().sum()
    print(f"    NaN values: {nan_before} → {nan_after} (imputed {nan_before - nan_after})")

    print(f"[✓] Theft feature engineering complete. Shape: {df.shape}\n")
    return df
