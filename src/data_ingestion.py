"""
Data Ingestion Module
Handles loading raw Excel datasets and performing initial feature selection.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_demand_dataset(filepath):
    """
    Load the Demand Forecasting dataset from Excel.

    Expected columns: Date, Time, MW, Year, Month, DateTime
    Returns a cleaned DataFrame with proper types.
    """
    print(f"[→] Loading demand dataset from: {filepath}")
    df = pd.read_excel(filepath)

    print(f"    Raw shape: {df.shape}")
    print(f"    Columns: {df.columns.tolist()}")

    # --- Basic validation ---
    required_cols = ['Date', 'MW']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # --- Parse DateTime ---
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            errors='coerce'
        )
    else:
        df['DateTime'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where DateTime or MW is null
    before = len(df)
    df = df.dropna(subset=['DateTime', 'MW'])
    after = len(df)
    if before != after:
        print(f"    Dropped {before - after} rows with null DateTime/MW")

    # --- Filter invalid dates (before 2019 or after 2030) ---
    before = len(df)
    df = df[(df['DateTime'].dt.year >= 2019) & (df['DateTime'].dt.year <= 2030)]
    after = len(df)
    if before != after:
        print(f"    Filtered {before - after} rows with invalid dates (outside 2019-2030)")

    # Ensure MW is numeric
    df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    df = df.dropna(subset=['MW'])

    # --- Remove extreme MW outliers (beyond 3 std deviations) ---
    mw_mean = df['MW'].mean()
    mw_std = df['MW'].std()
    before = len(df)
    df = df[(df['MW'] >= 0) & (df['MW'] <= mw_mean + 3 * mw_std)]
    after = len(df)
    if before != after:
        print(f"    Removed {before - after} MW outliers (beyond 3σ: >{mw_mean + 3*mw_std:.1f})")

    # Sort by DateTime
    df = df.sort_values('DateTime').reset_index(drop=True)

    # --- Remove duplicate timestamps (keep last reading) ---
    before = len(df)
    df = df.drop_duplicates(subset='DateTime', keep='last').reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"    Removed {before - after} duplicate timestamps")

    # --- Data quality summary ---
    print(f"    Final shape: {df.shape}")
    print(f"    Date range: {df['DateTime'].min()} → {df['DateTime'].max()}")
    print(f"    MW range: {df['MW'].min():.2f} → {df['MW'].max():.2f}")
    print(f"    Unique dates: {df['DateTime'].nunique()}")
    time_diffs = df['DateTime'].diff().dropna()
    if len(time_diffs) > 0:
        most_common = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 'N/A'
        print(f"    Most common time step: {most_common}")
    print(f"[✓] Demand dataset loaded successfully.\n")

    return df


def load_theft_dataset(filepath):
    """
    Load the Theft Detection dataset from Excel.

    Expected columns: Energy_Meter_ID, Year, Month, Present_Reading,
                      Previous_Reading, Difference, Meter_Constant,
                      Net_Energy_MU, Meter_Status, Type of Meter, Particulars
    Returns a cleaned DataFrame.
    """
    print(f"[→] Loading theft dataset from: {filepath}")
    if str(filepath).endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    print(f"    Raw shape: {df.shape}")
    print(f"    Columns: {df.columns.tolist()}")

    # --- Remove garbage/header rows at the bottom ---
    # These rows have non-numeric values in numeric columns
    # Filter: keep only rows where Year is a valid year (2019-2030)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df = df[(df['Year'] >= 2019) & (df['Year'] <= 2030)]
    df['Year'] = df['Year'].astype(int)

    # --- Clean numeric columns ---
    numeric_cols = ['Present_Reading', 'Previous_Reading', 'Difference',
                    'Meter_Constant', 'Net_Energy_MU']

    for col in numeric_cols:
        if col in df.columns:
            # Extract numeric part from strings like "342281.8 ON(28-1-20)"
            df[col] = df[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Clean Meter_Status ---
    if 'Meter_Status' in df.columns:
        df['Meter_Status'] = df['Meter_Status'].fillna('Unknown')
        df['Meter_Status'] = df['Meter_Status'].str.strip().str.upper()

    # --- Clean Month ---
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    if 'Month' in df.columns:
        # Leave NaN so downstream imputation fills with meter-group mode
        df['Month_Num'] = df['Month'].map(month_map)

    # --- Clean Energy_Meter_ID ---
    df['Energy_Meter_ID'] = df['Energy_Meter_ID'].astype(str).str.strip()

    # Drop rows where essential numeric columns are all null
    essential = ['Present_Reading', 'Previous_Reading', 'Difference']
    existing_essential = [c for c in essential if c in df.columns]
    if existing_essential:
        df = df.dropna(subset=existing_essential, how='all')

    df = df.reset_index(drop=True)

    print(f"    Final shape: {df.shape}")
    print(f"    Unique meters: {df['Energy_Meter_ID'].nunique()}")
    print(f"    Year range: {df['Year'].min()} → {df['Year'].max()}")
    print(f"[✓] Theft dataset loaded successfully.\n")

    return df


def select_features_demand(df):
    """
    Select and create relevant features for demand forecasting.
    Returns DataFrame with selected features.
    """
    print("[→] Selecting features for demand forecasting...")

    features = pd.DataFrame()
    features['DateTime'] = df['DateTime']
    features['MW'] = df['MW']

    # Extract temporal features
    features['Hour'] = df['DateTime'].dt.hour
    features['DayOfWeek'] = df['DateTime'].dt.dayofweek
    features['DayOfMonth'] = df['DateTime'].dt.day
    features['Month'] = df['DateTime'].dt.month
    features['Year'] = df['DateTime'].dt.year
    features['Quarter'] = df['DateTime'].dt.quarter
    features['WeekOfYear'] = df['DateTime'].dt.isocalendar().week.astype(int)
    features['IsWeekend'] = (df['DateTime'].dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for temporal features
    features['Hour_sin'] = np.sin(2 * np.pi * features['Hour'] / 24)
    features['Hour_cos'] = np.cos(2 * np.pi * features['Hour'] / 24)
    features['DayOfWeek_sin'] = np.sin(2 * np.pi * features['DayOfWeek'] / 7)
    features['DayOfWeek_cos'] = np.cos(2 * np.pi * features['DayOfWeek'] / 7)
    features['Month_sin'] = np.sin(2 * np.pi * features['Month'] / 12)
    features['Month_cos'] = np.cos(2 * np.pi * features['Month'] / 12)

    print(f"    Features selected: {features.columns.tolist()}")
    print(f"    Shape: {features.shape}")
    print(f"[✓] Feature selection complete.\n")

    return features


def select_features_theft(df):
    """
    Select and create relevant features for theft detection.
    Returns DataFrame with selected features.
    """
    print("[→] Selecting features for theft detection...")

    features = pd.DataFrame()
    features['Energy_Meter_ID'] = df['Energy_Meter_ID']
    features['Year'] = df['Year']
    features['Month_Num'] = df['Month_Num']
    
    # Core consumption features
    features['Present_Reading'] = df['Present_Reading']
    features['Previous_Reading'] = df['Previous_Reading']
    features['Difference'] = df['Difference']
    features['Meter_Constant'] = df['Meter_Constant']
    if 'Net_Energy_MU' in df.columns:
        features['Net_Energy_MU'] = df['Net_Energy_MU']

    # Derived features
    features['Consumption_Ratio'] = np.where(
        df['Previous_Reading'] > 0,
        features['Difference'] / df['Previous_Reading'],
        0
    )

    # Meter status encoding (OK=0, others may indicate issues)
    features['Meter_Status_OK'] = (df['Meter_Status'] == 'OK').astype(int)

    if 'Is_Theft' in df.columns:
        features['Is_Theft'] = df['Is_Theft']

    print(f"    Features selected: {features.columns.tolist()}")
    print(f"    Shape: {features.shape}")
    print(f"[✓] Feature selection complete.\n")

    return features


# ============================================================
# Main ingestion pipeline
# ============================================================

def run_ingestion(config):
    """Run the complete data ingestion pipeline."""
    print("=" * 60)
    print(" STEP 1: DATA INGESTION")
    print("=" * 60)

    # Load raw datasets
    demand_df = load_demand_dataset(config['data']['demand_raw'])
    theft_df = load_theft_dataset(config['data']['theft_raw'])

    # Select features
    demand_features = select_features_demand(demand_df)
    theft_features = select_features_theft(theft_df)

    return demand_features, theft_features


if __name__ == "__main__":
    from utils import load_config, create_directories
    config = load_config()
    create_directories(config)
    demand_features, theft_features = run_ingestion(config)
    print("Demand features sample:")
    print(demand_features.head())
    print("\nTheft features sample:")
    print(theft_features.head())
