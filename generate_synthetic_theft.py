import pandas as pd
import numpy as np
import os
import random

def generate_balanced_theft_data(input_path, output_path, total_meters=1000, months=24):
    """
    Generate a balanced (50/50) synthetic dataset for theft detection
    based on the distributions found in the real dataset.
    """
    print("Loading original dataset to learn distributions...")
    try:
        df_real = pd.read_excel(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # Extract realistic baseline consumption
    if 'Difference' in df_real.columns:
        valid_diffs = pd.to_numeric(df_real['Difference'], errors='coerce').dropna()
        valid_diffs = valid_diffs[valid_diffs > 0]
        
        # We will use log-normal distribution to mimic realistic energy consumption
        # Fit a log-normal distribution to the non-zero differences
        log_data = np.log(valid_diffs)
        mu = np.mean(log_data)
        sigma = np.std(log_data)
    else:
        mu = 5.0
        sigma = 1.0

    print(f"Learned consumption distribution: Log-Normal(mu={mu:.2f}, sigma={sigma:.2f})")

    # Half normal, half theft
    normal_meters_count = total_meters // 2
    theft_meters_count = total_meters - normal_meters_count
    
    records = []
    meter_idx = 1
    
    # Generate Normal Profiles
    print(f"Generating {normal_meters_count} normal meter profiles...")
    for _ in range(normal_meters_count):
        meter_id = f"SYN_N_{meter_idx:04d}"
        meter_idx += 1
        
        base_monthly = max(10, np.random.lognormal(mean=mu, sigma=sigma))
        seasonal_mult = 1.0 + 0.3 * np.sin(np.linspace(0, 2*np.pi * (months/12), months))
        prev_reading = np.random.uniform(1000, 50000)
        
        for m in range(months):
            # Normal variance around base consumption + seasonality
            actual_consumption = max(10, np.random.normal(base_monthly * seasonal_mult[m], base_monthly * 0.15))
            
            present_reading = prev_reading + actual_consumption
            
            records.append({
                'Energy_Meter_ID': meter_id,
                'Month_Index': m + 1,
                'Previous_Reading': round(prev_reading, 2),
                'Present_Reading': round(present_reading, 2),
                'Difference': round(actual_consumption, 2),
                'Meter_Status': 'OK',
                'Is_Theft': 0,
                'Theft_Type': 'None'
            })
            prev_reading = present_reading

    # Generate Theft Profiles
    print(f"Generating {theft_meters_count} theft meter profiles...")
    theft_types = ['Bypass', 'Tampering_Percentage', 'Tampering_Intermittent']
    
    for _ in range(theft_meters_count):
        meter_id = f"SYN_T_{meter_idx:04d}"
        meter_idx += 1
        theft_style = random.choice(theft_types)
        
        base_monthly = max(10, np.random.lognormal(mean=mu, sigma=sigma))
        seasonal_mult = 1.0 + 0.3 * np.sin(np.linspace(0, 2*np.pi * (months/12), months))
        prev_reading = np.random.uniform(1000, 50000)
        
        theft_start_month = random.randint(3, months - 5)
        if theft_style == 'Tampering_Percentage':
            reduction_factor = random.uniform(0.1, 0.7) 
        
        for m in range(months):
            actual_consumption = max(10, np.random.normal(base_monthly * seasonal_mult[m], base_monthly * 0.15))
            recorded_consumption = actual_consumption
            
            is_theft_active = 0
            current_theft_type = 'None'
            
            if m >= theft_start_month:
                is_theft_active = 1
                current_theft_type = theft_style
                if theft_style == 'Bypass':
                    # Almost zero recorded consumption
                    recorded_consumption = random.uniform(2, 10)
                elif theft_style == 'Tampering_Percentage':
                    recorded_consumption = actual_consumption * reduction_factor
                elif theft_style == 'Tampering_Intermittent':
                    if random.random() > 0.5:
                        recorded_consumption = random.uniform(2, 10)
                    else:
                        is_theft_active = 0
                        current_theft_type = 'None'
            
            present_reading = prev_reading + recorded_consumption
            
            records.append({
                'Energy_Meter_ID': meter_id,
                'Month_Index': m + 1,
                'Previous_Reading': round(prev_reading, 2),
                'Present_Reading': round(present_reading, 2),
                'Difference': round(recorded_consumption, 2),
                'Meter_Status': 'OK' if random.random() > 0.05 else 'TAMPERED',
                'Is_Theft': is_theft_active,
                'Theft_Type': current_theft_type
            })
            prev_reading = present_reading

    df_synth = pd.DataFrame(records)
    
    # Shuffle the meters
    groups = [df for _, df in df_synth.groupby('Energy_Meter_ID')]
    random.shuffle(groups)
    df_synth = pd.concat(groups).reset_index(drop=True)
    
    df_synth.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated {len(df_synth)} records across {total_meters} meters.")
    print(f"Dataset completely balanced: 50% normal meters, 50% theft meters.")
    print(f"File saved to: {output_path}")
    
    print("\nTheft Label Records Distribution:")
    print(df_synth['Is_Theft'].value_counts(normalize=True))

if __name__ == "__main__":
    INPUT = 'datasets/Theft_Detection_Dataset_2020_2025.xlsx'
    OUTPUT = 'datasets/Synthesized_Balanced_Theft_Data.csv'
    generate_balanced_theft_data(INPUT, OUTPUT, total_meters=1000, months=60)
