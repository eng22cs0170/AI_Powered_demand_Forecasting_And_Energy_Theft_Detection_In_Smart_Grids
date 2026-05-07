# AI-Powered Demand Forecasting & Energy Theft Detection in Smart Grids

## Overview

An intelligent analytics system that enhances the efficiency and reliability of modern power distribution networks using deep learning. The project addresses two critical challenges:

1. **Demand Forecasting** — Predicting future electricity load (MW) based on temporal patterns
2. **Energy Theft Detection** — Identifying abnormal consumption behavior and generating probability-based alerts

## Architecture

The core is a **Hybrid GRU + TCN (Temporal Convolutional Network)** multitask model:

```
Input Sequence
    ├→ GRU Encoder (sequential patterns)
    ├→ TCN Encoder (temporal convolutions)
    └→ Concatenate → Latent Feature Representation
        ├→ Forecasting Head (Dense Regression) → Predicted MW
        └→ Theft Detection Head (Dense + Sigmoid) → Theft Probability
```

## Project Structure

```
major-project/
├── datasets/              # Raw Excel datasets
├── data/                  # Processed data
│   ├── processed/         # Cleaned CSVs
│   └── sequences/         # Sliding window sequences
├── src/                   # Source code
│   ├── data_ingestion.py  # Data loading & feature selection
│   ├── data_preprocessing.py  # Cleaning, normalization, windowing
│   ├── feature_engineering.py # Temporal & statistical features
│   ├── model.py           # Hybrid GRU + TCN architecture
│   ├── train.py           # Multitask training pipeline
│   ├── evaluate.py        # Evaluation metrics
│   ├── predict.py         # Inference pipeline
│   └── utils.py           # Helpers & plotting
├── models/                # Saved model weights & scalers
├── dashboard/             # Streamlit visualization dashboard
├── results/               # Plots, metrics, reports
├── config/config.yaml     # Configuration & hyperparameters
├── main.py                # Main pipeline entry point
└── requirements.txt       # Python dependencies
```

## Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the Complete Pipeline
```bash
python main.py
```

This runs: Data Ingestion → Preprocessing → Feature Engineering → Training → Evaluation → Visualization

### 2. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

## Technologies Used

- **Python 3.9+**
- **PyTorch** — Deep learning framework
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — Preprocessing & evaluation metrics
- **Streamlit** — Interactive dashboard
- **Plotly** — Interactive visualizations
- **Matplotlib / Seaborn** — Static plots

## Model Details

| Component | Details |
|-----------|---------|
| GRU Encoder | 2 layers, 128 hidden units, dropout 0.2 |
| TCN Encoder | 3 blocks [64, 128, 128], kernel size 3, exponential dilation |
| Latent Dim | 128 |
| Forecasting | MSE loss, regression output |
| Theft Detection | BCE loss, sigmoid probability output |
| Optimizer | Adam (lr=0.001, weight_decay=0.0001) |
| Early Stopping | Patience=15 epochs |

## Results

After running the pipeline, check:
- `results/plots/` — Training curves, predictions, confusion matrix
- `results/metrics/` — JSON files with evaluation metrics
- `models/` — Saved model weights and scalers
