"""
Training Module
Handles multitask training with both forecasting and theft detection objectives.
Supports early stopping, learning rate scheduling, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=15, min_delta=1e-5, save_path='models/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n[!] Early stopping triggered after {self.counter} epochs without improvement.")
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)


class MultitaskTrainer:
    """
    Trainer for the hybrid GRU+TCN multitask model.

    Trains both forecasting (MSE) and theft detection (BCE) simultaneously
    with configurable loss weights.
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        train_cfg = config['training']

        # Optimizers
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

        # Loss functions
        self.forecast_criterion = nn.MSELoss()
        self.theft_criterion = nn.BCELoss()

        # Loss weights
        self.forecast_weight = train_cfg['loss_weights']['forecasting']
        self.theft_weight = train_cfg['loss_weights']['theft_detection']
        self._theft_pos_weight = 1.0  # will be updated if class imbalance detected

        # Training params
        self.epochs = train_cfg['epochs']
        self.batch_size = train_cfg['batch_size']

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_cfg['patience'],
            save_path=train_cfg['save_path']
        )

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_forecast_loss': [], 'val_forecast_loss': [],
            'train_theft_loss': [], 'val_theft_loss': [],
            'lr': []
        }

    def create_dataloader(self, X, y, shuffle=True):
        """Create a PyTorch DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_epoch(self, train_loader, task='forecast'):
        """Train for one epoch on a specific task."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()

            output = self.model(X_batch, task=task)

            if task == 'forecast':
                loss = self.forecast_criterion(output['forecast'], y_batch)
            else:
                loss = self.theft_criterion(output['theft'], y_batch)

            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def validate_epoch(self, val_loader, task='forecast'):
        """Validate for one epoch on a specific task."""
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = self.model(X_batch, task=task)

                if task == 'forecast':
                    loss = self.forecast_criterion(output['forecast'], y_batch)
                else:
                    loss = self.theft_criterion(output['theft'], y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, demand_data=None, theft_data=None):
        """
        Full training loop with alternating task training.

        Args:
            demand_data: tuple of ((X_train, y_train), (X_val, y_val))
            theft_data: tuple of ((X_train, y_train), (X_val, y_val))
        """
        print("\n" + "=" * 60)
        print(" TRAINING: Hybrid GRU + TCN Multitask Model")
        print("=" * 60)

        # Create data loaders
        demand_train_loader = demand_val_loader = None
        theft_train_loader = theft_val_loader = None

        if demand_data is not None:
            (X_train_d, y_train_d), (X_val_d, y_val_d) = demand_data
            demand_train_loader = self.create_dataloader(X_train_d, y_train_d, shuffle=True)
            demand_val_loader = self.create_dataloader(X_val_d, y_val_d, shuffle=False)
            print(f"[→] Demand: {X_train_d.shape[0]} train, {X_val_d.shape[0]} val samples")

        if theft_data is not None:
            (X_train_t, y_train_t), (X_val_t, y_val_t) = theft_data
            theft_train_loader = self.create_dataloader(X_train_t, y_train_t, shuffle=True)
            theft_val_loader = self.create_dataloader(X_val_t, y_val_t, shuffle=False)
            print(f"[→] Theft:  {X_train_t.shape[0]} train, {X_val_t.shape[0]} val samples")

            # Note: SMOTE already balances the training data
            # No additional class weighting needed
            n_theft = y_train_t.sum()
            n_normal = len(y_train_t) - n_theft
            print(f"[→] Train distribution: Normal={int(n_normal)}, Theft={int(n_theft)} "
                  f"({n_theft/len(y_train_t)*100:.1f}% theft)")

        print(f"[→] Device: {self.device}")
        print(f"[→] Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"[→] LR: {self.config['training']['learning_rate']}")
        print(f"[→] Loss weights: forecast={self.forecast_weight}, theft={self.theft_weight}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            train_forecast_loss = 0.0
            train_theft_loss = 0.0
            val_forecast_loss = 0.0
            val_theft_loss = 0.0

            # --- Train Multitask (Interleaved) ---
            self.model.train()
            iter_d = iter(demand_train_loader) if demand_train_loader else None
            iter_t = iter(theft_train_loader) if theft_train_loader else None
            
            n_batches_d, n_batches_t = 0, 0
            while iter_d is not None or iter_t is not None:
                # Train one forecast batch
                if iter_d is not None:
                    try:
                        X_d, y_d = next(iter_d)
                        self.optimizer.zero_grad()
                        out_d = self.model(X_d, task='forecast')
                        loss_d = self.forecast_criterion(out_d['forecast'], y_d)
                        (loss_d * self.forecast_weight).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        train_forecast_loss += loss_d.item()
                        n_batches_d += 1
                    except StopIteration:
                        iter_d = None
                
                # Train one theft batch
                if iter_t is not None:
                    try:
                        X_t, y_t = next(iter_t)
                        self.optimizer.zero_grad()
                        out_t = self.model(X_t, task='theft')
                        loss_t = self.theft_criterion(out_t['theft'], y_t)
                        (loss_t * self.theft_weight).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        train_theft_loss += loss_t.item()
                        n_batches_t += 1
                    except StopIteration:
                        iter_t = None
                        
            train_forecast_loss = train_forecast_loss / max(n_batches_d, 1)
            train_theft_loss = train_theft_loss / max(n_batches_t, 1)

            # Validation remains sequential (it doesn't affect weights)
            if demand_val_loader:
                val_forecast_loss = self.validate_epoch(demand_val_loader, 'forecast')
            if theft_val_loader:
                val_theft_loss = self.validate_epoch(theft_val_loader, 'theft')

            # Combined loss
            train_total = (self.forecast_weight * train_forecast_loss +
                          self.theft_weight * train_theft_loss)
            val_total = (self.forecast_weight * val_forecast_loss +
                        self.theft_weight * val_theft_loss)

            # Record history
            self.history['train_loss'].append(train_total)
            self.history['val_loss'].append(val_total)
            self.history['train_forecast_loss'].append(train_forecast_loss)
            self.history['val_forecast_loss'].append(val_forecast_loss)
            self.history['train_theft_loss'].append(train_theft_loss)
            self.history['val_theft_loss'].append(val_theft_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Learning rate scheduling
            self.scheduler.step(val_total)

            # Early stopping
            self.early_stopping(val_total, self.model)

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch [{epoch+1:3d}/{self.epochs}] | "
                      f"Train: {train_total:.6f} (F:{train_forecast_loss:.4f} T:{train_theft_loss:.4f}) | "
                      f"Val: {val_total:.6f} (F:{val_forecast_loss:.4f} T:{val_theft_loss:.4f}) | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {elapsed:.1f}s")

            if self.early_stopping.early_stop:
                break

        total_time = time.time() - start_time
        print(f"\n[✓] Training complete! Total time: {total_time:.1f}s")
        print(f"    Best validation loss: {self.early_stopping.best_loss:.6f}")
        print(f"    Model saved to: {self.early_stopping.save_path}")

        # Load best model
        self.model.load_state_dict(torch.load(self.early_stopping.save_path,
                                               map_location=self.device, weights_only=True))

        return self.history


def train_model(model, config, demand_data=None, theft_data=None, device=None):
    """
    Convenience function to train the model.

    Args:
        model: SmartGridModel instance
        config: configuration dict
        demand_data: ((X_train, y_train), (X_val, y_val)) for demand
        theft_data: ((X_train, y_train), (X_val, y_val)) for theft
        device: torch device

    Returns:
        history dict with training curves
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = MultitaskTrainer(model, config, device)
    history = trainer.train(demand_data, theft_data)
    return history
