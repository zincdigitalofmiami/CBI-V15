#!/usr/bin/env python3
"""
LSTM baseline training for ZL - Minimal Feature Set
Trains one model per horizon (1w, 1m, 3m, 6m, 12m)
Uses ~15 features from zl_training_minimal_{h} tables
PyTorch with MPS backend for Mac M4 GPU acceleration
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use local directory structure
DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local/dl_round1")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

# Sequence parameters
LOOKBACK_WINDOW = 30  # days (tuned: 21, 30, 60)
BATCH_SIZE = 32  # Mac M4 memory constraint

# MPS setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.backends.mps.allow_tf32 = True
    logging.info("✅ Using MPS (Metal) backend")
else:
    device = torch.device("cpu")
    logging.warning("⚠️  MPS not available, using CPU")

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """Simple LSTM model for price prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.relu(last_output))
        return output.squeeze()

def create_sequences(data, lookback, target_col):
    """Create sequences from time series data"""
    sequences = []
    targets = []
    
    for i in range(lookback, len(data)):
        seq = data.iloc[i-lookback:i].values
        target = data.iloc[i][target_col]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def load_split(horizon: str, split_name: str) -> pd.DataFrame:
    """Load a train/val/test split from Parquet for specific horizon"""
    file_path = DATA_DIR / f"zl_training_minimal_{horizon}_{split_name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    return pd.read_parquet(file_path)

def train_lstm_for_horizon(horizon: str):
    """Train LSTM model for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training LSTM for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    # Load splits
    train = load_split(horizon, "train")
    val = load_split(horizon, "val")
    test = load_split(horizon, "test")
    
    # Target column name
    target_col = f"target_{horizon}_price"
    
    # Filter out rows with missing target
    train = train[train[target_col].notna()].copy()
    val = val[val[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()
    
    logging.info(f"Train: {len(train):,} rows")
    logging.info(f"Val: {len(val):,} rows")
    logging.info(f"Test: {len(test):,} rows")
    
    # Feature columns: all except target and price_current
    feature_cols = sorted(
        col for col in train.columns
        if col not in [target_col, "price_current"]
    )
    
    logging.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    
    # Prepare data (fillna before scaling)
    train_features = train[feature_cols].fillna(0)
    val_features = val[feature_cols].fillna(0)
    test_features = test[feature_cols].fillna(0)
    
    # Standardize features (fit on train only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    
    # Add target to scaled data for sequence creation
    train_data = pd.DataFrame(train_scaled, columns=feature_cols)
    train_data[target_col] = train[target_col].values
    val_data = pd.DataFrame(val_scaled, columns=feature_cols)
    val_data[target_col] = val[target_col].values
    test_data = pd.DataFrame(test_scaled, columns=feature_cols)
    test_data[target_col] = test[target_col].values
    
    # Create sequences
    logging.info(f"Creating sequences with lookback={LOOKBACK_WINDOW}...")
    X_train, y_train = create_sequences(train_data, LOOKBACK_WINDOW, target_col)
    X_val, y_val = create_sequences(val_data, LOOKBACK_WINDOW, target_col)
    X_test, y_test = create_sequences(test_data, LOOKBACK_WINDOW, target_col)
    
    logging.info(f"Train sequences: {X_train.shape}")
    logging.info(f"Val sequences: {X_val.shape}")
    logging.info(f"Test sequences: {X_test.shape}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = len(feature_cols)
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1, dropout=0.1).to(device)
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logging.info("Starting training...")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODELS_DIR / f"zl_{horizon}_lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"zl_{horizon}_lstm_best.pt"))
    
    # Evaluate
    model.eval()
    predictions = {}
    targets_dict = {}
    
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        preds = []
        targets = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x).cpu().numpy()
                preds.extend(outputs)
                targets.extend(batch_y.numpy())
        
        predictions[split_name] = np.array(preds)
        targets_dict[split_name] = np.array(targets)
    
    # Calculate metrics
    metrics = {}
    for split_name in ["train", "val", "test"]:
        y_true = targets_dict[split_name]
        y_pred = predictions[split_name]
        metrics[split_name] = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Results for {horizon}:")
    logging.info(f"{'='*60}")
    logging.info(f"Train - MAE: {metrics['train']['mae']:.4f}, RMSE: {metrics['train']['rmse']:.4f}, R²: {metrics['train']['r2']:.4f}")
    logging.info(f"Val   - MAE: {metrics['val']['mae']:.4f}, RMSE: {metrics['val']['rmse']:.4f}, R²: {metrics['val']['r2']:.4f}")
    logging.info(f"Test  - MAE: {metrics['test']['mae']:.4f}, RMSE: {metrics['test']['rmse']:.4f}, R²: {metrics['test']['r2']:.4f}")
    
    # Save model and metadata
    model_path = MODELS_DIR / f"zl_{horizon}_lstm.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1
        },
        'scaler': scaler,
        'feature_cols': feature_cols,
        'horizon': horizon,
        'target_col': target_col,
        'lookback_window': LOOKBACK_WINDOW,
        'metrics': metrics
    }, model_path)
    logging.info(f"✅ Model saved to {model_path}")
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'LSTM Training Curves - {horizon}')
    plt.legend()
    plt.grid(True)
    curve_path = MODELS_DIR / f"zl_{horizon}_training_curves.png"
    plt.savefig(curve_path)
    plt.close()
    logging.info(f"✅ Training curves saved to {curve_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'target': targets_dict['test'],
        'prediction': predictions['test'],
        'horizon': horizon
    })
    predictions_path = DATA_DIR / f"predictions_lstm_{horizon}.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logging.info(f"✅ Predictions saved to {predictions_path}")
    
    return model, metrics

if __name__ == "__main__":
    logging.info("🚀 Starting LSTM training for ZL (minimal feature set)...")
    
    models = {}
    results = {}
    for horizon in HORIZONS:
        try:
            model, metrics = train_lstm_for_horizon(horizon)
            models[horizon] = model
            results[horizon] = metrics
        except Exception as e:
            logging.error(f"❌ Failed to train {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info(f"\n{'='*60}")
    logging.info("✅ All models trained successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")
    logging.info(f"\nSummary:")
    for horizon, metrics in results.items():
        logging.info(f"  {horizon:3s}: Test MAE={metrics['test']['mae']:.4f}, R²={metrics['test']['r2']:.4f}")
    logging.info(f"{'='*60}")

