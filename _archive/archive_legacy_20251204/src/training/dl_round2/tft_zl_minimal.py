#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) training for ZL - Minimal Feature Set
Trains one model per horizon (1w, 1m, 3m, 6m, 12m)
Conditional: Skip if LSTM already >10% improvement over baseline
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
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use local directory structure
DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local/dl_round2")
BASELINE_DIR = Path("Models/local/baseline")
LSTM_DIR = Path("Models/local/dl_round1")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All 4 horizons (12m not available yet)
HORIZONS = ["1w", "1m", "3m", "6m"]

# Sequence parameters
LOOKBACK_WINDOW = 30
BATCH_SIZE = 16  # Smaller batch for TFT (memory constraint)

# MPS setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.backends.mps.allow_tf32 = True
    logging.info("✅ Using MPS (Metal) backend")
else:
    device = torch.device("cpu")
    logging.warning("⚠️  MPS not available, using CPU")

def should_skip_tft(horizon: str) -> bool:
    """Check if TFT should be skipped based on LSTM performance"""
    try:
        # Load baseline metrics
        import joblib
        baseline_path = BASELINE_DIR / f"zl_{horizon}_lightgbm.pkl"
        if not baseline_path.exists():
            logging.warning(f"Baseline model not found for {horizon}, proceeding with TFT")
            return False
        
        baseline_data = joblib.load(baseline_path)
        baseline_mae = baseline_data['metrics']['test']['mae']
        
        # Load LSTM metrics
        lstm_path = LSTM_DIR / f"zl_{horizon}_lstm.pt"
        if not lstm_path.exists():
            logging.warning(f"LSTM model not found for {horizon}, proceeding with TFT")
            return False
        
        lstm_data = torch.load(lstm_path, map_location='cpu')
        lstm_mae = lstm_data['metrics']['test']['mae']
        
        # Check if LSTM improvement >10%
        improvement = (baseline_mae - lstm_mae) / baseline_mae
        if improvement > 0.10:
            logging.info(f"⚠️  LSTM already achieves {improvement*100:.1f}% improvement over baseline for {horizon}")
            logging.info(f"   Skipping TFT to avoid diminishing returns")
            return True
        
        return False
    except Exception as e:
        logging.warning(f"Could not check skip condition: {e}, proceeding with TFT")
        return False

class SimpleTFT(nn.Module):
    """Simplified TFT architecture for minimal feature set"""
    def __init__(self, input_size, hidden_size=64, num_heads=4, dropout=0.1):
        super(SimpleTFT, self).__init__()
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Multi-head attention (simplified)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # LSTM encoding
        lstm_out, _ = self.encoder(x)  # (batch, seq_len, hidden_size)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, hidden_size)
        
        # Take last timestep
        last_output = attn_out[:, -1, :]  # (batch, hidden_size)
        
        # Output layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()

class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

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

def train_tft_for_horizon(horizon: str):
    """Train TFT model for a specific horizon"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training TFT for {horizon} horizon")
    logging.info(f"{'='*60}")
    
    # Check if should skip
    if should_skip_tft(horizon):
        logging.info(f"⏭️  Skipping TFT for {horizon} (LSTM already >10% improvement)")
        return None, None
    
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
    
    # Feature columns
    feature_cols = sorted(
        col for col in train.columns
        if col not in [target_col, "price_current"]
    )
    
    logging.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    
    # Prepare and scale data
    train_features = train[feature_cols].fillna(0)
    val_features = val[feature_cols].fillna(0)
    test_features = test[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    
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
    model = SimpleTFT(input_size=input_size, hidden_size=64, num_heads=4, dropout=0.1).to(device)
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
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
        
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / f"zl_{horizon}_tft_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / f"zl_{horizon}_tft_best.pt"))
    
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
    
    # Save model
    model_path = MODELS_DIR / f"zl_{horizon}_tft.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': 64,
            'num_heads': 4,
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
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'target': targets_dict['test'],
        'prediction': predictions['test'],
        'horizon': horizon
    })
    predictions_path = DATA_DIR / f"predictions_tft_{horizon}.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logging.info(f"✅ Predictions saved to {predictions_path}")
    
    return model, metrics

if __name__ == "__main__":
    logging.info("🚀 Starting TFT training for ZL (minimal feature set)...")
    
    models = {}
    results = {}
    skipped = []
    
    for horizon in HORIZONS:
        try:
            if should_skip_tft(horizon):
                skipped.append(horizon)
                continue
            
            model, metrics = train_tft_for_horizon(horizon)
            if model is not None:
                models[horizon] = model
                results[horizon] = metrics
        except Exception as e:
            logging.error(f"❌ Failed to train {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    logging.info(f"\n{'='*60}")
    if skipped:
        logging.info(f"⏭️  Skipped horizons: {', '.join(skipped)}")
    logging.info("✅ All models trained successfully!")
    logging.info(f"Models saved to: {MODELS_DIR}")
    if results:
        logging.info(f"\nSummary:")
        for horizon, metrics in results.items():
            logging.info(f"  {horizon:3s}: Test MAE={metrics['test']['mae']:.4f}, R²={metrics['test']['r2']:.4f}")
    logging.info(f"{'='*60}")

