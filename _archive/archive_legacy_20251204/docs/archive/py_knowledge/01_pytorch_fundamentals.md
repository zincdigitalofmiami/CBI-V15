---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# PyTorch Fundamentals for CBI-V15

## Core Concepts Essential for Commodity Forecasting

### 1. Tensors - The Foundation

**What they are**: Multi-dimensional arrays that can run on GPU (or Apple Silicon via MPS)

**CBI-V15 Application**:
```python
import torch

# Example: Representing commodity price data
# Shape: [batch_size, sequence_length, features]
# Features might include: open, high, low, close, volume, indicators
commodity_data = torch.randn(32, 252, 15)  # 32 samples, 252 trading days, 15 features

# Move to Apple Silicon GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
commodity_data = commodity_data.to(device)
```

**Key Operations for Time Series**:
- `.view()` / `.reshape()` - Restructure data for different model layers
- `.squeeze()` / `.unsqueeze()` - Add/remove dimensions
- `.permute()` - Rearrange dimensions (critical for LSTM input)

### 2. Autograd - Automatic Differentiation

**What it is**: PyTorch's automatic differentiation engine

**CBI-V15 Relevance**:
- Automatically computes gradients for backpropagation
- Essential for training our forecasting models
- Tracks operations on tensors with `requires_grad=True`

```python
# Example: Tracking gradients for model parameters
prices = torch.tensor([100.0, 105.0, 103.0], requires_grad=True)
prediction = prices.mean() * 1.02  # Simple prediction model
prediction.backward()  # Computes gradients
print(prices.grad)  # Shows how each price affects the prediction
```

### 3. Neural Network Module (nn.Module)

**What it is**: Base class for all neural network modules

**CBI-V15 Custom Model Structure**:
```python
import torch.nn as nn

class CommodityForecaster(nn.Module):
    def __init__(self, input_features=15, hidden_size=128, num_commodities=5, horizons=4):
        super().__init__()
        
        # Feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(64, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        # Attention mechanism for important time steps
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Output heads for different prediction horizons (3, 6, 9, 12 months)
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_size, num_commodities) for _ in range(horizons)
        ])
        
    def forward(self, x):
        # x shape: [batch, sequence, features]
        batch_size, seq_len, _ = x.shape
        
        # Extract features
        x = self.feature_extractor(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Generate predictions for each horizon
        predictions = []
        final_hidden = attn_out[:, -1, :]  # Use last time step
        
        for head in self.prediction_heads:
            predictions.append(head(final_hidden))
            
        return torch.stack(predictions, dim=1)  # [batch, horizons, commodities]
```

### 4. DataLoaders and Datasets

**What they are**: Utilities for handling data batching and loading

**CBI-V15 Implementation**:
```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CommodityDataset(Dataset):
    def __init__(self, data_path, sequence_length=252, forecast_horizons=[3, 6, 9, 12]):
        """
        Custom dataset for commodity price forecasting
        
        Args:
            data_path: Path to parquet/csv files
            sequence_length: Number of historical days to use
            forecast_horizons: Months ahead to predict
        """
        self.data = pd.read_parquet(data_path)
        self.sequence_length = sequence_length
        self.horizons = forecast_horizons
        
        # Normalize features
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.normalized_data = (self.data - self.mean) / self.std
        
    def __len__(self):
        return len(self.data) - self.sequence_length - max(self.horizons) * 21  # ~21 trading days/month
        
    def __getitem__(self, idx):
        # Get historical sequence
        sequence = self.normalized_data.iloc[idx:idx + self.sequence_length].values
        
        # Get targets for each horizon
        targets = []
        for horizon in self.horizons:
            target_idx = idx + self.sequence_length + (horizon * 21)
            targets.append(self.normalized_data.iloc[target_idx].values)
            
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor(np.array(targets))
        )

# Usage
dataset = CommodityDataset('data/commodity_prices.parquet')
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster GPU transfer
)
```

### 5. Optimizers

**What they are**: Algorithms for updating model parameters

**Best for CBI-V15**:
```python
import torch.optim as optim

model = CommodityForecaster()

# AdamW - Best for our use case
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,  # L2 regularization
    betas=(0.9, 0.999)  # Momentum parameters
)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Epochs
    eta_min=1e-6
)
```

### 6. Loss Functions

**Critical for Forecasting**:
```python
class WeightedForecastLoss(nn.Module):
    """Custom loss for multi-horizon forecasting"""
    
    def __init__(self, horizon_weights=[1.0, 0.8, 0.6, 0.4]):
        super().__init__()
        self.weights = torch.tensor(horizon_weights)
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        # predictions/targets shape: [batch, horizons, commodities]
        losses = self.mse(predictions, targets)
        
        # Weight by horizon (near-term predictions more important)
        weighted_losses = losses * self.weights.view(1, -1, 1)
        
        return weighted_losses.mean()
```

### 7. Training Loop Pattern

**Optimized for M4 Mac**:
```python
def train_model(model, dataloader, optimizer, criterion, epochs=100):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (sequences, targets) in enumerate(dataloader):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            loss.backward()
            
            # Gradient clipping (important for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        # Log progress
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Adjust learning rate
        scheduler.step()
```

### 8. Model Persistence

**Saving for Vertex AI Deployment**:
```python
# Save complete model (architecture + weights)
torch.save(model, 'models/commodity_forecaster_full.pth')

# Save just state dict (recommended)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'mean': dataset.mean,
    'std': dataset.std
}, 'models/checkpoint.pth')

# Export for production (TorchScript)
model.eval()
example_input = torch.randn(1, 252, 15).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('models/commodity_forecaster_traced.pt')
```

## Key Takeaways for CBI-V15

1. **Always use MPS backend** on M4 Mac for GPU acceleration
2. **Batch operations** are crucial for performance
3. **Mixed precision training** can speed up training significantly
4. **DataLoaders with pin_memory** improve GPU transfer speeds
5. **TorchScript export** enables deployment to Vertex AI

## Common Pitfalls to Avoid

1. **Don't forget to normalize** commodity price data
2. **Don't use simple RNNs** - always LSTM/GRU for long sequences
3. **Don't ignore gradient clipping** with time series models
4. **Don't train on CPU** when MPS is available
5. **Don't forget to set model.eval()** during inference

## Next Steps

Continue to [Deep Dive Optimizations](./02_deep_dive_optimizations.md) for advanced performance tuning techniques.

---

*Source: [PyTorch Tutorials - Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)*


