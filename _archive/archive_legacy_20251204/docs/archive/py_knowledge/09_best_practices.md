---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# PyTorch Best Practices for CBI-V15

## Essential Do's and Don'ts for Commodity Forecasting

### Architecture Best Practices

#### ✅ DO: Use Appropriate Model Architecture

```python
# GOOD: LSTM/GRU for time series
class GoodTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=15, hidden_size=256, 
                           num_layers=3, batch_first=True)
        
# BAD: Using only feedforward networks for sequences
class BadTimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(15 * 252, 256)  # Flattening time series loses temporal info
```

#### ✅ DO: Initialize Weights Properly

```python
# GOOD: Proper initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)  # Orthogonal for RNNs
            elif 'bias' in name:
                nn.init.constant_(param, 0)

model.apply(init_weights)
```

#### ❌ DON'T: Use Inappropriate Architectures

```python
# BAD: Vanilla RNN for long sequences
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(15, 256)  # Suffers from vanishing gradients
        
# BAD: Too deep for limited data
class TooDeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(256, 256) for _ in range(100)])
```

### Data Handling Best Practices

#### ✅ DO: Normalize Financial Data Properly

```python
# GOOD: Normalize with statistics from training set only
class DataNormalizer:
    def fit(self, train_data):
        self.mean = train_data.mean()
        self.std = train_data.std()
        return self
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean

# BAD: Normalizing with test set statistics (data leakage)
all_data = pd.concat([train_data, test_data])
normalized = (all_data - all_data.mean()) / all_data.std()  # WRONG!
```

#### ✅ DO: Handle Missing Data Appropriately

```python
# GOOD: Forward fill for time series continuity
df['price'].fillna(method='ffill', inplace=True)

# GOOD: Interpolation for small gaps
df['price'].interpolate(method='time', limit=5, inplace=True)

# BAD: Using mean for time series
df['price'].fillna(df['price'].mean(), inplace=True)  # Loses temporal patterns
```

#### ❌ DON'T: Leak Future Information

```python
# BAD: Using future data in features
df['future_return'] = df['price'].shift(-1) / df['price'] - 1
features = df[['price', 'volume', 'future_return']]  # LEAK!

# GOOD: Only use past data
df['past_return'] = df['price'].pct_change()  # Uses previous values
```

### Training Best Practices

#### ✅ DO: Use Proper Training Techniques

```python
# GOOD: Gradient clipping for RNNs
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

# GOOD: Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# GOOD: Early stopping
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(epochs):
    val_loss = validate(model, val_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
```

#### ❌ DON'T: Train Without Validation

```python
# BAD: No validation split
model.fit(all_data, all_targets, epochs=1000)  # Overfitting likely

# BAD: Not monitoring validation metrics
for epoch in range(1000):
    train_model(train_loader)  # No validation check
```

### Memory Management Best Practices

#### ✅ DO: Manage GPU Memory Efficiently

```python
# GOOD: Clear gradients properly
optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

# GOOD: Use no_grad for inference
with torch.no_grad():
    predictions = model(test_data)

# GOOD: Delete unnecessary tensors
del intermediate_tensor
torch.cuda.empty_cache()

# GOOD: Use mixed precision training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
```

#### ❌ DON'T: Accumulate Gradients Unintentionally

```python
# BAD: Forgetting to zero gradients
for epoch in range(epochs):
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Gradients accumulate!
    optimizer.step()  # Should call zero_grad first
```

### Performance Best Practices

#### ✅ DO: Optimize for Your Hardware

```python
# GOOD: Use MPS on M4 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# GOOD: Use DataLoader optimizations
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)

# GOOD: Compile model (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")
```

#### ✅ DO: Profile Before Optimizing

```python
# GOOD: Profile to find bottlenecks
import torch.profiler

with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Deployment Best Practices

#### ✅ DO: Export Models Properly

```python
# GOOD: Use TorchScript for production
model.eval()
example_input = torch.randn(1, 252, 15)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model.pt")

# GOOD: Include preprocessing in exported model
class DeployableModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x):
        # Normalize input
        x = (x - self.mean) / self.std
        # Run model
        return self.model(x)
```

#### ❌ DON'T: Deploy Without Testing

```python
# BAD: Direct deployment without validation
model.save("production_model.pt")
deploy_to_production()  # No testing!

# GOOD: Comprehensive testing before deployment
def test_deployment_model(model_path):
    model = torch.jit.load(model_path)
    
    # Test with various inputs
    test_cases = generate_test_cases()
    for test_input, expected_shape in test_cases:
        output = model(test_input)
        assert output.shape == expected_shape
    
    # Benchmark performance
    benchmark_results = benchmark(model)
    assert benchmark_results['latency'] < 10  # ms
    
    return True
```

### Error Handling Best Practices

#### ✅ DO: Handle Errors Gracefully

```python
# GOOD: Robust error handling
class RobustPredictor:
    def predict(self, data):
        try:
            # Validate input
            if data.shape[1] != self.expected_features:
                raise ValueError(f"Expected {self.expected_features} features")
            
            # Check for NaN/Inf
            if torch.isnan(data).any() or torch.isinf(data).any():
                raise ValueError("Input contains NaN or Inf")
            
            # Make prediction
            with torch.no_grad():
                output = self.model(data)
            
            # Validate output
            if torch.isnan(output).any():
                raise RuntimeError("Model produced NaN values")
            
            return output
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return safe default or raise
            raise
```

### Testing Best Practices

#### ✅ DO: Test Model Components

```python
# GOOD: Unit test model components
import unittest

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = CommodityModel()
    
    def test_forward_shape(self):
        input = torch.randn(32, 252, 15)
        output = self.model(input)
        self.assertEqual(output['predictions'].shape, (32, 4, 5))
    
    def test_gradient_flow(self):
        input = torch.randn(1, 252, 15, requires_grad=True)
        output = self.model(input)
        loss = output['predictions'].sum()
        loss.backward()
        self.assertIsNotNone(input.grad)
```

### Debugging Best Practices

#### ✅ DO: Use Debugging Tools

```python
# GOOD: Use hooks for debugging
def print_grad(name):
    def hook(grad):
        print(f"{name} gradient: min={grad.min()}, max={grad.max()}")
    return hook

# Register hooks
for name, param in model.named_parameters():
    param.register_hook(print_grad(name))

# GOOD: Check for gradient issues
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if (param.grad == 0).all():
                print(f"Zero gradient in {name}")
```

## CBI-V15 Specific Best Practices

### Financial Data Considerations

```python
# DO: Account for market hours
def filter_trading_hours(df):
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    return df[(df['hour'] >= 9) & (df['hour'] < 16)]

# DO: Handle holidays and weekends
def remove_non_trading_days(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.dayofweek < 5]  # Remove weekends
    # Remove holidays (use market calendar)
    return df

# DO: Use log returns for stability
df['log_return'] = np.log(df['price'] / df['price'].shift(1))

# DON'T: Use raw prices without normalization
model_input = df['price'].values  # BAD - scale issues
```

### Model Evaluation for Trading

```python
# DO: Use appropriate financial metrics
def evaluate_trading_model(predictions, actuals):
    metrics = {}
    
    # Directional accuracy (most important for trading)
    direction_pred = np.sign(predictions[1:] - predictions[:-1])
    direction_true = np.sign(actuals[1:] - actuals[:-1])
    metrics['directional_accuracy'] = (direction_pred == direction_true).mean()
    
    # Sharpe ratio
    returns = predictions - actuals.shift(1)
    metrics['sharpe_ratio'] = returns.mean() / returns.std()
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    return metrics
```

### Production Monitoring

```python
# DO: Monitor model performance in production
class ModelMonitor:
    def __init__(self, alert_threshold=0.1):
        self.baseline_metrics = {}
        self.alert_threshold = alert_threshold
    
    def check_prediction_drift(self, predictions):
        """Check if predictions are drifting from baseline"""
        
        current_mean = predictions.mean()
        current_std = predictions.std()
        
        if abs(current_mean - self.baseline_metrics['mean']) > self.alert_threshold:
            self.send_alert("Prediction mean drift detected")
        
        if abs(current_std - self.baseline_metrics['std']) > self.alert_threshold:
            self.send_alert("Prediction variance drift detected")
```

## Summary Checklist

### Before Training
- [ ] Data properly normalized with train stats only
- [ ] No future information leakage
- [ ] Validation set created
- [ ] Hardware optimizations enabled (MPS/CUDA)
- [ ] Gradient clipping configured for RNNs

### During Training
- [ ] Monitoring validation metrics
- [ ] Early stopping enabled
- [ ] Learning rate scheduling active
- [ ] Gradient health checks
- [ ] Memory usage monitored

### Before Deployment
- [ ] Model thoroughly tested
- [ ] Performance benchmarked
- [ ] Error handling implemented
- [ ] Preprocessing included in model
- [ ] Monitoring system ready

### In Production
- [ ] Performance metrics tracked
- [ ] Drift detection active
- [ ] Fallback mechanism ready
- [ ] Update pipeline tested
- [ ] Logging comprehensive

---

*Following these best practices ensures robust, efficient, and reliable commodity forecasting with CBI-V15*


