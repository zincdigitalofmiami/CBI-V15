---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# Refined CBI-V15 PyTorch Architecture (Production-Ready)

## Critical Corrections and Refinements

Based on expert review, here are the **refined recommendations** that align with production best practices and the BigQuery/Mac architecture.

## ✅ Green Lights (Keep These)

### 1. MPS Backend Configuration

```python
import torch

# Enable MPS with optimizations
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Enable TF32 for faster training on MPS
torch.backends.mps.allow_tf32 = True

# Mixed precision training (AMP)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Usage in training loop
with autocast(dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)
```

### 2. torch.compile with Feature Flag

```python
# Feature flag for compilation
USE_COMPILE = True

if USE_COMPILE:
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ Model compiled successfully")
    except Exception as e:
        print(f"⚠️ Compilation failed, using uncompiled: {e}")
        USE_COMPILE = False
else:
    print("ℹ️ Compilation disabled")

# Realistic expectations: 1.2-2.0x speedup (not 2.5x guaranteed)
```

### 3. Gradient Clipping (Mandatory)

```python
# ALWAYS clip gradients for RNNs
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Multi-Horizon Heads (Correct Approach)

```python
class MultiHorizonHead(nn.Module):
    """
    Shared encoder + separate heads per horizon
    Better than recursive forecasting
    """
    def __init__(self, encoder_dim, horizons=[3, 6, 9, 12]):
        super().__init__()
        self.encoder = self._build_encoder()
        self.heads = nn.ModuleDict({
            f"horizon_{h}": nn.Linear(encoder_dim, 1) 
            for h in horizons
        })
    
    def forward(self, x):
        encoded = self.encoder(x)
        return {h: head(encoded) for h, head in self.heads.items()}
```

## ⚠️ Yellow Lights (Tune These)

### 1. Model Architecture: TCN vs LSTM+MHA

**CORRECTION**: LSTM+MHA is good, but **TCN often matches/bests LSTM** for commodity forecasting.

```python
class TemporalConvolutionalNetwork(nn.Module):
    """
    Baseline B: TCN often outperforms LSTM on noisy commodity data
    Compiles cleaner on MPS, converts easier to CoreML
    """
    def __init__(self, input_dim, num_filters=64, kernel_size=5, num_blocks=5):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        dilation = 1
        
        for i in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_dim if i == 0 else num_filters,
                        num_filters,
                        kernel_size,
                        dilation=dilation,
                        padding=(kernel_size - 1) * dilation
                    ),
                    nn.GroupNorm(8, num_filters),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            )
            dilation *= 2  # Exponential dilation: 1, 2, 4, 8, 16, 32
        
        self.bottleneck = nn.Linear(num_filters, 64)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x = x.mean(dim=-1)  # [batch, features]
        x = self.bottleneck(x)
        
        return x

# Run BOTH architectures in parallel bake-off
baselines = {
    'LSTM_MHA': LSTMWithAttention(),
    'TCN': TemporalConvolutionalNetwork(),
    'NBEATS': NBEATSx()  # Optional for interpretability
}
```

### 2. Feature Count: 30-60 Curated Features

**CORRECTION**: 15 features is **too skinny**. Use **30-60 curated features** from Big-8 pillars.

```python
class CuratedFeatureSet:
    """
    Curated 30-60 feature set (not 15, not 290)
    """
    
    FEATURE_CATEGORIES = {
        'prices': [
            'close', 'close_returns', 'log_returns',
            'high_low_spread', 'open_close_spread'
        ],
        
        'technical': [
            'RSI_14', 'MACD_signal', 'ATR_14',
            'BB_upper', 'BB_lower', 'BB_width',
            'ADX_14', 'CCI_14'
        ],
        
        'substitution_macro': [
            'palm_price', 'palm_spread',  # Critical for ZL (soybean oil)
            'palm_soybean_oil_spread',  # Direct substitution spread
            'soybean_crush_spread',  # ZL vs ZS (soybeans) relationship
            'soybean_meal_spread',  # ZL vs ZM relationship
            'WTI_price', 'WTI_spread',
            'USD_BRL', 'USD_BRL_returns',  # Critical for Brazil soybean exports
            'VIX_level', 'VIX_zscore',
            'yield_curve_slope',  # If available
            'renewable_diesel_mandate'  # Biofuel demand driver
        ],
        
        'weather': [
            # Critical for ZL: Brazil/Argentina are top soybean producers
            'brazil_precip_7d_zscore',  # Primary production region
            'brazil_precip_30d_zscore',
            'brazil_GDD_base10C',
            'argentina_precip_7d_zscore',  # Secondary production region
            'argentina_precip_30d_zscore',
            'argentina_GDD_base10C',
            # US Midwest (less critical for ZL vs corn/wheat)
            'midwest_precip_7d_zscore',
            'midwest_precip_30d_zscore',
            'midwest_GDD_base10C'
        ],
        
        'positioning_regime': [
            'CFTC_managed_money_long',
            'CFTC_managed_money_short',
            'CFTC_commercial_long',
            'CFTC_commercial_short',
            'regime_bullish',
            'regime_bearish',
            'regime_sideways',
            'regime_volatile'
        ]
    }
    
    @classmethod
    def get_feature_list(cls):
        """Get curated feature list (30-60 features)"""
        features = []
        for category, feature_names in cls.FEATURE_CATEGORIES.items():
            features.extend(feature_names)
        return features
    
    @classmethod
    def select_features_with_shap(cls, model, X, y):
        """Use SHAP to validate feature importance"""
        import shap
        
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # Select top features by importance
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = np.argsort(feature_importance)[-60:]  # Top 60
        
        return top_features_idx
```

### 3. Multi-Commodity: Start Single-Asset

**CORRECTION**: Start with **single-asset multi-horizon** (ZL = Soybean Oil Futures). Add others as context inputs later.

**Soybean Oil (ZL) Specific Considerations:**
- **Direct substitution**: Palm oil (FCPO) is primary substitute - critical context feature
- **Crush spread**: Relationship with soybeans (ZS) and soybean meal (ZM)
- **Weather drivers**: Brazil/Argentina soybean production (not just Midwest)
- **Biofuel demand**: Renewable diesel mandates affect demand
- **CFTC positioning**: Managed money flows are significant

```python
class SingleAssetMultiHorizon(nn.Module):
    """
    Start here: Single commodity (ZL = Soybean Oil Futures) with multiple horizons
    Add other commodities (palm oil, crude oil, etc.) as context inputs, not separate targets
    """
    
    def __init__(self, zl_features=50, context_features=20):
        super().__init__()
        
        # Primary asset encoder (ZL)
        self.zl_encoder = self._build_encoder(zl_features)
        
        # Context encoders (palm, crude as context)
        self.context_encoder = self._build_encoder(context_features)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),  # ZL + context
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Multi-horizon heads (ZL only)
        self.horizon_heads = nn.ModuleDict({
            f"horizon_{h}": nn.Linear(64, 1) 
            for h in [3, 6, 9, 12]
        })
    
    def forward(self, zl_data, context_data=None):
        zl_encoded = self.zl_encoder(zl_data)
        
        if context_data is not None:
            context_encoded = self.context_encoder(context_data)
            fused = self.fusion(torch.cat([zl_encoded, context_encoded], dim=-1))
        else:
            fused = zl_encoded
        
        predictions = {
            h: head(fused) 
            for h, head in self.horizon_heads.items()
        }
        
        return predictions
```

## 🚨 Red Flags (Critical Adjustments)

### 1. CoreML: NOT Primary Production Path

**CORRECTION**: **BigQuery is SSOT**. PyTorch inference on M4 → upload to BigQuery.

```python
class ProductionInferencePipeline:
    """
    CORRECT production path:
    1. Load parquet from BigQuery export
    2. Run PyTorch inference on M4
    3. Write predictions to BigQuery
    4. Dashboard reads from BigQuery views
    """
    
    def __init__(self, model_path):
        # Load PyTorch model (NOT CoreML)
        self.model = torch.load(model_path, map_location='mps')
        self.model.eval()
        
        # Load scaler (fitted on training only)
        self.scaler = joblib.load('scaler.pkl')
        
    def daily_inference(self, parquet_path):
        """Daily inference workflow"""
        # 1. Load latest parquet bundle
        df = pd.read_parquet(parquet_path)
        
        # 2. Preprocess (using training scaler)
        features = self.scaler.transform(df[self.feature_columns])
        tensor = torch.FloatTensor(features).unsqueeze(0).to('mps')
        
        # 3. Run PyTorch inference
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # 4. Write to BigQuery
        predictions_df = self._format_predictions(predictions)
        self._upload_to_bigquery(predictions_df)
        
        return predictions_df
    
    def export_coreml_optional(self):
        """
        Optional: Export CoreML for on-device demos
        NOT the canonical serving path
        """
        import coremltools as ct
        
        example_input = torch.randn(1, 252, 50)  # 50 features
        traced = torch.jit.trace(self.model, example_input)
        
        coreml_model = ct.convert(traced, inputs=[ct.TensorType(shape=example_input.shape)])
        coreml_model.save('model_optional_coreml.mlpackage')
        
        # Validate parity before using
        self._validate_coreml_parity(coreml_model)
```

### 2. Throughput Targets: Focus on Quality, Not Speed

**CORRECTION**: Optimize for **MAPE/Sharpe parity**, not 35k preds/s.

```python
class QualityFocusedTraining:
    """
    Focus on calibration quality, not raw throughput
    """
    
    def train_with_quality_gates(self):
        """Training with quality focus"""
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Quality gates (not speed)
            quality_checks = {
                'mape': val_metrics['mape'] < 0.05,  # 5% MAPE target
                'sharpe': val_metrics['sharpe'] > 1.0,
                'direction_accuracy': val_metrics['direction_acc'] > 0.65,
                'calibration': val_metrics['calibration_error'] < 0.1
            }
            
            if all(quality_checks.values()):
                self.save_model('best_model.pt')
            
            # Early stopping on MAPE
            if val_metrics['mape'] < best_mape:
                best_mape = val_metrics['mape']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
```

## 🏗️ Refined Architecture

### Model Family (Choose 2-3 for Bake-Off)

```python
class ModelFamily:
    """
    Three baselines to compare:
    A: LSTM + Scaled Dot-Product Attention
    B: TCN (often best for commodities)
    C: N-BEATSx (interpretable)
    """
    
    @staticmethod
    def create_baseline_a():
        """LSTM + Attention"""
        return LSTMWithScaledAttention(
            input_dim=50,  # 30-60 curated features
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            dropout=0.2
        )
    
    @staticmethod
    def create_baseline_b():
        """TCN (recommended)"""
        return TemporalConvolutionalNetwork(
            input_dim=50,
            num_filters=64,
            kernel_size=5,
            num_blocks=6
        )
    
    @staticmethod
    def create_baseline_c():
        """N-BEATSx (interpretable)"""
        return NBEATSx(
            input_dim=50,
            forecast_lengths=[3, 6, 9, 12],
            num_stacks=2,
            num_blocks=3
        )
```

### Loss Function (Refined)

```python
class RefinedLossFunction(nn.Module):
    """
    Huber loss with horizon weights + directionality penalty
    """
    
    def __init__(self, horizon_weights=[1.0, 0.9, 0.8, 0.7], delta=1.0):
        super().__init__()
        self.horizon_weights = torch.tensor(horizon_weights)
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction='none')
        
    def forward(self, predictions, targets):
        """
        predictions: [batch, horizons, 1]
        targets: [batch, horizons, 1]
        """
        # Huber loss per horizon
        huber_losses = self.huber(predictions, targets)
        
        # Weight by horizon (shorter horizons heavier)
        weighted_losses = huber_losses * self.horizon_weights.view(1, -1, 1)
        
        # Directionality penalty
        pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
        true_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
        direction_penalty = (pred_direction != true_direction).float().mean()
        
        # Combined loss
        total_loss = weighted_losses.mean() + 0.1 * direction_penalty
        
        return total_loss
```

### Feature Pipeline (Neural Driver Set)

```python
class NeuralDriverSet:
    """
    Curated 30-60 features with fixed order & dtype
    Locked after selection to avoid drift
    """
    
    def __init__(self):
        self.feature_list = CuratedFeatureSet.get_feature_list()
        self.scaler = RobustScaler()  # Or StandardScaler
        self.feature_order = None  # Locked after first fit
        
    def fit_transform(self, train_data):
        """Fit scaler on training ONLY"""
        # Select features
        X = train_data[self.feature_list]
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Lock feature order
        self.feature_order = self.feature_list.copy()
        
        # Persist scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_order, 'feature_order.pkl')
        
        return X_scaled
    
    def transform(self, data):
        """Transform using locked scaler and order"""
        if self.feature_order is None:
            self.feature_order = joblib.load('feature_order.pkl')
            self.scaler = joblib.load('scaler.pkl')
        
        X = data[self.feature_order]
        return self.scaler.transform(X)
```

### Training Loop (M4-Optimized)

```python
class M4OptimizedTraining:
    """
    Optimized for M4 Mac with MPS + AMP + compile
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Enable optimizations
        torch.backends.mps.allow_tf32 = True
        
        # Compile if available
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except:
            pass
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Loss
        self.criterion = RefinedLossFunction()
        
    def train_epoch(self):
        """Training with all optimizations"""
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward
            with autocast(dtype=torch.float16):
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Scaled backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
```

### Walk-Forward Cross-Validation

```python
class WalkForwardCV:
    """
    Time-based CV: rolling origin, never shuffle across time
    """
    
    def __init__(self, data, n_splits=5, train_size=252*3, test_size=252):
        self.data = data
        self.n_splits = n_splits
        self.train_size = train_size  # 3 years
        self.test_size = test_size    # 1 year
        
    def split(self):
        """Generate walk-forward splits"""
        total_size = len(self.data)
        
        for i in range(self.n_splits):
            train_start = i * self.test_size
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            if test_end > total_size:
                break
            
            train_idx = range(train_start, train_end)
            test_idx = range(test_start, test_end)
            
            yield train_idx, test_idx
    
    def evaluate(self, model_factory):
        """Evaluate model on all splits"""
        results = []
        
        for train_idx, test_idx in self.split():
            # Split data
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Train model
            model = model_factory()
            model.fit(train_data)
            
            # Evaluate
            metrics = model.evaluate(test_data)
            results.append(metrics)
        
        return {
            'mean_mape': np.mean([r['mape'] for r in results]),
            'std_mape': np.std([r['mape'] for r in results]),
            'mean_sharpe': np.mean([r['sharpe'] for r in results]),
            'per_split': results
        }
```

## Integration with BigQuery/Mac Architecture

```python
class BigQueryMacIntegration:
    """
    Complete integration with BigQuery/Mac architecture
    """
    
    def daily_pipeline(self):
        """Daily production pipeline"""
        
        # 1. BigQuery exports parquet with manifest
        # (Handled by Cloud Scheduler + Dataform)
        
        # 2. Mac syncs parquet (LaunchDaemon)
        parquet_path = self.sync_from_gcs()
        
        # 3. Load and validate
        df = pd.read_parquet(parquet_path)
        features = self.feature_pipeline.transform(df)
        
        # 4. Run PyTorch inference on M4
        predictions = self.model.predict(features)
        
        # 5. Calculate quality metrics
        metrics = self.calculate_metrics(predictions, df['actual'])
        
        # 6. Upload to BigQuery
        self.upload_predictions(predictions)
        self.upload_metrics(metrics)
        
        # 7. Parity gate
        if self.check_parity(metrics):
            self.mark_production_ready()
        else:
            self.hold_deployment()
```

## Summary of Corrections

| Aspect | Original | Corrected |
|--------|----------|-----------|
| **Model** | LSTM+MHA only | LSTM+MHA, TCN, N-BEATSx (bake-off) |
| **Features** | 15 | 30-60 curated |
| **Commodities** | Multi-output | Single-asset (ZL = Soybean Oil) first |
| **Inference** | CoreML primary | PyTorch → BigQuery |
| **Focus** | Throughput | MAPE/Sharpe quality |
| **CV** | Random split | Walk-forward |
| **Loss** | MSE | Huber + directionality |

---

*These refinements align with production best practices and the BigQuery/Mac architecture*
