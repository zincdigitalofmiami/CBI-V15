---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# Neural Network Recipes for CBI-V15

## Practical Patterns for Commodity Price Forecasting

### 1. Defining a Complete Neural Network

**Core Structure for Time Series Forecasting**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class CommodityForecastNetwork(nn.Module):
    """
    Complete neural network for multi-horizon commodity price forecasting
    
    Architecture:
    - Feature extraction layers
    - Temporal processing (LSTM/GRU)
    - Attention mechanisms
    - Multi-task output heads
    """
    
    def __init__(
        self,
        input_features: int = 15,  # OHLCV + indicators
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_commodities: int = 5,
        forecast_horizons: List[int] = [3, 6, 9, 12],
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super(CommodityForecastNetwork, self).__init__()
        
        # Store configuration
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_commodities = num_commodities
        self.forecast_horizons = forecast_horizons
        self.use_attention = use_attention
        
        # Feature extraction layers
        self.feature_extractor = self._build_feature_extractor()
        
        # Temporal processing
        self.temporal_encoder = self._build_temporal_encoder(num_layers, dropout)
        
        # Attention mechanism
        if use_attention:
            self.attention = self._build_attention()
        
        # Output heads for each horizon
        self.prediction_heads = self._build_prediction_heads()
        
        # Auxiliary outputs (confidence, volatility)
        self.confidence_head = nn.Linear(hidden_dim, len(forecast_horizons))
        self.volatility_head = nn.Linear(hidden_dim, num_commodities)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction layers"""
        return nn.Sequential(
            nn.Linear(self.input_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
    
    def _build_temporal_encoder(self, num_layers: int, dropout: float) -> nn.Module:
        """Build LSTM encoder for temporal patterns"""
        return nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Causal for forecasting
        )
    
    def _build_attention(self) -> nn.Module:
        """Build multi-head attention layer"""
        return nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def _build_prediction_heads(self) -> nn.ModuleDict:
        """Build separate prediction head for each horizon"""
        heads = nn.ModuleDict()
        for horizon in self.forecast_horizons:
            heads[f"horizon_{horizon}"] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.num_commodities)
            )
        return heads
    
    def _init_weights(self, module):
        """Custom weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, sequence, features]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions, confidence, and optional attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Temporal encoding
        lstm_out, (hidden, cell) = self.temporal_encoder(features)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            attended_features, attention_weights = self.attention(
                lstm_out, lstm_out, lstm_out
            )
            # Combine with residual connection
            lstm_out = lstm_out + attended_features
        
        # Use last hidden state for predictions
        final_hidden = lstm_out[:, -1, :]
        
        # Generate predictions for each horizon
        predictions = {}
        for horizon in self.forecast_horizons:
            head_name = f"horizon_{horizon}"
            predictions[head_name] = self.prediction_heads[head_name](final_hidden)
        
        # Auxiliary predictions
        confidence = torch.sigmoid(self.confidence_head(final_hidden))
        volatility = F.softplus(self.volatility_head(final_hidden))
        
        # Prepare output
        output = {
            'predictions': predictions,
            'confidence': confidence,
            'volatility': volatility
        }
        
        if return_attention and attention_weights is not None:
            output['attention'] = attention_weights
            
        return output
```

### 2. State Dict Management Recipe

**Saving and Loading Model States**:

```python
class ModelCheckpointing:
    """Comprehensive model checkpointing for training continuity"""
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        loss: float,
        metrics: Dict,
        filepath: str,
        additional_info: Dict = None
    ):
        """Save complete training state"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'model_config': {
                'input_features': model.input_features,
                'hidden_dim': model.hidden_dim,
                'num_commodities': model.num_commodities,
                'forecast_horizons': model.forecast_horizons
            }
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        if additional_info:
            checkpoint['additional_info'] = additional_info
            
        # Save with atomic write (prevent corruption)
        temp_path = filepath + '.tmp'
        torch.save(checkpoint, temp_path)
        
        # Rename atomically
        import os
        os.replace(temp_path, filepath)
        
        print(f"Checkpoint saved: {filepath}")
        
    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu'
    ) -> Dict:
        """Load checkpoint and restore training state"""
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'metrics': checkpoint.get('metrics', {}),
            'additional_info': checkpoint.get('additional_info', {})
        }
    
    @staticmethod
    def export_for_inference(
        model: nn.Module,
        example_input: torch.Tensor,
        export_path: str,
        optimize: bool = True
    ):
        """Export model for production inference"""
        
        model.eval()
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        if optimize:
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save traced model
        traced_model.save(export_path)
        
        # Also save as ONNX for cross-platform deployment
        onnx_path = export_path.replace('.pt', '.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['predictions', 'confidence', 'volatility'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to: {export_path} and {onnx_path}")
```

### 3. Gradient Management Recipe

**Proper Gradient Handling**:

```python
class GradientManagement:
    """Best practices for gradient management"""
    
    @staticmethod
    def zero_gradients_efficiently(optimizer: torch.optim.Optimizer):
        """Efficient gradient zeroing"""
        # More efficient than optimizer.zero_grad()
        for param in optimizer.param_groups:
            for p in param['params']:
                if p.grad is not None:
                    p.grad = None  # Faster than p.grad.zero_()
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """Clip gradients to prevent explosion"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm, 
            norm_type
        )
        return total_norm.item()
    
    @staticmethod
    def adaptive_gradient_clipping(
        model: nn.Module,
        clip_factor: float = 0.01
    ):
        """AGC - Adaptive Gradient Clipping"""
        for param in model.parameters():
            if param.grad is None:
                continue
                
            param_norm = param.norm(2)
            grad_norm = param.grad.norm(2)
            
            if param_norm > 0 and grad_norm > 0:
                max_norm = param_norm * clip_factor
                if grad_norm > max_norm:
                    param.grad.mul_(max_norm / grad_norm)
    
    @staticmethod
    def accumulate_gradients(
        model: nn.Module,
        data_loader,
        criterion,
        optimizer,
        accumulation_steps: int = 4
    ):
        """Gradient accumulation for larger effective batch sizes"""
        
        model.train()
        optimizer.zero_grad()
        
        for i, (data, target) in enumerate(data_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output['predictions'], target)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                # Clip gradients
                GradientManagement.clip_gradients(model)
                
                # Optimizer step
                optimizer.step()
                
                # Zero gradients
                GradientManagement.zero_gradients_efficiently(optimizer)
```

### 4. Custom Loss Functions Recipe

```python
class CommodityLossFunctions:
    """Custom loss functions for commodity forecasting"""
    
    @staticmethod
    def directional_loss(predictions, targets, alpha=0.5):
        """
        Combines MSE with directional accuracy
        
        Args:
            predictions: Model predictions
            targets: True values
            alpha: Weight for directional component
        """
        # MSE component
        mse = F.mse_loss(predictions, targets, reduction='none')
        
        # Directional component
        pred_direction = torch.sign(predictions[:, 1:] - predictions[:, :-1])
        true_direction = torch.sign(targets[:, 1:] - targets[:, :-1])
        direction_accuracy = (pred_direction == true_direction).float()
        direction_loss = 1.0 - direction_accuracy
        
        # Combine losses
        total_loss = alpha * mse.mean() + (1 - alpha) * direction_loss.mean()
        
        return total_loss
    
    @staticmethod
    def sharpe_ratio_loss(predictions, targets, risk_free_rate=0.02):
        """
        Optimize for Sharpe ratio instead of pure accuracy
        """
        returns = predictions - targets.roll(1, dims=1)
        returns = returns[:, 1:]  # Remove first element
        
        mean_return = returns.mean(dim=1)
        std_return = returns.std(dim=1)
        
        sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8)
        
        # Negative because we want to maximize Sharpe
        return -sharpe.mean()
    
    @staticmethod
    def quantile_loss(predictions, targets, quantiles=[0.1, 0.5, 0.9]):
        """
        Quantile regression loss for uncertainty estimation
        """
        losses = []
        
        for i, q in enumerate(quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max(q * errors, (q - 1) * errors))
        
        return torch.stack(losses).mean()
```

### 5. Learning Rate Scheduling Recipe

```python
class LearningRateSchedulers:
    """Advanced learning rate scheduling strategies"""
    
    @staticmethod
    def get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5
    ):
        """Cosine schedule with linear warmup"""
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            
            return max(
                0.0, 
                0.5 * (1.0 + torch.cos(torch.tensor(math.pi * float(num_cycles) * 2.0 * progress)))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def get_exponential_decay_schedule(
        optimizer,
        decay_rate: float = 0.96,
        decay_steps: int = 1000
    ):
        """Exponential decay schedule"""
        
        def lr_lambda(current_step: int):
            return decay_rate ** (current_step / decay_steps)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def get_one_cycle_schedule(
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3
    ):
        """One-cycle learning rate schedule"""
        
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
```

### 6. Data Augmentation Recipe

```python
class TimeSeriesAugmentation:
    """Data augmentation techniques for time series"""
    
    @staticmethod
    def add_noise(data: torch.Tensor, noise_level: float = 0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    @staticmethod
    def time_shift(data: torch.Tensor, max_shift: int = 5):
        """Random time shifting"""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift > 0:
            return F.pad(data[shift:], (0, 0, shift, 0))
        elif shift < 0:
            return F.pad(data[:shift], (0, 0, 0, -shift))
        return data
    
    @staticmethod
    def magnitude_scaling(data: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)):
        """Random magnitude scaling"""
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        return data * scale
    
    @staticmethod
    def window_slicing(data: torch.Tensor, slice_ratio: float = 0.9):
        """Random window slicing"""
        seq_len = data.shape[0]
        slice_len = int(seq_len * slice_ratio)
        start_idx = torch.randint(0, seq_len - slice_len + 1, (1,)).item()
        return data[start_idx:start_idx + slice_len]
    
    @staticmethod
    def mixup(data1: torch.Tensor, data2: torch.Tensor, alpha: float = 0.2):
        """Mixup augmentation"""
        lam = torch.tensor(np.random.beta(alpha, alpha))
        return lam * data1 + (1 - lam) * data2
```

### 7. Model Ensemble Recipe

```python
class EnsembleModel(nn.Module):
    """Ensemble multiple models for robust predictions"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Weighted ensemble prediction"""
        all_predictions = []
        all_confidences = []
        
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            all_predictions.append(output['predictions'])
            all_confidences.append(output['confidence'])
        
        # Stack and weight predictions
        stacked_predictions = torch.stack(all_predictions, dim=0)
        weighted_predictions = (stacked_predictions * self.weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        # Confidence as weighted average
        stacked_confidences = torch.stack(all_confidences, dim=0)
        weighted_confidence = (stacked_confidences * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        return {
            'predictions': weighted_predictions,
            'confidence': weighted_confidence,
            'individual_predictions': all_predictions
        }
    
    def train_ensemble(self, train_loader, val_loader, epochs: int = 100):
        """Train ensemble with different initializations"""
        
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.001)
            for model in self.models
        ]
        
        for epoch in range(epochs):
            # Train each model
            for i, (model, optimizer) in enumerate(zip(self.models, optimizers)):
                model.train()
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Add different noise to each model's input
                    noisy_data = TimeSeriesAugmentation.add_noise(
                        data, 
                        noise_level=0.01 * (i + 1)
                    )
                    
                    optimizer.zero_grad()
                    output = model(noisy_data)
                    loss = F.mse_loss(output['predictions'], target)
                    loss.backward()
                    optimizer.step()
            
            # Validate ensemble
            self.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.forward(data)
                    val_loss += F.mse_loss(output['predictions'], target).item()
            
            print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")
```

### 8. Performance Benchmarking Recipe

```python
import time
import numpy as np

class ModelBenchmark:
    """Comprehensive model benchmarking"""
    
    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_iterations: int = 1000,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """Benchmark model inference performance"""
        
        model = model.to(device).eval()
        input_tensor = torch.randn(input_shape).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Synchronize if using GPU
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Timing
        times = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': 1.0 / np.mean(times),
            'latency_p50': np.percentile(times, 50),
            'latency_p95': np.percentile(times, 95),
            'latency_p99': np.percentile(times, 99)
        }
    
    @staticmethod
    def profile_memory(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """Profile memory usage"""
        
        if device != 'cuda':
            print("Memory profiling only available for CUDA")
            return {}
        
        model = model.to(device).eval()
        input_tensor = torch.randn(input_shape).to(device)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get memory stats
        allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        
        return {
            'peak_memory_mb': allocated,
            'reserved_memory_mb': reserved,
            'model_size_mb': sum(p.numel() * p.element_size() 
                                for p in model.parameters()) / 1024**2
        }
```

## Complete Training Pipeline Recipe

```python
class CommodityTrainingPipeline:
    """Complete training pipeline for CBI-V15"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'mps'
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = LearningRateSchedulers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_loader) * 50  # 50 epochs
        )
        
        # Setup loss
        self.criterion = CommodityLossFunctions.directional_loss
        
        # Checkpointing
        self.checkpointer = ModelCheckpointing()
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Data augmentation
            if np.random.random() < 0.5:
                data = TimeSeriesAugmentation.add_noise(data)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output['predictions'], target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            GradientManagement.clip_gradients(self.model)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = F.mse_loss(output['predictions'], target)
                total_loss += loss.item()
                
                # Directional accuracy
                pred_direction = torch.sign(output['predictions'][:, 1:] - 
                                          output['predictions'][:, :-1])
                true_direction = torch.sign(target[:, 1:] - target[:, :-1])
                accuracy = (pred_direction == true_direction).float().mean()
                total_accuracy += accuracy.item()
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': total_accuracy / len(self.val_loader)
        }
    
    def train(self, epochs: int = 50):
        """Complete training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_loss,
                    val_metrics,
                    'best_model.pt'
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.checkpointer.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    train_loss,
                    val_metrics,
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
```

## Key Takeaways for CBI-V15

1. **Always define clear model architecture** with proper initialization
2. **Use checkpointing** to save training progress
3. **Implement proper gradient management** to prevent instabilities
4. **Create custom loss functions** for domain-specific objectives
5. **Use ensemble methods** for robust predictions
6. **Benchmark thoroughly** before deployment

## Next Steps

Continue to [TorchCodec](./05_torchcodec.md) for multimedia processing capabilities.

---

*Sources: [PyTorch Recipes](https://pytorch.org/tutorials/recipes_index.html) & [Defining Neural Networks](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)*


