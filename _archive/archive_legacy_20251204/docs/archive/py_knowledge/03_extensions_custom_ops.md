---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# PyTorch Extensions & Custom Operators for CBI-V15

## Building Specialized Operations for Commodity Forecasting

### 1. Custom Python Operators

**When to Use**: For complex business logic that doesn't need C++ speed

```python
import torch
from torch import Tensor
from typing import List, Tuple

class CommodityCorrelation(torch.autograd.Function):
    """
    Custom operator for computing dynamic commodity correlations
    with special handling for market regimes
    """
    
    @staticmethod
    def forward(ctx, prices: Tensor, window_size: int = 20, regime_threshold: float = 0.3):
        """
        Compute rolling correlations with regime detection
        
        Args:
            prices: [batch, time, commodities]
            window_size: Rolling window for correlation
            regime_threshold: Threshold for regime change detection
        """
        batch_size, seq_len, n_commodities = prices.shape
        
        # Compute returns
        returns = torch.diff(prices, dim=1) / prices[:, :-1, :]
        
        # Initialize correlation tensor
        correlations = torch.zeros(batch_size, seq_len - window_size, n_commodities, n_commodities)
        
        # Compute rolling correlations
        for i in range(seq_len - window_size):
            window_returns = returns[:, i:i+window_size, :]
            
            # Standardize returns
            mean = window_returns.mean(dim=1, keepdim=True)
            std = window_returns.std(dim=1, keepdim=True)
            normalized = (window_returns - mean) / (std + 1e-8)
            
            # Compute correlation matrix
            corr = torch.matmul(normalized.transpose(-1, -2), normalized) / window_size
            correlations[:, i, :, :] = corr
        
        # Save for backward pass
        ctx.save_for_backward(prices, returns, correlations)
        ctx.window_size = window_size
        
        return correlations
    
    @staticmethod
    def backward(ctx, grad_output):
        """Custom backward pass for correlation computation"""
        prices, returns, correlations = ctx.saved_tensors
        window_size = ctx.window_size
        
        # Approximate gradient (simplified for demonstration)
        grad_prices = torch.zeros_like(prices)
        
        # Gradient flows through returns calculation
        for i in range(grad_output.shape[1]):
            window_grad = grad_output[:, i, :, :]
            
            # Approximate contribution of each price to correlation
            price_contribution = torch.sum(window_grad, dim=(-1, -2), keepdim=True)
            grad_prices[:, i:i+window_size, :] += price_contribution.expand_as(
                prices[:, i:i+window_size, :]
            ) / window_size
        
        return grad_prices, None, None

# Register as custom op
commodity_correlation = CommodityCorrelation.apply

# Usage example
class MarketRegimeAwareModel(nn.Module):
    """Model that uses custom correlation operator"""
    
    def __init__(self, n_commodities=5):
        super().__init__()
        self.n_commodities = n_commodities
        
        # Process correlation features
        self.corr_processor = nn.Sequential(
            nn.Conv2d(n_commodities, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Final prediction
        self.predictor = nn.Linear(16 * n_commodities * n_commodities, n_commodities * 4)
        
    def forward(self, prices):
        # Compute custom correlations
        correlations = commodity_correlation(prices, window_size=20)
        
        # Process correlations
        batch_size = correlations.shape[0]
        seq_len = correlations.shape[1]
        
        # Reshape for conv2d [batch*seq, channels, height, width]
        corr_reshaped = correlations.view(-1, self.n_commodities, 
                                         self.n_commodities, self.n_commodities)
        
        # Apply convolutions
        features = self.corr_processor(corr_reshaped)
        
        # Flatten and predict
        features_flat = features.view(batch_size, seq_len, -1)
        predictions = self.predictor(features_flat[:, -1, :])  # Use last timestep
        
        return predictions.view(batch_size, 4, self.n_commodities)  # 4 horizons
```

### 2. Custom C++ Extensions

**For Performance-Critical Operations**:

First, create the C++ implementation:

```cpp
// commodity_ops.cpp
#include <torch/extension.h>
#include <vector>

// Fast volatility calculation for commodity prices
torch::Tensor calculate_garch_volatility(
    torch::Tensor returns,
    double omega,
    double alpha,
    double beta
) {
    auto options = torch::TensorOptions()
        .dtype(returns.dtype())
        .device(returns.device());
    
    auto volatility = torch::zeros_like(returns);
    auto variance = torch::full({returns.size(0), 1}, omega, options);
    
    for (int64_t t = 0; t < returns.size(1); ++t) {
        // GARCH(1,1) update
        auto returns_t = returns.index({torch::indexing::Slice(), t}).unsqueeze(1);
        variance = omega + alpha * returns_t.pow(2) + beta * variance;
        volatility.index_put_({torch::indexing::Slice(), t}, variance.sqrt().squeeze());
    }
    
    return volatility;
}

// Optimized technical indicators
torch::Tensor fast_technical_indicators(
    torch::Tensor prices,
    int64_t short_window,
    int64_t long_window
) {
    auto batch_size = prices.size(0);
    auto seq_len = prices.size(1);
    auto n_assets = prices.size(2);
    
    auto indicators = torch::zeros({batch_size, seq_len, n_assets, 4},
                                  prices.options());
    
    // Parallel computation of indicators
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t a = 0; a < n_assets; ++a) {
            auto asset_prices = prices.index({b, torch::indexing::Slice(), a});
            
            // Moving averages
            for (int64_t t = short_window; t < seq_len; ++t) {
                auto short_ma = asset_prices.slice(0, t - short_window, t).mean();
                indicators.index_put_({b, t, a, 0}, short_ma);
            }
            
            for (int64_t t = long_window; t < seq_len; ++t) {
                auto long_ma = asset_prices.slice(0, t - long_window, t).mean();
                indicators.index_put_({b, t, a, 1}, long_ma);
                
                // RSI approximation
                auto changes = asset_prices.slice(0, t - 14, t).diff();
                auto gains = changes.clamp_min(0).mean();
                auto losses = (-changes.clamp_max(0)).mean();
                auto rsi = 100 - (100 / (1 + gains / (losses + 1e-8)));
                indicators.index_put_({b, t, a, 2}, rsi);
                
                // Bollinger band width
                auto std = asset_prices.slice(0, t - 20, t).std();
                indicators.index_put_({b, t, a, 3}, std * 2);
            }
        }
    }
    
    return indicators;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_garch_volatility", &calculate_garch_volatility, "GARCH volatility");
    m.def("fast_technical_indicators", &fast_technical_indicators, "Technical indicators");
}
```

Setup script for building:

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='commodity_ops',
    ext_modules=[
        CppExtension(
            'commodity_ops',
            ['commodity_ops.cpp'],
            extra_compile_args=['-O3', '-fopenmp'],  # Enable OpenMP
            extra_link_args=['-fopenmp']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

Using the custom C++ operators:

```python
import commodity_ops

class EnhancedCommodityModel(nn.Module):
    """Model using custom C++ operators for speed"""
    
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4 + 4, 128, batch_first=True)  # 4 GARCH + 4 indicators
        self.predictor = nn.Linear(128, 20)  # 5 commodities * 4 horizons
        
    def forward(self, prices):
        # Calculate returns
        returns = torch.diff(prices, dim=1) / prices[:, :-1, :]
        
        # Fast GARCH volatility (C++)
        volatilities = []
        for i in range(returns.shape[-1]):
            vol = commodity_ops.calculate_garch_volatility(
                returns[:, :, i], 
                omega=0.01, 
                alpha=0.1, 
                beta=0.8
            )
            volatilities.append(vol)
        volatility = torch.stack(volatilities, dim=-1)
        
        # Fast technical indicators (C++)
        indicators = commodity_ops.fast_technical_indicators(prices, 10, 30)
        
        # Combine features
        features = torch.cat([
            volatility.unsqueeze(-1),
            indicators
        ], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features[:, 1:, :, :].view(
            features.shape[0], features.shape[1]-1, -1
        ))
        
        # Predictions
        return self.predictor(lstm_out[:, -1, :])
```

### 3. CUDA Extensions (If Using NVIDIA GPUs)

```cpp
// commodity_cuda_kernels.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void correlation_matrix_kernel(
    const float* __restrict__ returns,
    float* __restrict__ correlations,
    int batch_size,
    int window_size,
    int n_assets
) {
    int batch = blockIdx.x;
    int asset_i = blockIdx.y;
    int asset_j = threadIdx.x;
    
    if (asset_j >= n_assets) return;
    
    float sum = 0.0f;
    float sum_i = 0.0f;
    float sum_j = 0.0f;
    float sum_ii = 0.0f;
    float sum_jj = 0.0f;
    
    for (int t = 0; t < window_size; ++t) {
        int idx_i = batch * window_size * n_assets + t * n_assets + asset_i;
        int idx_j = batch * window_size * n_assets + t * n_assets + asset_j;
        
        float val_i = returns[idx_i];
        float val_j = returns[idx_j];
        
        sum += val_i * val_j;
        sum_i += val_i;
        sum_j += val_j;
        sum_ii += val_i * val_i;
        sum_jj += val_j * val_j;
    }
    
    float mean_i = sum_i / window_size;
    float mean_j = sum_j / window_size;
    float var_i = sum_ii / window_size - mean_i * mean_i;
    float var_j = sum_jj / window_size - mean_j * mean_j;
    
    float correlation = (sum / window_size - mean_i * mean_j) / 
                       (sqrtf(var_i) * sqrtf(var_j) + 1e-8f);
    
    int out_idx = batch * n_assets * n_assets + asset_i * n_assets + asset_j;
    correlations[out_idx] = correlation;
}

torch::Tensor cuda_correlation_matrix(torch::Tensor returns, int window_size) {
    auto batch_size = returns.size(0);
    auto n_assets = returns.size(2);
    
    auto correlations = torch::zeros({batch_size, n_assets, n_assets}, 
                                    returns.options());
    
    dim3 blocks(batch_size, n_assets);
    dim3 threads(n_assets);
    
    correlation_matrix_kernel<<<blocks, threads>>>(
        returns.data_ptr<float>(),
        correlations.data_ptr<float>(),
        batch_size,
        window_size,
        n_assets
    );
    
    return correlations;
}
```

### 4. Double Backward for Advanced Training

```python
class DoublBackwardGAN(nn.Module):
    """GAN for synthetic commodity data generation with double backward"""
    
    def __init__(self):
        super().__init__()
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
    def _build_generator(self):
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 252 * 5)  # 252 days * 5 commodities
        )
    
    def _build_discriminator(self):
        return nn.Sequential(
            nn.Linear(252 * 5, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(real_data.device)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients with create_graph=True for double backward
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,  # Enable double backward
            retain_graph=True
        )[0]
        
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
```

### 5. Fusing Operations for Performance

```python
class FusedCommodityBlock(nn.Module):
    """Fused operations for better performance"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        
        # Running stats for batch norm
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
    def forward(self, x):
        # Fused Linear + BatchNorm + ReLU
        return self._fused_forward(x)
    
    @torch.jit.script_method
    def _fused_forward(self, x):
        # Linear transform
        out = F.linear(x, self.weight, self.bias)
        
        # Fused batch norm + relu
        if self.training:
            mean = out.mean(dim=0, keepdim=True)
            var = out.var(dim=0, keepdim=True, unbiased=False)
            
            # Update running stats
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean.squeeze()
            self.running_var = 0.9 * self.running_var + 0.1 * var.squeeze()
            
            # Normalize and scale in one operation
            out = (out - mean) / torch.sqrt(var + 1e-5)
        else:
            out = (out - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        
        # Apply batch norm parameters and activation
        out = out * self.bn_weight + self.bn_bias
        return F.relu(out, inplace=True)  # In-place for memory efficiency
```

### 6. Registering Dispatched Operators

```python
import torch.library

# Define custom namespace for commodity operations
commodity_lib = torch.library.Library("commodity", "IMPL")

# Register operation schema
commodity_lib.define("seasonal_decompose(Tensor prices, int period) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(commodity_lib, "seasonal_decompose", "CPU")
def seasonal_decompose_cpu(prices, period):
    """CPU implementation of seasonal decomposition"""
    batch_size, seq_len, n_assets = prices.shape
    
    trend = torch.zeros_like(prices)
    seasonal = torch.zeros_like(prices)
    residual = torch.zeros_like(prices)
    
    # Moving average for trend
    for i in range(period // 2, seq_len - period // 2):
        trend[:, i, :] = prices[:, i - period // 2:i + period // 2 + 1, :].mean(dim=1)
    
    # Seasonal component
    detrended = prices - trend
    for i in range(seq_len):
        seasonal_idx = i % period
        seasonal[:, i, :] = detrended[:, seasonal_idx::period, :].mean(dim=1)
    
    # Residual
    residual = prices - trend - seasonal
    
    return trend, seasonal, residual

@torch.library.impl(commodity_lib, "seasonal_decompose", "MPS")
def seasonal_decompose_mps(prices, period):
    """Optimized MPS (Apple Silicon) implementation"""
    # MPS-specific optimizations
    return seasonal_decompose_cpu(prices.cpu(), period)

# Usage
def use_custom_op(prices):
    trend, seasonal, residual = torch.ops.commodity.seasonal_decompose(prices, 252)
    return trend, seasonal, residual
```

## Best Practices for Custom Operations

### ✅ When to Create Custom Ops

1. **Complex Business Logic**: Commodity-specific calculations
2. **Performance Bottlenecks**: Operations called frequently
3. **Memory Optimization**: Fusing multiple operations
4. **Special Hardware**: Leveraging MPS/CUDA capabilities

### ⚠️ Considerations

1. **Test Thoroughly**: Custom ops can introduce bugs
2. **Profile Performance**: Ensure actual speedup
3. **Maintain Compatibility**: Consider deployment targets
4. **Document Well**: Others need to understand your ops

### ❌ Avoid Custom Ops When

1. **Standard Ops Exist**: Don't reinvent the wheel
2. **One-time Operations**: Not worth the complexity
3. **Deployment Constraints**: Some platforms don't support
4. **Simple Logic**: Python is fast enough

## Integration with CBI-V15 Pipeline

```python
class CBI_V14_CustomOpsModel(nn.Module):
    """Complete model using custom operations"""
    
    def __init__(self):
        super().__init__()
        
        # Custom correlation extractor
        self.correlation_extractor = CommodityCorrelation.apply
        
        # Fused blocks for efficiency
        self.fused_block1 = FusedCommodityBlock(260, 128)
        self.fused_block2 = FusedCommodityBlock(128, 64)
        
        # Standard LSTM
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Output
        self.output = nn.Linear(128, 20)  # 5 commodities * 4 horizons
        
    def forward(self, prices):
        # Custom correlation features
        correlations = self.correlation_extractor(prices)
        
        # Technical indicators (C++ extension)
        indicators = commodity_ops.fast_technical_indicators(prices, 10, 30)
        
        # Seasonal decomposition (custom op)
        trend, seasonal, _ = torch.ops.commodity.seasonal_decompose(prices, 252)
        
        # Combine features
        features = torch.cat([
            correlations.view(prices.shape[0], -1, correlations.shape[-1] * correlations.shape[-2]),
            indicators.view(prices.shape[0], prices.shape[1], -1),
            trend,
            seasonal
        ], dim=-1)
        
        # Process through fused blocks
        x = self.fused_block1(features)
        x = self.fused_block2(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Predictions
        return self.output(lstm_out[:, -1, :])
```

## Next Steps

Continue to [Neural Network Recipes](./04_neural_network_recipes.md) for practical patterns.

---

*Source: [PyTorch Extension Tutorials](https://docs.pytorch.org/tutorials/extension.html)*


