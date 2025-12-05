---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# Deep Dive Optimizations for CBI-V15

## Advanced PyTorch Techniques for Commodity Price Forecasting

### 1. Profiling PyTorch Models

**Why it matters for CBI-V15**: Identify bottlenecks in our forecasting pipeline

```python
import torch
import torch.profiler
from torch.profiler import ProfilerActivity

def profile_commodity_model(model, dataloader):
    """Profile model to identify performance bottlenecks"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= 10:  # Profile first 10 batches
                    break
                data = data.to(device)
                with torch.no_grad():
                    _ = model(data)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Export to Chrome tracing format
    prof.export_chrome_trace("trace.json")
    
    # Identify memory bottlenecks
    print("\nMemory Usage:")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

**Key Metrics to Monitor**:
- CPU/GPU time per operation
- Memory allocation patterns
- Data transfer overhead between CPU and GPU
- Kernel launch overhead

### 2. Model Parametrization

**Application**: Constrain model weights for stability in financial predictions

```python
import torch.nn.utils.parametrize as parametrize

class PositiveLinear(nn.Module):
    """Ensures weights remain positive (useful for price relationships)"""
    def forward(self, X):
        return X.abs()

class OrthogonalLSTM(nn.Module):
    """LSTM with orthogonal weight constraints for gradient stability"""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Apply orthogonal parametrization to recurrent weights
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                parametrize.register_parametrization(
                    self.lstm, name, parametrize.orthogonal()
                )

# Example: Constrained commodity relationship model
class ConstrainedCommodityModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Correlation layer with positive weights
        self.correlation_layer = nn.Linear(10, 10, bias=False)
        parametrize.register_parametrization(
            self.correlation_layer, "weight", PositiveLinear()
        )
        
        # Orthogonal LSTM for stable gradients
        self.lstm = OrthogonalLSTM(10, 128)
```

### 3. Model Pruning

**Purpose**: Reduce model size for faster inference on Vertex AI

```python
import torch.nn.utils.prune as prune

def prune_commodity_model(model, sparsity=0.3):
    """Prune model weights to reduce size and improve inference speed"""
    
    parameters_to_prune = []
    
    # Collect all Linear and Conv layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply structured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # Remove pruning reparametrization (make permanent)
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    # Calculate sparsity
    total_params = 0
    pruned_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        pruned_params += (param == 0).sum().item()
    
    print(f"Sparsity: {100 * pruned_params / total_params:.2f}%")
    print(f"Model size reduction: {pruned_params / 1e6:.2f}M parameters removed")
    
    return model

# Custom pruning for time series models
class TemporalImportancePruning:
    """Prune based on temporal feature importance"""
    
    @staticmethod
    def compute_importance(model, dataloader):
        """Compute importance scores for each weight"""
        importance_scores = {}
        
        model.eval()
        for name, param in model.named_parameters():
            importance_scores[name] = torch.zeros_like(param)
        
        # Accumulate gradients as importance measure
        for data, target in dataloader:
            data = data.requires_grad_()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    importance_scores[name] += param.grad.abs()
        
        return importance_scores
```

### 4. Inductor and torch.compile()

**Massive Performance Gains**: JIT compilation for production

```python
import torch._inductor.config

# Configure Inductor for M4 Mac
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = True

@torch.compile(mode="reduce-overhead", backend="inductor")
def optimized_forecast_model(model, x):
    """Compiled version of model for 2-3x faster inference"""
    return model(x)

# Specific optimizations for time series
@torch.compile(mode="max-autotune")
def optimized_lstm_cell(input, hidden, cell, weight_ih, weight_hh, bias):
    """Optimized LSTM cell computation"""
    gates = torch.mm(input, weight_ih.t()) + torch.mm(hidden, weight_hh.t()) + bias
    
    # Efficient gate computation
    i, f, g, o = gates.chunk(4, 1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    
    new_cell = f * cell + i * g
    new_hidden = o * torch.tanh(new_cell)
    
    return new_hidden, new_cell

# Benchmark compiled vs regular
def benchmark_compilation(model, input_tensor, iterations=100):
    import time
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Regular execution
    start = time.time()
    for _ in range(iterations):
        _ = model(input_tensor)
    regular_time = time.time() - start
    
    # Compiled execution
    compiled_model = torch.compile(model, mode="reduce-overhead")
    
    # Warmup compiled model
    for _ in range(10):
        _ = compiled_model(input_tensor)
    
    start = time.time()
    for _ in range(iterations):
        _ = compiled_model(input_tensor)
    compiled_time = time.time() - start
    
    print(f"Regular: {regular_time:.3f}s")
    print(f"Compiled: {compiled_time:.3f}s")
    print(f"Speedup: {regular_time/compiled_time:.2f}x")
```

### 5. Scaled Dot Product Attention (SDPA)

**Optimized Attention for Commodity Relationships**:

```python
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

class OptimizedCommodityAttention(nn.Module):
    """
    High-performance attention mechanism for cross-commodity relationships
    Uses Flash Attention when available
    """
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use optimized SDPA (automatically uses Flash Attention if available)
        attn_output = scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Not causal for commodity relationships
            scale=1.0 / (self.head_dim ** 0.5)
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        return self.out_proj(attn_output)

# Benchmark against standard attention
def compare_attention_performance():
    batch_size, seq_len, embed_dim = 32, 252, 512
    x = torch.randn(batch_size, seq_len, embed_dim).cuda()
    
    # Standard attention
    standard_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
    
    # Optimized SDPA attention
    optimized_attn = OptimizedCommodityAttention(embed_dim, 8)
    
    # Time both
    import time
    
    # Standard
    start = time.time()
    for _ in range(100):
        _ = standard_attn(x, x, x)
    print(f"Standard: {time.time() - start:.3f}s")
    
    # Optimized
    start = time.time()
    for _ in range(100):
        _ = optimized_attn(x)
    print(f"Optimized SDPA: {time.time() - start:.3f}s")
```

### 6. Knowledge Distillation

**Train Smaller Models for Edge Deployment**:

```python
class KnowledgeDistillation:
    """Distill large model knowledge into smaller, deployable model"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
    def distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Combined loss: distillation + true label loss"""
        
        # Distillation loss (soft targets)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        distillation_loss *= self.temperature ** 2
        
        # True label loss (hard targets)
        student_loss = F.mse_loss(student_logits, true_labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss
    
    def train_student(self, dataloader, epochs=50):
        """Train student model using teacher's knowledge"""
        
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=0.001)
        self.teacher.eval()  # Teacher in eval mode
        
        for epoch in range(epochs):
            total_loss = 0
            
            for data, labels in dataloader:
                # Get teacher predictions (no gradients needed)
                with torch.no_grad():
                    teacher_outputs = self.teacher(data)
                
                # Student predictions
                student_outputs = self.student(data)
                
                # Calculate distillation loss
                loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
                
                # Optimize student
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Example: Create a smaller student model
def create_student_model(teacher_model):
    """Create a smaller version of teacher model"""
    
    # Assuming teacher has certain dimensions
    student = nn.Sequential(
        nn.Linear(teacher_model.input_dim, 64),  # Smaller hidden size
        nn.ReLU(),
        nn.LSTM(64, 32, batch_first=True),  # Smaller LSTM
        nn.Linear(32, teacher_model.output_dim)
    )
    
    return student
```

### 7. Mixed Precision Training

**Faster Training on M4 Mac**:

```python
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training(model, dataloader, epochs=100):
    """Train with automatic mixed precision for 2x speedup"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For gradient scaling
    
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                output = model(data)
                loss = F.mse_loss(output, target)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping in mixed precision
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
```

### 8. Channels Last Memory Format

**Optimize Memory Layout for Convolutions**:

```python
def optimize_memory_format(model):
    """Convert model to channels_last format for better performance"""
    
    model = model.to(memory_format=torch.channels_last)
    
    # Also convert inputs
    def optimized_forward(x):
        x = x.to(memory_format=torch.channels_last)
        return model(x)
    
    return optimized_forward
```

## Performance Optimization Checklist for CBI-V15

### ✅ Must-Do Optimizations

1. **Profile First**: Always profile before optimizing
2. **Use torch.compile()**: 2-3x speedup for free
3. **Enable Mixed Precision**: Faster training with minimal accuracy loss
4. **Batch Operations**: Maximize GPU utilization
5. **Data Pipeline**: Use DataLoader with multiple workers
6. **Gradient Accumulation**: For larger effective batch sizes

### ⚠️ Conditional Optimizations

1. **Pruning**: Only if model size is a constraint
2. **Quantization**: For edge deployment
3. **Knowledge Distillation**: When deploying to resource-constrained environments
4. **Custom CUDA Kernels**: Only for bottleneck operations

### ❌ Avoid These

1. **Premature Optimization**: Profile first
2. **Over-Pruning**: Can hurt accuracy significantly
3. **Aggressive Quantization**: Financial data needs precision
4. **Ignoring Data Pipeline**: Often the real bottleneck

## M4 Mac Specific Optimizations

```python
# Check and use MPS backend
if torch.backends.mps.is_available():
    device = torch.device("mps")
    
    # Set MPS specific options
    torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    torch.mps.empty_cache()  # Clear cache when needed
else:
    device = torch.device("cpu")
    
    # CPU optimizations
    torch.set_num_threads(8)  # Use all performance cores
    torch.set_float32_matmul_precision('high')
```

## Next Steps

Continue to [Extensions & Custom Operators](./03_extensions_custom_ops.md) for building specialized operations.

---

*Source: [PyTorch Deep Dive Tutorials](https://docs.pytorch.org/tutorials/deep-dive.html)*


