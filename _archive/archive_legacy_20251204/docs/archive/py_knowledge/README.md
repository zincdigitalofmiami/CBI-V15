---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# PyTorch Knowledge Base for CBI-V15 Project

## Overview

This knowledge base contains comprehensive documentation on PyTorch specifically tailored for the CBI-V15 commodity price forecasting project. All content is analyzed and documented with our specific use case in mind: training neural networks on local M4 Mac hardware and uploading predictions to BigQuery for dashboard consumption.

## Project Context

- **Hardware**: M4 Mac (local training + inference) → BigQuery (storage + dashboard)
- **Use Case**: ZL (Soybean Oil Futures) price forecasting with multiple time horizons (1w, 1m, 3m, 6m, 12m)
- **Data**: 25+ years of historical commodity data (2000-2025) from multiple sources
- **Current Architecture**: Hybrid Python + BigQuery SQL (already in production)
  - BigQuery: Light calculations (correlations, regimes), scheduling, storage
  - Python: Complex features (sentiment, NLP, policy extraction)
  - PyTorch: Training on M4 Mac, uploads predictions to BigQuery

## Knowledge Base Structure

### 1. Core PyTorch
- **[PyTorch Fundamentals](./01_pytorch_fundamentals.md)** - Essential concepts for CBI-V15
- **[Deep Dive Optimizations](./02_deep_dive_optimizations.md)** - Performance tuning for our models
- **[Extensions & Custom Ops](./03_extensions_custom_ops.md)** - Custom operators for specialized needs
- **[Neural Network Recipes](./04_neural_network_recipes.md)** - Practical patterns for commodity forecasting

### 2. Specialized Tools
- **[TorchCodec](./05_torchcodec.md)** - Multimedia processing capabilities (future enhancement potential)
- **[ExecuTorch](./06_executorch_deployment.md)** - On-device deployment options
- **[CoreML Backend](./07_coreml_integration.md)** - Apple Silicon optimization for M4 Mac

### 3. CBI-V15 Specific Implementation
- **[Implementation Guide](./08_cbi_v14_implementation.md)** - How to apply PyTorch to our project
- **[Best Practices](./09_best_practices.md)** - Do's and don'ts for commodity forecasting
- **[Performance Benchmarks](./10_performance_benchmarks.md)** - M4 Mac specific optimizations
- **[BigQuery Integration](./14_BIGQUERY_INTEGRATION.md)** - CRITICAL: Actual production architecture

## Key Capabilities for CBI-V15

### ✅ What We CAN Do

1. **Local M4 Mac Training**
   - Leverage Apple Silicon GPU acceleration via MPS backend
   - Use mixed precision training for faster convergence
   - Implement data parallelism across M4's CPU cores

2. **Model Architecture**
   - Build custom LSTM/GRU networks for time series
   - Implement attention mechanisms for feature importance
   - Create ensemble models for robust predictions

3. **Optimization**
   - Profile and optimize model performance
   - Implement quantization for deployment
   - Use torch.compile() for faster inference

4. **Deployment**
   - Upload predictions to BigQuery (production path)
   - Dashboard reads from BigQuery views
   - Optional: Export to ExecuTorch for edge inference

### ⚠️ What We SHOULD NOT Do

1. **Avoid These Anti-patterns**
   - Don't use AutoML (we need custom architectures)
   - Don't rely on BQML (limited for our complex models)
   - Don't train directly on Vertex AI (expensive, use M4 Mac)

2. **Performance Pitfalls**
   - Avoid unnecessary CPU-GPU transfers
   - Don't use single precision where double is needed for financial data
   - Avoid blocking operations in data loading

3. **Architecture Mistakes**
   - Don't use vanilla RNNs (use LSTM/GRU for long sequences)
   - Avoid overly deep networks for limited data
   - Don't ignore temporal patterns in commodity data

## Quick Start for CBI-V15

```python
# Example: Setting up PyTorch for M4 Mac training with BigQuery integration
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from google.cloud import bigquery

# Check MPS availability (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.mps.allow_tf32 = True  # Enable TF32 for faster training
print(f"Using device: {device}")

# Load data from BigQuery training table
client = bigquery.Client(project='cbi-v15')
query = """
SELECT * FROM `cbi-v15.training.zl_training_prod_allhistory_1m`
WHERE date >= '2000-01-01' ORDER BY date
"""
df = client.query(query).to_dataframe()

# Example model for ZL (Soybean Oil) price forecasting
class ZLTemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_size=50, num_filters=64, output_horizons=5):
        super().__init__()
        # TCN architecture (often best for commodities)
        self.tcn = TemporalConvolutionalNetwork(input_size, num_filters)
        self.heads = nn.ModuleDict({
            h: nn.Linear(num_filters, 1) 
            for h in ['1w', '1m', '3m', '6m', '12m']
        })
        
    def forward(self, x):
        encoded = self.tcn(x)
        return {h: head(encoded) for h, head in self.heads.items()}
```

## Documentation Sources

All documentation is derived from official PyTorch resources:
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [PyTorch Recipes](https://docs.pytorch.org/tutorials/recipes/)
- [TorchCodec Documentation](https://meta-pytorch.org/torchcodec/)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)

## Navigation

Start with [PyTorch Fundamentals](./01_pytorch_fundamentals.md) for core concepts, then proceed to specialized topics based on your current development needs.

---

*Last Updated: November 17, 2025*
*Tailored for CBI-V15 Commodity Price Forecasting Project*

## Recent Updates (November 17, 2025)
- ✅ Architecture audit completed - confirmed hybrid Python + BigQuery SQL pattern
- See `docs/plans/FINAL_GPT_INTEGRATION_DIRECTIVE.md` for complete integration plan


