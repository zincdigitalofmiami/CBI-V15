---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# ExecuTorch for CBI-V15 On-Device Deployment

## Lightweight PyTorch Runtime for Edge Deployment

### Overview

ExecuTorch is a PyTorch framework for deploying models on edge devices, mobile platforms, and embedded systems. For CBI-V15, this enables running commodity forecasting models directly on trading terminals, mobile apps, or edge servers.

### Key Benefits for CBI-V15

1. **Low Latency**: Sub-millisecond inference for real-time trading decisions
2. **Privacy**: Keep sensitive trading strategies on-device
3. **Offline Capability**: Continue forecasting without internet connection
4. **Resource Efficiency**: Optimized for limited compute/memory

## Installation and Setup

```bash
# Install ExecuTorch
pip install executorch

# Additional backends (optional)
pip install executorch-coreml  # For Apple devices
pip install executorch-qnn     # For Qualcomm devices
pip install executorch-xnnpack  # For CPU optimization
```

## Converting CBI-V15 Models to ExecuTorch

### 1. Model Export Pipeline

```python
import torch
import executorch
from executorch.exir import to_edge
from executorch.exir.backend import BackendDetails
from torch.export import export

class CommodityModelExporter:
    """
    Export CBI-V15 models to ExecuTorch format
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        
    def prepare_model(self):
        """
        Prepare model for export (remove training-specific components)
        """
        # Remove dropout layers (replaced with identity in eval mode)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        
        # Fuse batch norm layers for efficiency
        torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']] if hasattr(self.model, 'conv') else [],
            inplace=True
        )
        
        return self.model
    
    def export_to_executorch(
        self,
        example_input: torch.Tensor,
        output_path: str = "commodity_model.pte",
        optimize_for_device: str = "cpu"
    ):
        """
        Export model to ExecuTorch format
        
        Args:
            example_input: Sample input tensor [batch, sequence, features]
            output_path: Path to save .pte file
            optimize_for_device: Target device type
        """
        # Prepare model
        model = self.prepare_model()
        
        # Export to EXIR (ExecuTorch IR)
        exported_program = export(
            model,
            (example_input,),
            dynamic_shapes={
                "x": {0: torch.export.Dim("batch", min=1, max=128)}
            }
        )
        
        # Convert to Edge format
        edge_program = to_edge(exported_program)
        
        # Apply device-specific optimizations
        if optimize_for_device == "apple":
            edge_program = self._optimize_for_apple(edge_program)
        elif optimize_for_device == "android":
            edge_program = self._optimize_for_android(edge_program)
        elif optimize_for_device == "cpu":
            edge_program = self._optimize_for_cpu(edge_program)
        
        # Save ExecuTorch program
        with open(output_path, "wb") as f:
            f.write(edge_program.buffer())
        
        print(f"Model exported to {output_path}")
        print(f"Model size: {os.path.getsize(output_path) / 1024:.2f} KB")
        
        return edge_program
    
    def _optimize_for_apple(self, edge_program):
        """Apply Apple-specific optimizations"""
        from executorch.backends.apple.coreml import CoreMLBackend
        
        backend = CoreMLBackend()
        edge_program = edge_program.to_backend(backend)
        
        return edge_program
    
    def _optimize_for_android(self, edge_program):
        """Apply Android-specific optimizations"""
        from executorch.backends.qnn import QNNBackend
        
        backend = QNNBackend()
        edge_program = edge_program.to_backend(backend)
        
        return edge_program
    
    def _optimize_for_cpu(self, edge_program):
        """Apply CPU optimizations using XNNPACK"""
        from executorch.backends.xnnpack import XNNPACKBackend
        
        backend = XNNPACKBackend()
        edge_program = edge_program.to_backend(backend)
        
        return edge_program
```

### 2. Model Quantization for Edge

```python
import torch.quantization as quant

class EdgeModelQuantizer:
    """
    Quantize models for efficient edge deployment
    """
    
    @staticmethod
    def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
        """
        Dynamic quantization (easiest, least accuracy loss)
        """
        quantized_model = quant.quantize_dynamic(
            model,
            qconfig_spec={
                torch.nn.Linear: quant.default_dynamic_qconfig,
                torch.nn.LSTM: quant.default_dynamic_qconfig,
            },
            dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def quantize_static(
        model: torch.nn.Module,
        calibration_data: torch.utils.data.DataLoader
    ) -> torch.nn.Module:
        """
        Static quantization (requires calibration data)
        """
        # Set quantization config
        model.qconfig = quant.get_default_qconfig('fbgemm')
        
        # Prepare model
        quant.prepare(model, inplace=True)
        
        # Calibrate with representative data
        model.eval()
        with torch.no_grad():
            for data, _ in calibration_data:
                model(data)
        
        # Convert to quantized model
        quantized_model = quant.convert(model, inplace=True)
        
        return quantized_model
    
    @staticmethod
    def quantize_aware_training(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5
    ) -> torch.nn.Module:
        """
        Quantization-aware training (best accuracy)
        """
        # Prepare QAT
        model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        model_qat = quant.prepare_qat(model.train())
        
        # Fine-tune with quantization
        optimizer = torch.optim.AdamW(model_qat.parameters(), lr=0.0001)
        
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model_qat(data)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()
        
        # Convert to quantized
        model_qat.eval()
        quantized_model = quant.convert(model_qat, inplace=True)
        
        return quantized_model
```

### 3. Runtime Implementation

```python
class CommodityEdgeInference:
    """
    Run commodity forecasting on edge devices
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ExecuTorch runtime
        
        Args:
            model_path: Path to .pte file
        """
        import executorch.runtime
        
        self.runtime = executorch.runtime.Runtime(model_path)
        self.model = self.runtime.load()
        
        # Get input/output specifications
        self.input_spec = self.model.get_input_spec(0)
        self.output_spec = self.model.get_output_spec(0)
        
        print(f"Model loaded. Input shape: {self.input_spec.shape}")
        
    def preprocess(self, raw_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw commodity data for inference
        """
        # Normalize based on training statistics
        mean = np.array([100.0, 105.0, 95.0, 102.0, 1000000])  # Example
        std = np.array([10.0, 12.0, 8.0, 11.0, 500000])
        
        normalized = (raw_data - mean) / std
        
        # Convert to tensor with correct dtype
        tensor = torch.from_numpy(normalized).float()
        
        return tensor
    
    def predict(
        self,
        input_data: np.ndarray,
        return_confidence: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on edge device
        
        Args:
            input_data: Raw commodity data [sequence, features]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and optional confidence
        """
        # Preprocess
        input_tensor = self.preprocess(input_data)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        # Run inference
        start_time = time.perf_counter()
        output = self.model.forward(input_tensor)
        inference_time = time.perf_counter() - start_time
        
        # Parse outputs
        predictions = output[0].numpy()
        
        results = {
            'predictions': predictions,
            'inference_time_ms': inference_time * 1000
        }
        
        if return_confidence and len(output) > 1:
            results['confidence'] = output[1].numpy()
        
        return results
    
    def benchmark(self, num_iterations: int = 1000):
        """
        Benchmark model performance on device
        """
        # Create dummy input
        dummy_input = np.random.randn(*self.input_spec.shape[1:])
        
        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input, return_confidence=False)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.predict(dummy_input, return_confidence=False)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        print(f"Benchmark Results ({num_iterations} iterations):")
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Std: {np.std(times):.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print(f"  Max: {np.max(times):.2f} ms")
        print(f"  P50: {np.percentile(times, 50):.2f} ms")
        print(f"  P95: {np.percentile(times, 95):.2f} ms")
        print(f"  P99: {np.percentile(times, 99):.2f} ms")
```

## Platform-Specific Deployments

### 1. iOS Deployment

```swift
// CommodityPredictor.swift
import ExecuTorch
import CoreML

class CommodityPredictor {
    private var model: ETModel!
    
    init(modelPath: String) {
        // Load ExecuTorch model
        self.model = try! ETModel(contentsOf: URL(fileURLWithPath: modelPath))
    }
    
    func predict(data: [[Float]]) -> PredictionResult {
        // Convert to tensor
        let inputTensor = ETTensor(shape: [1, data.count, data[0].count], 
                                  data: data.flatMap { $0 })
        
        // Run inference
        let startTime = CFAbsoluteTimeGetCurrent()
        let output = try! model.forward(inputTensor)
        let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Parse results
        let predictions = output[0].toArray()
        
        return PredictionResult(
            predictions: predictions,
            inferenceTimeMs: inferenceTime * 1000
        )
    }
}
```

### 2. Android Deployment

```java
// CommodityPredictor.java
package com.cbi.v14.edge;

import org.pytorch.executorch.*;

public class CommodityPredictor {
    private Module model;
    
    public CommodityPredictor(String modelPath) {
        // Load ExecuTorch model
        this.model = Module.load(modelPath);
    }
    
    public PredictionResult predict(float[][] data) {
        // Convert to tensor
        long[] shape = {1, data.length, data[0].length};
        Tensor inputTensor = Tensor.fromBlob(flatten(data), shape);
        
        // Run inference
        long startTime = System.currentTimeMillis();
        IValue output = model.forward(IValue.from(inputTensor));
        long inferenceTime = System.currentTimeMillis() - startTime;
        
        // Parse results
        Tensor outputTensor = output.toTensor();
        float[] predictions = outputTensor.getDataAsFloatArray();
        
        return new PredictionResult(predictions, inferenceTime);
    }
    
    private float[] flatten(float[][] array) {
        // Flatten 2D array to 1D
        int totalLength = array.length * array[0].length;
        float[] result = new float[totalLength];
        int index = 0;
        for (float[] row : array) {
            for (float value : row) {
                result[index++] = value;
            }
        }
        return result;
    }
}
```

### 3. Web Deployment (WebAssembly)

```javascript
// commodityPredictor.js
class CommodityPredictor {
    constructor() {
        this.model = null;
    }
    
    async loadModel(modelPath) {
        // Load ExecuTorch WASM module
        const Module = await import('./executorch_wasm.js');
        await Module.ready;
        
        // Load model
        const modelBuffer = await fetch(modelPath).then(r => r.arrayBuffer());
        this.model = Module.loadModel(new Uint8Array(modelBuffer));
    }
    
    predict(data) {
        if (!this.model) {
            throw new Error('Model not loaded');
        }
        
        // Convert to flat array
        const flatData = data.flat();
        const inputPtr = Module._malloc(flatData.length * 4);
        Module.HEAPF32.set(flatData, inputPtr / 4);
        
        // Run inference
        const startTime = performance.now();
        const outputPtr = this.model.forward(inputPtr, data.length, data[0].length);
        const inferenceTime = performance.now() - startTime;
        
        // Read output
        const outputSize = this.model.getOutputSize();
        const predictions = new Float32Array(
            Module.HEAPF32.buffer,
            outputPtr,
            outputSize
        );
        
        // Cleanup
        Module._free(inputPtr);
        Module._free(outputPtr);
        
        return {
            predictions: Array.from(predictions),
            inferenceTimeMs: inferenceTime
        };
    }
}
```

## Memory and Performance Optimization

### 1. Memory-Efficient Loading

```python
class MemoryEfficientLoader:
    """
    Load models efficiently on memory-constrained devices
    """
    
    @staticmethod
    def load_with_memory_map(model_path: str):
        """
        Use memory mapping to reduce RAM usage
        """
        import mmap
        
        with open(model_path, 'rb') as f:
            # Memory map the file
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Load model from memory map
            model = executorch.runtime.Runtime(mmapped_file)
            
            return model
    
    @staticmethod
    def load_partial_model(model_path: str, layers_to_load: List[str]):
        """
        Load only specific layers for specialized tasks
        """
        runtime = executorch.runtime.Runtime(model_path)
        
        # Load only specified layers
        partial_model = runtime.load_partial(layers_to_load)
        
        return partial_model
```

### 2. Batching for Throughput

```python
class BatchedEdgeInference:
    """
    Optimize throughput with batching
    """
    
    def __init__(self, model_path: str, max_batch_size: int = 32):
        self.model = executorch.runtime.Runtime(model_path).load()
        self.max_batch_size = max_batch_size
        self.batch_queue = []
        
    def add_to_batch(self, data: np.ndarray, callback):
        """
        Add request to batch queue
        """
        self.batch_queue.append((data, callback))
        
        # Process if batch is full
        if len(self.batch_queue) >= self.max_batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """
        Process accumulated batch
        """
        if not self.batch_queue:
            return
        
        # Stack inputs
        batch_data = np.stack([item[0] for item in self.batch_queue])
        
        # Run batch inference
        batch_tensor = torch.from_numpy(batch_data)
        outputs = self.model.forward(batch_tensor)
        
        # Distribute results
        for i, (_, callback) in enumerate(self.batch_queue):
            result = outputs[i].numpy()
            callback(result)
        
        # Clear queue
        self.batch_queue.clear()
```

## Monitoring and Debugging

### 1. Performance Profiler

```python
class EdgePerformanceProfiler:
    """
    Profile ExecuTorch model performance
    """
    
    def __init__(self, model_path: str):
        self.model = executorch.runtime.Runtime(model_path).load()
        self.metrics = []
        
    def profile_inference(self, input_data: torch.Tensor):
        """
        Profile single inference
        """
        import psutil
        import os
        
        # Memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU usage before
        cpu_before = process.cpu_percent()
        
        # Run inference
        start_time = time.perf_counter()
        output = self.model.forward(input_data)
        end_time = time.perf_counter()
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU usage after
        cpu_after = process.cpu_percent()
        
        metrics = {
            'inference_time_ms': (end_time - start_time) * 1000,
            'memory_used_mb': mem_after - mem_before,
            'cpu_usage': (cpu_before + cpu_after) / 2,
            'output_shape': output.shape
        }
        
        self.metrics.append(metrics)
        
        return metrics
    
    def generate_report(self):
        """
        Generate performance report
        """
        if not self.metrics:
            return "No metrics collected"
        
        df = pd.DataFrame(self.metrics)
        
        report = f"""
        ExecuTorch Performance Report
        =============================
        
        Inference Time:
          Mean: {df['inference_time_ms'].mean():.2f} ms
          Std:  {df['inference_time_ms'].std():.2f} ms
          Min:  {df['inference_time_ms'].min():.2f} ms
          Max:  {df['inference_time_ms'].max():.2f} ms
        
        Memory Usage:
          Mean: {df['memory_used_mb'].mean():.2f} MB
          Peak: {df['memory_used_mb'].max():.2f} MB
        
        CPU Usage:
          Mean: {df['cpu_usage'].mean():.1f}%
          Peak: {df['cpu_usage'].max():.1f}%
        """
        
        return report
```

## CBI-V15 Edge Deployment Strategy

### Deployment Options

1. **Trading Terminal App**: Deploy on trader workstations
2. **Mobile App**: iOS/Android for on-the-go analysis
3. **Edge Server**: Local server in trading office
4. **Web Browser**: WebAssembly for universal access

### Recommended Architecture

```python
class CBI_V14_EdgeDeployment:
    """
    Complete edge deployment solution for CBI-V15
    """
    
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """
        Load different model variants for different devices
        """
        # Full model for edge servers
        self.models['full'] = self._load_model('models/commodity_full.pte')
        
        # Quantized model for mobile
        self.models['mobile'] = self._load_model('models/commodity_mobile_q8.pte')
        
        # Tiny model for web
        self.models['web'] = self._load_model('models/commodity_tiny.pte')
    
    def _load_model(self, path: str):
        """Load ExecuTorch model"""
        return executorch.runtime.Runtime(path).load()
    
    def select_model(self, device_type: str, available_memory: int):
        """
        Select appropriate model based on device capabilities
        """
        if device_type == 'server' and available_memory > 4096:
            return self.models['full']
        elif device_type in ['ios', 'android'] and available_memory > 512:
            return self.models['mobile']
        else:
            return self.models['web']
    
    def deploy(self, target_platform: str):
        """
        Deploy to specific platform
        """
        if target_platform == 'ios':
            self._deploy_ios()
        elif target_platform == 'android':
            self._deploy_android()
        elif target_platform == 'web':
            self._deploy_web()
        elif target_platform == 'edge_server':
            self._deploy_edge_server()
    
    def _deploy_ios(self):
        """Deploy to iOS App Store"""
        # Generate iOS framework
        os.system("python -m executorch.sdk.ios_framework --model models/commodity_mobile_q8.pte")
        
    def _deploy_android(self):
        """Deploy to Google Play"""
        # Generate AAR
        os.system("python -m executorch.sdk.android_aar --model models/commodity_mobile_q8.pte")
        
    def _deploy_web(self):
        """Deploy to web"""
        # Generate WASM module
        os.system("python -m executorch.sdk.wasm --model models/commodity_tiny.pte")
        
    def _deploy_edge_server(self):
        """Deploy to edge server"""
        # Package with Docker
        os.system("docker build -t cbi-v15-edge:latest .")
```

## Benefits for CBI-V15

### ✅ Why Use ExecuTorch

1. **Real-time Trading**: Sub-10ms predictions on edge
2. **Offline Capability**: Continue trading during outages
3. **Data Privacy**: Keep strategies on-device
4. **Cost Savings**: Reduce cloud inference costs
5. **Global Deployment**: Run anywhere, any device

### ⚠️ Considerations

1. **Model Size**: Need to optimize/quantize for edge
2. **Update Mechanism**: How to update deployed models
3. **Monitoring**: Track performance across devices
4. **Compatibility**: Test on target hardware

## Next Steps

Continue to [CoreML Integration](./07_coreml_integration.md) for Apple Silicon optimization.

---

*Source: [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/index.html)*


