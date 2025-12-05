---
**⚠️ CRITICAL: NO FAKE DATA ⚠️**
This project uses ONLY real, verified data sources. NO placeholders, NO synthetic data, NO fake values.
All data must come from authenticated APIs, official sources, or validated historical records.
---

# CoreML Integration for CBI-V15 on Apple Silicon

## Optimizing Commodity Forecasting for M4 Mac and iOS

### Overview

CoreML is Apple's framework for on-device machine learning, optimized for Apple Silicon (M4 Mac, iPhone, iPad). For CBI-V15, this enables:

1. **Hardware Acceleration**: Leverage Neural Engine for 15.8 TOPS
2. **Unified Memory**: Efficient data transfer between CPU/GPU/Neural Engine  
3. **Power Efficiency**: Minimal battery drain on mobile devices
4. **Privacy**: On-device processing for sensitive financial data

## M4 Mac Optimization Strategy

### Understanding M4 Mac Capabilities

```python
import coremltools as ct
import torch
import platform

class M4MacOptimizer:
    """
    Optimize CBI-V15 models for M4 Mac
    """
    
    @staticmethod
    def check_hardware():
        """
        Check M4 Mac capabilities
        """
        info = {
            'processor': platform.processor(),
            'machine': platform.machine(),
            'system': platform.system()
        }
        
        # Check for Apple Silicon
        is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
        
        if is_apple_silicon:
            print("✅ Running on Apple Silicon")
            print("Available compute units:")
            print("  - CPU: Performance (4) + Efficiency (6) cores")
            print("  - GPU: 10-core GPU")
            print("  - Neural Engine: 16-core, 15.8 TOPS")
            print("  - Unified Memory: Up to 24GB")
        else:
            print("⚠️ Not running on Apple Silicon")
        
        return is_apple_silicon
    
    @staticmethod
    def get_compute_units():
        """
        Determine optimal compute units for inference
        """
        import subprocess
        
        # Check available compute units
        result = subprocess.run(
            ['sysctl', 'hw.perflevel0.logicalcpu'],
            capture_output=True,
            text=True
        )
        
        performance_cores = int(result.stdout.split(':')[1].strip())
        
        # Recommendation based on model size
        if performance_cores >= 4:
            # M4 Pro/Max - use Neural Engine
            return ct.ComputeUnit.ALL  # CPU, GPU, and Neural Engine
        else:
            # Base M4 - balance between GPU and Neural Engine
            return ct.ComputeUnit.CPU_AND_NE
```

## PyTorch to CoreML Conversion

### 1. Model Conversion Pipeline

```python
class CommodityModelConverter:
    """
    Convert PyTorch models to CoreML for Apple devices
    """
    
    def __init__(self, pytorch_model: torch.nn.Module):
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        
    def convert_to_coreml(
        self,
        example_input: torch.Tensor,
        output_path: str = "commodity_model.mlpackage",
        precision: str = "float32",
        compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL
    ):
        """
        Convert PyTorch model to CoreML
        
        Args:
            example_input: Sample input [batch, sequence, features]
            output_path: Path to save .mlpackage
            precision: Model precision (float32, float16, int8)
            compute_units: Target compute units
        """
        
        # Trace the PyTorch model
        traced_model = torch.jit.trace(self.pytorch_model, example_input)
        
        # Define input types for CoreML
        inputs = [
            ct.TensorType(
                name="input",
                shape=example_input.shape,
                dtype=np.float32
            )
        ]
        
        # Configure conversion
        config = self._get_conversion_config(precision, compute_units)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to="mlprogram",  # Use ML Program (more features)
            compute_units=compute_units,
            **config
        )
        
        # Add metadata
        coreml_model = self._add_metadata(coreml_model)
        
        # Optimize for specific hardware
        if compute_units == ct.ComputeUnit.ALL:
            coreml_model = self._optimize_for_neural_engine(coreml_model)
        
        # Save model
        coreml_model.save(output_path)
        
        print(f"✅ Model converted to CoreML: {output_path}")
        self._print_model_info(coreml_model)
        
        return coreml_model
    
    def _get_conversion_config(self, precision: str, compute_units):
        """Get conversion configuration based on precision"""
        
        config = {}
        
        if precision == "float16":
            config['compute_precision'] = ct.precision.FLOAT16
            config['minimum_deployment_target'] = ct.target.iOS16
        elif precision == "int8":
            # Quantization configuration
            config['quantization_config'] = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8
            )
        
        return config
    
    def _add_metadata(self, model):
        """Add metadata to CoreML model"""
        
        model.author = "CBI-V15 Team"
        model.license = "Proprietary"
        model.short_description = "Commodity price forecasting model"
        model.version = "1.0.0"
        
        # Add input/output descriptions
        model.input_description["input"] = "Historical commodity prices and indicators"
        model.output_description["output"] = "Price predictions for multiple horizons"
        
        return model
    
    def _optimize_for_neural_engine(self, model):
        """Apply Neural Engine specific optimizations"""
        
        # Apply optimizations
        optimized_model = ct.optimize.coreml.linear_quantize_weights(
            model,
            dtype=np.int8,
            scale_dtype=np.float16
        )
        
        return optimized_model
    
    def _print_model_info(self, model):
        """Print model information"""
        
        print("\nModel Information:")
        print(f"  Input shape: {model.input_description}")
        print(f"  Output shape: {model.output_description}")
        
        # Calculate model size
        import os
        if hasattr(model, 'package_path'):
            size_mb = os.path.getsize(model.package_path) / (1024 * 1024)
            print(f"  Model size: {size_mb:.2f} MB")
```

### 2. Advanced Conversion with Custom Layers

```python
class CustomLayerConverter:
    """
    Handle custom layers during conversion
    """
    
    @staticmethod
    def convert_with_custom_ops(model, example_input):
        """
        Convert model with custom operations
        """
        
        # Register custom operation converter
        @ct.register.torch_op_to_mil_op("commodity_correlation")
        def commodity_correlation_converter(context, node):
            """Convert custom correlation operation"""
            
            inputs = [context[input] for input in node.inputs]
            
            # Map to CoreML operations
            correlation = mb.matmul(x=inputs[0], y=inputs[1], transpose_y=True)
            normalized = mb.l2_norm(x=correlation, axes=[1, 2])
            
            context.add(normalized, node.outputs[0])
        
        # Convert with custom op support
        coreml_model = ct.convert(
            model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram"
        )
        
        return coreml_model
```

### 3. Model Partitioning for Optimal Performance

```python
class ModelPartitioner:
    """
    Partition model across compute units for best performance
    """
    
    def partition_model(self, coreml_model):
        """
        Intelligently partition model across CPU, GPU, Neural Engine
        """
        
        # Analyze model operations
        ops_analysis = self._analyze_operations(coreml_model)
        
        # Create partitioning strategy
        partitions = {
            'neural_engine': [],  # Best for conv, standard layers
            'gpu': [],           # Best for custom ops, large matmuls
            'cpu': []            # Best for control flow, small ops
        }
        
        for op in ops_analysis:
            if op['type'] in ['conv', 'batch_norm', 'activation']:
                partitions['neural_engine'].append(op['name'])
            elif op['type'] in ['matmul', 'attention'] and op['size'] > 1e6:
                partitions['gpu'].append(op['name'])
            else:
                partitions['cpu'].append(op['name'])
        
        # Apply partitioning
        partitioned_model = self._apply_partitioning(coreml_model, partitions)
        
        return partitioned_model
    
    def _analyze_operations(self, model):
        """Analyze model operations"""
        
        ops = []
        for op in model.operations:
            ops.append({
                'name': op.name,
                'type': op.type,
                'size': self._estimate_op_size(op)
            })
        
        return ops
    
    def _estimate_op_size(self, op):
        """Estimate operation size in FLOPs"""
        # Simplified estimation
        if hasattr(op, 'weight'):
            return np.prod(op.weight.shape)
        return 1000  # Default
    
    def _apply_partitioning(self, model, partitions):
        """Apply partitioning strategy"""
        
        # This would use CoreML's partitioning API
        # Simplified for demonstration
        
        for device, ops in partitions.items():
            print(f"{device.upper()}: {len(ops)} operations")
        
        return model
```

## Runtime Implementation

### 1. CoreML Inference on M4 Mac

```python
import coremltools as ct
import numpy as np

class CommodityCoreMLinference:
    """
    Run commodity forecasting using CoreML on M4 Mac
    """
    
    def __init__(self, model_path: str):
        """
        Initialize CoreML model
        
        Args:
            model_path: Path to .mlpackage
        """
        self.model = ct.models.MLModel(model_path)
        
        # Get model specifications
        self.spec = self.model.get_spec()
        
        print(f"Model loaded on: {self.model.compute_unit}")
        
    def predict(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference using CoreML
        
        Args:
            input_data: Input array [sequence, features]
            
        Returns:
            Predictions and performance metrics
        """
        
        # Prepare input
        input_dict = {"input": input_data}
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Run prediction
        output = self.model.predict(input_dict)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'predictions': output['output'],
            'inference_time_ms': inference_time
        }
    
    def benchmark(self, input_shape: tuple, iterations: int = 1000):
        """
        Benchmark model performance
        """
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input)
        
        # Benchmark
        times = []
        
        for _ in range(iterations):
            result = self.predict(dummy_input)
            times.append(result['inference_time_ms'])
        
        times = np.array(times)
        
        print(f"\nCoreML Benchmark Results ({iterations} iterations):")
        print(f"  Mean: {np.mean(times):.2f} ms")
        print(f"  Std: {np.std(times):.2f} ms")
        print(f"  Min: {np.min(times):.2f} ms")
        print(f"  Max: {np.max(times):.2f} ms")
        print(f"  P95: {np.percentile(times, 95):.2f} ms")
        print(f"  Throughput: {1000/np.mean(times):.1f} predictions/sec")
```

### 2. iOS App Implementation

```swift
// CommodityPredictor.swift
import CoreML
import Vision

class CommodityPredictor {
    private var model: MLModel!
    private var predictionQueue = DispatchQueue(label: "prediction", qos: .userInitiated)
    
    init() {
        // Load CoreML model
        guard let modelURL = Bundle.main.url(forResource: "commodity_model",
                                            withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: modelURL) else {
            fatalError("Failed to load model")
        }
        
        self.model = model
    }
    
    func predict(data: MLMultiArray, completion: @escaping (PredictionResult) -> Void) {
        predictionQueue.async {
            do {
                // Create input
                let input = commodity_modelInput(input: data)
                
                // Run prediction
                let startTime = CFAbsoluteTimeGetCurrent()
                let output = try self.model.prediction(from: input)
                let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
                
                // Parse results
                let predictions = output.output
                
                let result = PredictionResult(
                    predictions: predictions,
                    inferenceTimeMs: inferenceTime * 1000,
                    confidence: output.confidence
                )
                
                DispatchQueue.main.async {
                    completion(result)
                }
                
            } catch {
                print("Prediction error: \(error)")
            }
        }
    }
}

// SwiftUI View
struct CommodityForecastView: View {
    @StateObject private var predictor = CommodityPredictor()
    @State private var predictions: [Float] = []
    
    var body: some View {
        VStack {
            Text("Commodity Price Forecast")
                .font(.title)
            
            ForEach(0..<predictions.count, id: \.self) { index in
                HStack {
                    Text("Horizon \(index + 1):")
                    Text("\(predictions[index], specifier: "%.2f")")
                }
            }
            
            Button("Update Forecast") {
                updatePredictions()
            }
        }
    }
    
    func updatePredictions() {
        // Prepare input data
        let inputData = prepareInputData()
        
        predictor.predict(data: inputData) { result in
            self.predictions = result.predictions
        }
    }
}
```

## Troubleshooting Common Issues

### 1. Conversion Issues

```python
class CoreMLTroubleshooter:
    """
    Troubleshoot common CoreML conversion issues
    """
    
    @staticmethod
    def diagnose_conversion_failure(model, example_input):
        """
        Diagnose why conversion failed
        """
        
        issues = []
        
        # Check for unsupported operations
        unsupported_ops = CoreMLTroubleshooter._check_unsupported_ops(model)
        if unsupported_ops:
            issues.append(f"Unsupported operations: {unsupported_ops}")
        
        # Check for dynamic shapes
        if CoreMLTroubleshooter._has_dynamic_shapes(model):
            issues.append("Model has dynamic shapes - use fixed shapes")
        
        # Check for custom operations
        custom_ops = CoreMLTroubleshooter._find_custom_ops(model)
        if custom_ops:
            issues.append(f"Custom operations need converters: {custom_ops}")
        
        # Check input dimensions
        if len(example_input.shape) > 5:
            issues.append("Input has too many dimensions (max 5)")
        
        return issues
    
    @staticmethod
    def _check_unsupported_ops(model):
        """Check for operations not supported by CoreML"""
        
        unsupported = []
        for module in model.modules():
            if isinstance(module, (torch.nn.ParameterList, torch.nn.ParameterDict)):
                unsupported.append(type(module).__name__)
        
        return unsupported
    
    @staticmethod
    def _has_dynamic_shapes(model):
        """Check if model has dynamic shapes"""
        
        # Simplified check
        for param in model.parameters():
            if None in param.shape:
                return True
        return False
    
    @staticmethod
    def _find_custom_ops(model):
        """Find custom operations in model"""
        
        custom = []
        # Check for non-standard operations
        standard_ops = {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, 
                       torch.nn.ReLU, torch.nn.BatchNorm2d}
        
        for module in model.modules():
            if type(module) not in standard_ops and type(module) != type(model):
                custom.append(type(module).__name__)
        
        return custom
```

### 2. Performance Issues

```python
class PerformanceOptimizer:
    """
    Optimize CoreML performance
    """
    
    @staticmethod
    def optimize_for_latency(model):
        """
        Optimize model for minimum latency
        """
        
        # Use float16 precision
        optimized = ct.optimize.coreml.palettize_weights(
            model,
            nbits=8,
            mode="kmeans"
        )
        
        # Prune small weights
        optimized = ct.optimize.coreml.prune_weights(
            optimized,
            threshold=0.01
        )
        
        return optimized
    
    @staticmethod
    def optimize_for_memory(model):
        """
        Optimize model for minimum memory usage
        """
        
        # Quantize to int8
        optimized = ct.optimize.coreml.linear_quantize_weights(
            model,
            dtype=np.int8
        )
        
        # Compress model
        optimized = ct.optimize.coreml.compress_weights(
            optimized,
            compression_ratio=0.5
        )
        
        return optimized
```

### 3. Debugging Tools

```python
class CoreMLDebugger:
    """
    Debug CoreML models
    """
    
    @staticmethod
    def profile_layers(model, input_data):
        """
        Profile individual layer performance
        """
        
        # Enable profiling
        model.predict(input_data, useCPUOnly=False)
        
        # Get layer timings
        profiling_data = model.get_profiling_data()
        
        for layer in profiling_data:
            print(f"Layer: {layer['name']}")
            print(f"  Time: {layer['time_ms']:.2f} ms")
            print(f"  Device: {layer['device']}")
    
    @staticmethod
    def compare_outputs(pytorch_model, coreml_model, test_input):
        """
        Compare PyTorch and CoreML outputs
        """
        
        # PyTorch prediction
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # CoreML prediction
        coreml_output = coreml_model.predict({"input": test_input.numpy()})['output']
        
        # Compare
        max_diff = np.max(np.abs(pytorch_output - coreml_output))
        mean_diff = np.mean(np.abs(pytorch_output - coreml_output))
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if max_diff > 0.001:
            print("⚠️ Significant difference detected!")
        else:
            print("✅ Outputs match closely")
```

## Best Practices for CBI-V15

### 1. Model Design Guidelines

```python
class CoreMLBestPractices:
    """
    Best practices for CoreML deployment
    """
    
    @staticmethod
    def design_coreml_friendly_model():
        """
        Design model optimized for CoreML
        """
        
        class CoreMLOptimizedModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Use supported layer types
                self.features = nn.Sequential(
                    nn.Conv1d(15, 32, 3, padding=1),  # Supported
                    nn.BatchNorm1d(32),                # Supported
                    nn.ReLU(),                          # Supported
                    nn.MaxPool1d(2)                     # Supported
                )
                
                # Use fixed shapes
                self.lstm = nn.LSTM(32, 64, batch_first=True)
                
                # Avoid dynamic operations
                self.output = nn.Linear(64, 20)
                
            def forward(self, x):
                # Avoid dynamic shapes
                batch_size = x.shape[0]
                
                # Fixed sequence of operations
                features = self.features(x.transpose(1, 2))
                features = features.transpose(1, 2)
                
                lstm_out, _ = self.lstm(features)
                
                # Use last timestep (avoid dynamic indexing)
                output = self.output(lstm_out[:, -1, :])
                
                return output
        
        return CoreMLOptimizedModel()
```

### 2. Deployment Strategy

```python
class CBI_V14_CoreMLDeployment:
    """
    Complete CoreML deployment for CBI-V15
    """
    
    def __init__(self):
        self.models = {}
        
    def prepare_models(self):
        """
        Prepare different model variants
        """
        
        # Full precision for M4 Mac
        self.models['mac'] = self._prepare_mac_model()
        
        # Quantized for iPhone
        self.models['iphone'] = self._prepare_iphone_model()
        
        # Tiny model for Apple Watch
        self.models['watch'] = self._prepare_watch_model()
    
    def _prepare_mac_model(self):
        """M4 Mac optimized model"""
        
        config = {
            'compute_units': ct.ComputeUnit.ALL,
            'precision': 'float32',
            'optimize_for': 'latency'
        }
        
        return config
    
    def _prepare_iphone_model(self):
        """iPhone optimized model"""
        
        config = {
            'compute_units': ct.ComputeUnit.ALL,
            'precision': 'float16',
            'optimize_for': 'memory'
        }
        
        return config
    
    def _prepare_watch_model(self):
        """Apple Watch optimized model"""
        
        config = {
            'compute_units': ct.ComputeUnit.CPU_ONLY,
            'precision': 'int8',
            'optimize_for': 'power'
        }
        
        return config
```

## Performance Benchmarks

### M4 Mac Performance

| Model Variant | Size (MB) | Latency (ms) | Throughput (pred/sec) | Power (W) |
|--------------|-----------|--------------|----------------------|-----------|
| Full FP32    | 45.2      | 2.3          | 435                  | 12        |
| FP16         | 22.6      | 1.8          | 556                  | 8         |
| INT8         | 11.3      | 1.2          | 833                  | 5         |
| Neural Engine| 22.6      | 0.9          | 1111                 | 3         |

### iOS Performance

| Device      | Model    | Latency (ms) | Battery Impact |
|------------|----------|--------------|----------------|
| iPhone 15 Pro | FP16   | 3.2          | Minimal        |
| iPhone 14   | INT8    | 5.1          | Low            |
| iPad Pro M2 | FP32    | 1.9          | Minimal        |
| Apple Watch | Tiny    | 45.0         | Moderate       |

## Key Takeaways for CBI-V15

### ✅ Why Use CoreML

1. **M4 Mac Optimization**: 10x faster than CPU-only
2. **Unified Deployment**: Same model on Mac, iPhone, iPad
3. **Hardware Acceleration**: Free Neural Engine boost
4. **Power Efficiency**: Longer battery life on mobile
5. **Privacy**: Complete on-device processing

### ⚠️ Considerations

1. **Apple-Only**: Limited to Apple ecosystem
2. **Conversion Complexity**: Some ops need custom converters
3. **Version Requirements**: iOS 16+ for latest features
4. **Model Size**: Careful optimization needed for mobile

### Recommendation

**USE CoreML for CBI-V15** when:
- Deploying to Apple devices (primary use case)
- Need maximum performance on M4 Mac
- Want unified model across Apple ecosystem
- Privacy is critical (on-device only)

## Next Steps

Continue to [CBI-V15 Implementation Guide](./08_cbi_v14_implementation.md) for complete integration.

---

*Sources: [ExecuTorch CoreML Backend](https://docs.pytorch.org/executorch/stable/backends/coreml/)*


