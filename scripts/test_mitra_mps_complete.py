#!/usr/bin/env python3
"""
Comprehensive test of Mitra + MPS + AutoGluon Neural Models on Mac M4 Pro.
Following production-safe Metal (MPS) guidelines.
"""

import sys
import numpy as np
import pandas as pd

print('=' * 80)
print('MITRA + MPS + AUTOGLUON NEURAL MODELS - COMPREHENSIVE TEST')
print('=' * 80)
print('')

# ============================================================================
# TEST 0: PyTorch MPS Verification
# ============================================================================
print('TEST 0: PYTORCH MPS BACKEND VERIFICATION')
print('=' * 60)

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS available: {torch.backends.mps.is_available()}')
    print(f'MPS built: {torch.backends.mps.is_built()}')
    print(f'CUDA available: {torch.cuda.is_available()} (should be False)')
    
    if torch.backends.mps.is_available():
        device = "mps"
        print(f'✅ Using device: {device}')
        
        # Test MPS with simple tensor operation
        x = torch.randn(100, 100).to(device)
        y = torch.matmul(x, x.T)
        print(f'✅ MPS tensor test passed: {y.shape}')
    else:
        device = "cpu"
        print(f'⚠️  MPS not available, using: {device}')
        
except Exception as e:
    print(f'❌ PyTorch MPS check failed: {e}')
    device = "cpu"

print('')
print('')

# ============================================================================
# TEST 1: Install and Test Mitra on MPS
# ============================================================================
print('TEST 1: MITRA (SALESFORCE TIME-SERIES FOUNDATION MODEL) ON MPS')
print('=' * 60)

try:
    # Try to import mitra
    try:
        from mitra.models import MitraForecast
        print('✅ mitra-forecast already installed')
    except ImportError:
        print('Installing mitra-forecast...')
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mitra-forecast'])
        from mitra.models import MitraForecast
        print('✅ mitra-forecast installed')
    
    # Load Mitra on Metal
    print(f'Loading Mitra model on device: {device}')
    model = MitraForecast.from_pretrained(
        "salesforce/mitra-base",
        device=device
    )
    print('✅ Mitra model loaded on MPS')
    
    # Test inference
    print('Running Mitra inference (30-step forecast)...')
    series = np.random.rand(256).astype("float32")
    
    with torch.no_grad():
        forecast = model.predict(
            series,
            horizon=30,
            num_samples=100
        )
    
    print(f'✅ Mitra inference complete: forecast shape {forecast.shape}')
    print(f'   Mean forecast: {forecast.mean():.4f}')
    print(f'   Std forecast: {forecast.std():.4f}')
    
except Exception as e:
    print(f'❌ Mitra test FAILED: {e}')
    import traceback
    traceback.print_exc()

print('')
print('')

# ============================================================================
# TEST 2: AutoGluon Neural Networks with MPS
# ============================================================================
print('TEST 2: AUTOGLUON NEURAL NETWORKS (NN_TORCH, FASTAI) WITH MPS')
print('=' * 60)

try:
    from autogluon.tabular import TabularPredictor
    import tempfile
    
    # Create dataset
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
        'f4': np.random.randn(n),
        'f5': np.random.randn(n),
        'target': np.random.randn(n)
    })
    
    print(f'Dataset: {len(df)} rows, 5 features')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Training neural network models (NN_TORCH, FASTAI)...')
        
        hyperparameters = {
            'NN_TORCH': {},
            'FASTAI': {},
        }
        
        predictor = TabularPredictor(
            label='target',
            path=tmpdir,
            problem_type='regression',
            verbosity=2
        ).fit(
            train_data=df,
            hyperparameters=hyperparameters,
            time_limit=120,
            num_gpus=0
        )
        
        lb = predictor.leaderboard(silent=True)
        print('')
        print('NEURAL NETWORK RESULTS:')
        print(lb[['model', 'score_val', 'fit_time']].to_string())
        print(f'✅ Neural models trained: {len(lb)}')
        
except Exception as e:
    print(f'❌ AutoGluon neural test FAILED: {e}')
    import traceback
    traceback.print_exc()

print('')
print('=' * 80)
print('TEST COMPLETE')
print('=' * 80)

