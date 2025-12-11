#!/bin/bash
# Install AutoGluon 1.4 on Mac M4 with libomp fix
# CRITICAL: This script prevents LightGBM segfaults on Apple Silicon

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "AutoGluon 1.4 Installation for Mac M4"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Install from https://brew.sh"
    exit 1
fi

echo "1️⃣  Installing libomp (OpenMP) via Homebrew..."
echo "   This is CRITICAL for LightGBM on Mac M4"
echo ""

brew install libomp

echo ""
echo "2️⃣  Setting up environment variables..."
echo ""

# Get libomp paths
LIBOMP_PREFIX=$(brew --prefix libomp)
export LDFLAGS="-L${LIBOMP_PREFIX}/lib"
export CPPFLAGS="-I${LIBOMP_PREFIX}/include"
export CFLAGS="-I${LIBOMP_PREFIX}/include"
export CXXFLAGS="-I${LIBOMP_PREFIX}/include"

echo "   LIBOMP_PREFIX: $LIBOMP_PREFIX"
echo "   LDFLAGS: $LDFLAGS"
echo "   CPPFLAGS: $CPPFLAGS"
echo ""

echo "3️⃣  Installing LightGBM from source (with libomp)..."
echo "   This prevents segfaults on Mac M4"
echo ""

# Build LightGBM from source with explicit OpenMP flags
pip install lightgbm --no-binary :all: \
  --config-settings=cmake.define.OpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  --config-settings=cmake.define.OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" \
  --config-settings=cmake.define.OpenMP_libomp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"

echo ""
echo "4️⃣  Installing AutoGluon 1.4..."
echo ""

# Install AutoGluon (will use pre-built LightGBM)
pip install autogluon.tabular[all]>=1.4.0
pip install autogluon.timeseries[all]>=1.4.0
pip install autogluon.core>=1.4.0

echo ""
echo "5️⃣  Verifying installation..."
echo ""

# Test imports
python3 << 'EOF'
import sys

print("Testing AutoGluon imports...")

try:
    from autogluon.tabular import TabularPredictor
    print("✅ autogluon.tabular imported successfully")
except ImportError as e:
    print(f"❌ Failed to import autogluon.tabular: {e}")
    sys.exit(1)

try:
    from autogluon.timeseries import TimeSeriesPredictor
    print("✅ autogluon.timeseries imported successfully")
except ImportError as e:
    print(f"❌ Failed to import autogluon.timeseries: {e}")
    sys.exit(1)

try:
    import lightgbm
    print(f"✅ LightGBM {lightgbm.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LightGBM: {e}")
    sys.exit(1)

try:
    import catboost
    print(f"✅ CatBoost {catboost.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CatBoost: {e}")
    sys.exit(1)

try:
    import xgboost
    print(f"✅ XGBoost {xgboost.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Failed to import XGBoost: {e}")
    sys.exit(1)

print("\n✨ AutoGluon 1.4 ready for Mac M4!")
print("   Available models: LightGBM, CatBoost, XGBoost, FastAI, PyTorch")
print("   Presets: 'best', 'high', 'good', 'medium' (CPU-compatible)")
print("\n⚠️  Note: 'extreme' preset requires GPU (not compatible with Mac M4)")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ AutoGluon 1.4 Installation Complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📋 Next Steps:"
echo "   1. Run: python scripts/sync_motherduck_to_local.py"
echo "   2. Test: python -c 'from autogluon.tabular import TabularPredictor; print(\"Ready!\")'"
echo "   3. Start Phase 2: Create TabularPredictor wrapper"
echo ""



