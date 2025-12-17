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

echo "3️⃣  Checking LightGBM installation..."
echo ""

# Check if LightGBM is already installed
if python3 -c "import lightgbm" 2>/dev/null; then
    echo "   ✅ LightGBM already installed"
    echo "   ⚠️  If you experience segfaults, rebuild with:"
    echo "      pip install lightgbm --force-reinstall --no-binary :all: --config-settings=..."
else
    echo "   Installing LightGBM (pre-built wheel)..."
    pip install lightgbm
fi

echo ""
echo "4️⃣  Installing AutoGluon 1.4..."
echo ""

# Install AutoGluon with pre-built wheels (avoids Rust compiler requirement)
pip install --upgrade pip
pip install autogluon.tabular[all]>=1.4.0 --no-build-isolation
pip install autogluon.timeseries[all]>=1.4.0 --no-build-isolation
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
print("   Available models: LightGBM, CatBoost, XGBoost, FastAI, PyTorch, TabPFNv2, Mitra, TabICL")
print("   Presets: 'extreme_quality', 'best', 'high', 'good', 'medium' (all CPU-compatible)")
print("\n⚠️  Note: 'extreme_quality' works on Mac M4 CPU but will be slower than GPU")
print("   Foundation models (TabPFNv2, Mitra, TabICL) included in extreme_quality run on CPU")
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
















