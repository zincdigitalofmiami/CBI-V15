#!/bin/bash
# Train all 8 bucket specialists with AutoGluon extreme_quality preset
# Expected runtime: 2-4 hours on Mac M4 (depending on time_limit setting)

set -e

echo "========================================================================"
echo "      L0 BUCKET SPECIALIST TRAINING - EXTREME QUALITY"
echo "========================================================================"
echo "Config:        extreme_quality preset"
echo "Buckets:       8 (Crush, China, FX, Fed, Tariff, Biofuel, Energy, Vol)"
echo "Horizons:      4 (1w, 1m, 3m, 6m)"
echo "Total Models:  32"
echo "Time per model: 10 minutes"
echo "Expected time:  2-4 hours"
echo "========================================================================"
echo ""

# Check if AutoGluon is installed
if ! python -c "import autogluon.tabular" 2>/dev/null; then
    echo "❌ AutoGluon not installed. Please install:"
    echo "   pip install autogluon.tabular"
    exit 1
fi

# Check if config exists
if [ ! -f "config/bucket_feature_selectors.yaml" ]; then
    echo "❌ Config not found. Generating from database..."
    python scripts/generate_config_from_db.py
fi

echo "🚀 Starting training..."
echo ""

# Run the python script
# time_limit=600 (10 min) per model * 32 models = ~5.3 hours max
# AutoGluon will use early stopping if it converges faster
python src/training/autogluon/bucket_specialist.py \
    --config config/bucket_feature_selectors.yaml \
    --preset extreme_quality \
    --time_limit 600 \
    --save-oof

echo ""
echo "========================================================================"
echo "✅ Training complete!"
echo "========================================================================"
echo "OOF predictions saved to 'training.bucket_predictions'"
echo "Model artifacts saved to 'models/bucket_specialists/'"
echo ""
echo "Next steps:"
echo "  1. Verify OOF predictions:"
echo "     python -c \"import duckdb; con=duckdb.connect('motherduck:cbi_v15'); print(con.execute('SELECT bucket, horizon_code, COUNT(*) FROM training.bucket_predictions GROUP BY 1,2').df())\""
echo ""
echo "  2. Train meta-model (L1) using these OOF predictions"
echo "========================================================================"



