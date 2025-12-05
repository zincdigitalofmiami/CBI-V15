#!/bin/bash
# Create Python virtual environment for CBI-V15
# This fixes the Python extension interpreter issue

set -e

cd "$(dirname "$0")/../.."

echo "🐍 Creating Python Virtual Environment"
echo "======================================"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "⚠️  venv directory already exists"
    read -p "Remove existing venv and recreate? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf venv
    else
        echo "Keeping existing venv. Exiting."
        exit 0
    fi
fi

# Create venv
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and upgrade pip
echo "Activating venv and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, skipping package installation"
fi

echo ""
echo "✅ Virtual environment created successfully!"
echo ""
echo "To activate manually:"
echo "  source venv/bin/activate"
echo ""
echo "Python extension should now detect: $(pwd)/venv/bin/python"
echo ""






