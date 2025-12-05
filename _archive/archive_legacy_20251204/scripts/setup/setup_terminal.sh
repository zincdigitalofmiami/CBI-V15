#!/bin/bash
# Setup Terminal Configuration for CBI-V15
# Adds CBI-V15 configuration to ~/.zshrc

set -e

CBI_V15_ROOT="/Volumes/Satechi Hub/CBI-V15"
ZSHRC_FILE="${HOME}/.zshrc"
CONFIG_FILE="${CBI_V15_ROOT}/.cbi-v15.zsh"
CONFIG_LINE="source ${CONFIG_FILE}"

echo "🔧 Setting up CBI-V15 Terminal Configuration"
echo "============================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check if already configured
if grep -q "$CONFIG_LINE" "$ZSHRC_FILE" 2>/dev/null; then
    echo "✅ CBI-V15 configuration already exists in ~/.zshrc"
    echo ""
    echo "To reload configuration, run:"
    echo "  source ~/.zshrc"
    echo ""
    echo "Or manually source:"
    echo "  source ${CONFIG_FILE}"
    exit 0
fi

# Backup .zshrc
echo "📋 Creating backup of ~/.zshrc..."
cp "$ZSHRC_FILE" "${ZSHRC_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
echo "✅ Backup created"

# Add configuration to .zshrc
echo ""
echo "➕ Adding CBI-V15 configuration to ~/.zshrc..."
echo "" >> "$ZSHRC_FILE"
echo "# CBI-V15 Terminal Configuration" >> "$ZSHRC_FILE"
echo "$CONFIG_LINE" >> "$ZSHRC_FILE"
echo "✅ Configuration added"

echo ""
echo "============================================="
echo "✅ Terminal Configuration Complete!"
echo "============================================="
echo ""
echo "To activate immediately, run:"
echo "  source ~/.zshrc"
echo ""
echo "Or manually source:"
echo "  source ${CONFIG_FILE}"
echo ""
echo "Available commands:"
echo "  cbi-status       - Show project status"
echo "  cbi-set-project  - Set GCP project"
echo "  cbi-datasets     - List BigQuery datasets"
echo "  cbi-quick-setup  - Run quick setup guide"
echo "  cbi              - Navigate to project root"
echo "  # cbidf (deprecated) - old Dataform directory helper"
echo ""
echo "Type 'cbi-status' to verify setup"





