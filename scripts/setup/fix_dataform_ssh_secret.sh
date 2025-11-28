#!/bin/bash
# Fix Dataform SSH secret - store private key correctly
# Usage: ./fix_dataform_ssh_secret.sh

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
PRIVATE_KEY_PATH="$HOME/.ssh/dataform_github_ed25519"

echo "🔧 Fixing Dataform SSH Secret"
echo "============================="
echo ""

# Check if private key exists
if [ ! -f "$PRIVATE_KEY_PATH" ]; then
    echo "❌ Private key not found: $PRIVATE_KEY_PATH"
    exit 1
fi

echo "📝 Reading private key..."
PRIVATE_KEY=$(cat "$PRIVATE_KEY_PATH")

# Verify it's a valid SSH private key
if ! echo "$PRIVATE_KEY" | grep -q "BEGIN.*PRIVATE KEY"; then
    echo "❌ Invalid SSH private key format"
    exit 1
fi

echo "✅ Private key format valid"
echo ""

# Store as plain text (not base64 encoded)
# Dataform expects the raw private key
echo "💾 Storing private key in Secret Manager..."
echo -n "$PRIVATE_KEY" | gcloud secrets versions add "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"

echo ""
echo "✅ Secret updated!"
echo ""
echo "📋 Verification:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | head -1"
echo ""
echo "🔗 Next: Test Dataform connection in UI"
echo "  https://console.cloud.google.com/dataform?project=$PROJECT_ID"

