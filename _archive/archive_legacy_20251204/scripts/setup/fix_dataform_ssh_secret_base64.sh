#!/bin/bash
# Fix Dataform SSH secret - store as base64 encoded
# Dataform expects base64 encoded private key
# Usage: ./fix_dataform_ssh_secret_base64.sh

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
PRIVATE_KEY_PATH="$HOME/.ssh/dataform_github_ed25519"

echo "🔧 Fixing Dataform SSH Secret (Base64 Encoded)"
echo "=============================================="
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

# Encode to base64 (single line, no newlines)
echo "🔐 Encoding private key to base64..."
BASE64_KEY=$(cat "$PRIVATE_KEY_PATH" | python3 -c "import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))")

echo "✅ Base64 encoded (length: ${#BASE64_KEY} characters)"
echo ""

# Store base64 encoded key
echo "💾 Storing base64 encoded key in Secret Manager..."
echo -n "$BASE64_KEY" | gcloud secrets versions add "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"

echo ""
echo "✅ Secret updated with base64 encoded key!"
echo ""
echo "📋 Verification:"
echo "  # Decode and verify:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | base64 -d | head -1"
echo ""
echo "🔗 Next: Test Dataform connection in UI"
echo "  https://console.cloud.google.com/dataform?project=$PROJECT_ID"

