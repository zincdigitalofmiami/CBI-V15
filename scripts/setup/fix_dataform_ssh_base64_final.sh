#!/bin/bash
# Fix Dataform SSH secret - store RSA key as BASE64 ENCODED
# Dataform requires base64 encoded private key
# Usage: ./fix_dataform_ssh_base64_final.sh

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
RSA_KEY_PATH="$HOME/.ssh/dataform_github_rsa"

echo "🔧 Fixing Dataform SSH Secret (Base64 Encoded RSA Key)"
echo "======================================================"
echo ""

# Check if RSA key exists
if [ ! -f "$RSA_KEY_PATH" ]; then
    echo "❌ RSA key not found: $RSA_KEY_PATH"
    exit 1
fi

echo "📝 Reading RSA private key..."
PRIVATE_KEY=$(cat "$RSA_KEY_PATH")

# Verify it's a valid SSH private key
if ! echo "$PRIVATE_KEY" | grep -q "BEGIN.*PRIVATE KEY"; then
    echo "❌ Invalid SSH private key format"
    exit 1
fi

echo "✅ RSA key format valid"
echo ""

# Encode to base64 (single line, no newlines)
echo "🔐 Encoding RSA private key to base64..."
BASE64_KEY=$(cat "$RSA_KEY_PATH" | python3 -c "import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))")

echo "✅ Base64 encoded (length: ${#BASE64_KEY} characters)"
echo ""

# Store base64 encoded key
echo "💾 Storing base64 encoded RSA key in Secret Manager..."
echo -n "$BASE64_KEY" | gcloud secrets versions add "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"

echo ""
echo "✅ Secret updated with base64 encoded RSA key!"
echo ""
echo "📋 Verification:"
echo "  # Check it's base64:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | head -1 | wc -c"
echo ""
echo "  # Decode and verify:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | base64 -d | head -1"
echo ""
echo "🔗 Next: Test Dataform connection in UI"
echo "  https://console.cloud.google.com/dataform?project=$PROJECT_ID"

