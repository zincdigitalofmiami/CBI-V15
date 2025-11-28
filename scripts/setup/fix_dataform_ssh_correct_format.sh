#!/bin/bash
# Fix Dataform SSH secret - CORRECT format: base64 encoded, single line, no newlines
# Dataform requires: Pure base64 string (A-Z, a-z, 0-9, +, /, =) with NO dashes or newlines
# Usage: ./fix_dataform_ssh_correct_format.sh

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
RSA_KEY_PATH="$HOME/.ssh/dataform_github_rsa"

echo "🔧 Fixing Dataform SSH Secret (Correct Format)"
echo "=============================================="
echo ""
echo "Requirements:"
echo "  - Base64 encoded"
echo "  - Single line (no newlines)"
echo "  - Pure base64 characters only (A-Z, a-z, 0-9, +, /, =)"
echo "  - NO dashes (-) or other special characters"
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

# Remove all newlines and encode to base64
echo "🔐 Encoding to base64 (removing newlines, single line)..."
BASE64_KEY=$(cat "$RSA_KEY_PATH" | tr -d '\n' | python3 -c "import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))")

# Verify it's pure base64
if ! echo "$BASE64_KEY" | grep -qE "^[A-Za-z0-9+/=]+$"; then
    echo "❌ Base64 encoding contains invalid characters"
    exit 1
fi

echo "✅ Base64 encoded (length: ${#BASE64_KEY} characters)"
echo "✅ Pure base64 format (no dashes/newlines)"
echo ""

# Store base64 encoded key
echo "💾 Storing base64 encoded key in Secret Manager..."
echo -n "$BASE64_KEY" | gcloud secrets versions add "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"

echo ""
echo "✅ Secret updated with correctly formatted base64 key!"
echo ""
echo "📋 Verification:"
echo "  # Check format:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | head -c 100"
echo ""
echo "  # Verify pure base64:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | grep -qE '^[A-Za-z0-9+/=]+$' && echo 'Pure base64'"
echo ""
echo "  # Decode and verify:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | base64 -d | head -1"
echo ""
echo "🔗 Next: Test Dataform connection in UI"
echo "  https://console.cloud.google.com/dataform?project=$PROJECT_ID"

