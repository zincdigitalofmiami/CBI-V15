#!/bin/bash
# Fix Dataform SSH secret - use RSA key in PEM format (plain text)
# Dataform requires PEM format RSA key, stored as plain text
# Usage: ./fix_dataform_ssh_pem_format.sh

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
RSA_KEY_PATH="$HOME/.ssh/dataform_github_rsa"
RSA_PUB_KEY_PATH="$HOME/.ssh/dataform_github_rsa.pub"

echo "🔧 Fixing Dataform SSH Secret (RSA PEM Format)"
echo "=============================================="
echo ""

# Check if RSA key exists, create if not
if [ ! -f "$RSA_KEY_PATH" ]; then
    echo "📝 Generating RSA SSH key pair..."
    ssh-keygen -t rsa -b 4096 -f "$RSA_KEY_PATH" \
        -C "dataform-${PROJECT_ID}@gcp" \
        -N ""
    echo "✅ RSA key generated"
else
    echo "✅ RSA key already exists"
fi

# Verify it's PEM format
if ! head -1 "$RSA_KEY_PATH" | grep -q "BEGIN.*PRIVATE KEY"; then
    echo "❌ Invalid SSH private key format"
    exit 1
fi

echo ""
echo "📋 Public Key (add this to GitHub):"
echo "===================================="
cat "$RSA_PUB_KEY_PATH"
echo ""
echo "===================================="
echo ""

read -p "Press Enter after adding the public key to GitHub..."

# Store as plain text (not base64)
echo ""
echo "💾 Storing RSA private key in Secret Manager (plain text)..."
cat "$RSA_KEY_PATH" | gcloud secrets versions add "$SECRET_NAME" \
    --data-file=- \
    --project="$PROJECT_ID"

echo ""
echo "✅ Secret updated with RSA PEM format key!"
echo ""
echo "📋 Verification:"
echo "  gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID | head -1"
echo ""
echo "🔗 Next: Test Dataform connection in UI"
echo "  https://console.cloud.google.com/dataform?project=$PROJECT_ID"

