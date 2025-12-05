#!/bin/bash
# Verify Dataform GitHub connection status

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"

echo "🔍 Checking Dataform Connection Status"
echo "======================================"
echo ""

# Check if SSH key exists
if [ -f ~/.ssh/dataform_github_ed25519.pub ]; then
    echo "✅ SSH key exists locally"
else
    echo "❌ SSH key not found"
    exit 1
fi

# Check if secret exists
if gcloud secrets describe dataform-github-ssh-key --project="$PROJECT_ID" &> /dev/null; then
    echo "✅ Secret stored in Secret Manager"
else
    echo "❌ Secret not found in Secret Manager"
    exit 1
fi

# Check SSH key on GitHub (if SSH is configured)
echo ""
echo "🔗 Testing GitHub SSH Connection:"
if ssh -T git@github.com -o StrictHostKeyChecking=no -o ConnectTimeout=5 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH key is on GitHub and working"
elif ssh -T git@github.com -o StrictHostKeyChecking=no -o ConnectTimeout=5 2>&1 | grep -q "Permission denied"; then
    echo "⚠️  SSH key not yet added to GitHub"
    echo "   Add at: https://github.com/settings/ssh/new"
else
    echo "⚠️  Could not verify GitHub connection"
    echo "   Add SSH key at: https://github.com/settings/ssh/new"
fi

echo ""
echo "📋 Next Steps:"
echo "1. Add SSH key to GitHub (if not done):"
echo "   https://github.com/settings/ssh/new"
echo ""
echo "2. Connect Dataform in UI:"
echo "   https://console.cloud.google.com/dataform?project=cbi-v15"
echo "   - Create repository: CBI-V15"
echo "   - Settings → Connect to GitHub"
echo "   - SSH URL: git@github.com:zincdigital/CBI-V15.git"
echo "   - Secret: dataform-github-ssh-key"
echo "   - Root Directory: dataform/"

