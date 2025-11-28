#!/bin/bash
# Add RSA public key to GitHub
# Usage: ./add_rsa_key_to_github.sh

set -e

RSA_PUB_KEY_PATH="$HOME/.ssh/dataform_github_rsa.pub"
KEY_TITLE="Dataform CBI-V15 RSA"

echo "🔑 Adding RSA Public Key to GitHub"
echo "=================================="
echo ""

# Check if public key exists
if [ ! -f "$RSA_PUB_KEY_PATH" ]; then
    echo "❌ Public key not found: $RSA_PUB_KEY_PATH"
    exit 1
fi

echo "📋 Your RSA Public Key:"
echo "======================="
cat "$RSA_PUB_KEY_PATH"
echo ""
echo "======================="
echo ""

# Try GitHub CLI first
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found"
    
    if gh auth status &> /dev/null; then
        echo "✅ GitHub CLI authenticated"
        echo ""
        echo "Adding SSH key via GitHub CLI..."
        
        if gh ssh-key add "$RSA_PUB_KEY_PATH" --title "$KEY_TITLE" 2>&1; then
            echo ""
            echo "✅ SSH key added to GitHub successfully!"
            echo ""
            echo "Verifying connection..."
            if ssh -T git@github.com -i ~/.ssh/dataform_github_rsa -o StrictHostKeyChecking=no 2>&1 | grep -q "successfully authenticated\|Hi"; then
                echo "✅ SSH connection verified!"
            else
                echo "⚠️  SSH connection test inconclusive"
            fi
            exit 0
        else
            echo "⚠️  Failed to add via CLI, use manual method below"
        fi
    else
        echo "⚠️  GitHub CLI not authenticated"
        echo "   Run: gh auth login"
        echo ""
    fi
else
    echo "⚠️  GitHub CLI not found"
    echo ""
fi

# Manual instructions
echo "📋 Manual Method:"
echo "=================="
echo ""
echo "1. Copy your public SSH key (shown above)"
echo ""
echo "2. Go to GitHub:"
echo "   https://github.com/settings/ssh/new"
echo ""
echo "3. Fill in the form:"
echo "   - Title: $KEY_TITLE"
echo "   - Key: Paste the key above"
echo "   - Key type: Authentication Key (default)"
echo ""
echo "4. Click 'Add SSH key'"
echo ""
echo "5. Verify connection:"
echo "   ssh -T git@github.com -i ~/.ssh/dataform_github_rsa"
echo ""

