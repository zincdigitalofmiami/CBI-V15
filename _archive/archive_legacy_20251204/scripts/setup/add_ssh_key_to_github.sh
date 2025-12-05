#!/bin/bash
# Add SSH key to GitHub using GitHub CLI or provide instructions

set -e

KEY_PATH="$HOME/.ssh/dataform_github_ed25519.pub"
KEY_TITLE="Dataform CBI-V15"

echo "🔑 Adding SSH Key to GitHub"
echo "=========================="
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found"
    
    # Check if authenticated
    if gh auth status &> /dev/null; then
        echo "✅ GitHub CLI authenticated"
        echo ""
        echo "Adding SSH key..."
        
        if gh ssh-key add "$KEY_PATH" --title "$KEY_TITLE" 2>&1; then
            echo ""
            echo "✅ SSH key added to GitHub successfully!"
            echo ""
            echo "You can now connect Dataform in the UI."
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
echo "1. Copy your public SSH key:"
echo ""
cat "$KEY_PATH"
echo ""
echo ""
echo "2. Go to GitHub:"
echo "   https://github.com/settings/ssh/new"
echo ""
echo "3. Paste the key above"
echo "4. Title: $KEY_TITLE"
echo "5. Click 'Add SSH key'"
echo ""
echo "After adding, you can connect Dataform in the UI."

