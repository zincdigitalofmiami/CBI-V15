#!/bin/bash
# Push CBI-V15 to GitHub
# Run this after creating the repository on GitHub

set -e

echo "🚀 Pushing CBI-V15 to GitHub..."

# Check if remote already exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "✅ Remote 'origin' already configured"
    git remote -v
else
    echo "📝 Adding remote 'origin'..."
    git remote add origin https://github.com/zincdigitalofmiami/CBI-V15.git
fi

# Ensure we're on main branch
git branch -M main

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Success! Repository pushed to GitHub"
echo "🌐 View at: https://github.com/zincdigitalofmiami/CBI-V15"

