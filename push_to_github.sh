#!/bin/bash
# Push CBI-V15 to GitHub
# Run this AFTER creating the repository on GitHub

set -e

REPO_URL="https://github.com/zincdigital/CBI-V15.git"
SSH_URL="git@github.com:zincdigital/CBI-V15.git"

echo "🚀 Pushing CBI-V15 to GitHub..."
echo ""

# Check if repository exists
echo "Checking if repository exists..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://github.com/zincdigital/CBI-V15 || echo "000")

if [ "$HTTP_CODE" = "404" ]; then
    echo "❌ Repository does not exist on GitHub yet!"
    echo ""
    echo "Please create it first:"
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: CBI-V15"
    echo "3. Owner: zincdigital"
    echo "4. DO NOT initialize with README"
    echo "5. Click 'Create repository'"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Repository exists on GitHub"
echo ""

# Try SSH first, fallback to HTTPS
if git ls-remote "$SSH_URL" &>/dev/null; then
    echo "Using SSH..."
    git remote set-url origin "$SSH_URL"
elif git ls-remote "$REPO_URL" &>/dev/null; then
    echo "Using HTTPS..."
    git remote set-url origin "$REPO_URL"
else
    echo "⚠️  Cannot access repository. Check your GitHub access."
    exit 1
fi

# Push
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "Next steps:"
echo "1. Go to Google Cloud Console → Dataform"
echo "2. Click 'Connect Repository'"
echo "3. Select: zincdigital/CBI-V15"
echo "4. Branch: main"
echo "5. Root Directory: dataform/"
echo "6. Click 'Connect'"
