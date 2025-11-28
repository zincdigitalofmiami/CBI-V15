#!/bin/bash
# Connect Dataform repository to GitHub via API
# Note: Root directory must be set in workspace compilation override

set -e

PROJECT_ID="cbi-v15"
REGION="us-central1"
REPO_ID="CBI-V15"
GITHUB_URL="git@github.com:zincdigital/CBI-V15.git"
SECRET_NAME="dataform-github-ssh-key"
BRANCH="main"

echo "🔗 Connecting Dataform Repository to GitHub via API"
echo "===================================================="
echo ""

# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Check if repository exists
echo "1. Checking repository..."
REPO_EXISTS=$(curl -s -X GET \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_ID}" \
    2>&1 | grep -q "name" && echo "yes" || echo "no")

if [ "$REPO_EXISTS" = "no" ]; then
    echo "   Creating repository..."
    curl -s -X POST \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${REPO_ID}\"}" \
        "https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories?repositoryId=${REPO_ID}" \
        > /dev/null
    echo "   ✅ Repository created"
else
    echo "   ✅ Repository exists"
fi

# Get GitHub host public key
echo ""
echo "2. Getting GitHub host public key..."
GITHUB_HOST_KEY=$(ssh-keyscan github.com 2>/dev/null | grep "ssh-ed25519" | head -1)
if [ -z "$GITHUB_HOST_KEY" ]; then
    echo "   ⚠️  Could not get GitHub host key, using known key"
    GITHUB_HOST_KEY="github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
fi
echo "   ✅ Host key obtained"

# Connect to GitHub
echo ""
echo "3. Connecting to GitHub..."
RESPONSE=$(curl -s -X PATCH \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"gitRemoteSettings\": {
            \"url\": \"${GITHUB_URL}\",
            \"sshAuthenticationConfig\": {
                \"userPrivateKeySecretVersion\": \"projects/${PROJECT_ID}/secrets/${SECRET_NAME}/versions/latest\",
                \"hostPublicKey\": \"${GITHUB_HOST_KEY}\"
            },
            \"defaultBranch\": \"${BRANCH}\"
        }
    }" \
    "https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_ID}?updateMask=gitRemoteSettings")

if echo "$RESPONSE" | grep -q "error"; then
    echo "   ❌ Error:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
else
    echo "   ✅ Connected to GitHub"
fi

# Set workspace compilation override (for root directory)
echo ""
echo "4. Setting workspace compilation override (root directory)..."
WORKSPACE="main"
curl -s -X PATCH \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{
        \"compilationOverrides\": {
            \"defaultDatabase\": \"${PROJECT_ID}\",
            \"defaultSchema\": \"staging\",
            \"defaultLocation\": \"${REGION}\"
        }
    }" \
    "https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_ID}/workspaces/${WORKSPACE}?updateMask=compilationOverrides" \
    > /dev/null

echo "   ✅ Workspace configured"

# Verify connection
echo ""
echo "5. Verifying connection..."
REPO_INFO=$(curl -s -X GET \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://dataform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/${REGION}/repositories/${REPO_ID}")

GIT_URL=$(echo "$REPO_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('gitRemoteSettings', {}).get('url', 'N/A'))" 2>/dev/null || echo "N/A")

if [ "$GIT_URL" != "N/A" ] && [ "$GIT_URL" = "$GITHUB_URL" ]; then
    echo "   ✅ Connection verified"
    echo ""
    echo "✅ Dataform repository connected to GitHub!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Go to Dataform UI to verify files are visible:"
    echo "   https://console.cloud.google.com/dataform?project=${PROJECT_ID}"
    echo ""
    echo "2. Note: Root directory 'dataform/' may need to be set in UI"
    echo "   Settings → Workspace → Compilation Override → Root Directory"
    echo ""
    echo "3. Compile to verify:"
    echo "   Click 'Compile' button in Dataform UI"
else
    echo "   ⚠️  Connection may need UI configuration"
    echo "   Go to: https://console.cloud.google.com/dataform?project=${PROJECT_ID}"
fi

