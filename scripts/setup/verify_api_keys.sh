#!/bin/bash
# Verify API keys are stored in Keychain and/or Secret Manager

set -e

PROJECT_ID="cbi-v15"

echo "🔍 Verifying API Keys"
echo "===================="
echo ""

# Check macOS Keychain
echo "📱 macOS Keychain:"
KEYS=("DATABENTO_API_KEY" "SCRAPECREATORS_API_KEY" "FRED_API_KEY" "GLIDE_API_KEY")
for key in "${KEYS[@]}"; do
    if security find-generic-password -s "$key" &> /dev/null; then
        echo "  ✅ $key found in Keychain"
    else
        echo "  ⚠️  $key not found in Keychain"
    fi
done

echo ""
echo "🔐 Secret Manager:"
SECRETS=("databento-api-key" "scrapecreators-api-key" "fred-api-key" "glide-api-key")
for secret in "${SECRETS[@]}"; do
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" &> /dev/null; then
        echo "  ✅ $secret exists in Secret Manager"
    else
        echo "  ⚠️  $secret not found in Secret Manager"
    fi
done

echo ""
echo "📋 Summary:"
KEYCHAIN_COUNT=0
for key in "${KEYS[@]}"; do
    if security find-generic-password -s "$key" &> /dev/null; then
        KEYCHAIN_COUNT=$((KEYCHAIN_COUNT + 1))
    fi
done

SECRET_COUNT=$(gcloud secrets list --project="$PROJECT_ID" --filter="name:databento-api-key OR name:scrapecreators-api-key OR name:fred-api-key OR name:glide-api-key" --format="value(name)" 2>/dev/null | wc -l | tr -d ' ' || echo "0")

echo "  Keychain keys: $KEYCHAIN_COUNT"
echo "  Secret Manager secrets: $SECRET_COUNT"
echo ""
if [ "$KEYCHAIN_COUNT" -gt 0 ] || [ "$SECRET_COUNT" -gt 0 ]; then
    echo "✅ Some API keys are stored"
else
    echo "⚠️  No API keys found. Run: ./scripts/setup/store_api_keys.sh"
fi

