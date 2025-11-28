#!/bin/bash
# Store API keys in macOS Keychain and GCP Secret Manager
# Usage: ./store_api_keys.sh

set -e

PROJECT_ID="cbi-v15"

echo "🔐 API Key Storage Setup"
echo "========================"
echo ""

# Function to store in Keychain
store_keychain() {
    local key_name=$1
    local service=$2
    
    echo "📝 Storing $key_name in macOS Keychain..."
    read -sp "Enter $key_name: " key_value
    echo ""
    
    if [ -z "$key_value" ]; then
        echo "   ⚠️  Skipping $key_name (empty)"
        return
    fi
    
    security add-generic-password \
        -a "$service" \
        -s "$key_name" \
        -w "$key_value" \
        -U
    
    echo "   ✅ Stored $key_name in Keychain"
}

# Function to store in Secret Manager
store_secret() {
    local secret_name=$1
    
    echo "📝 Storing $secret_name in Secret Manager..."
    read -sp "Enter value: " secret_value
    echo ""
    
    if [ -z "$secret_value" ]; then
        echo "   ⚠️  Skipping $secret_name (empty)"
        return
    fi
    
    # Check if secret exists
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &> /dev/null; then
        echo "   Secret exists, adding new version..."
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" \
            --data-file=- \
            --project="$PROJECT_ID"
    else
        echo "   Creating new secret..."
        echo -n "$secret_value" | gcloud secrets create "$secret_name" \
            --data-file=- \
            --project="$PROJECT_ID" \
            --replication-policy="automatic"
    fi
    
    echo "   ✅ Stored $secret_name in Secret Manager"
}

echo "Choose storage location:"
echo "1) macOS Keychain only (for local scripts)"
echo "2) Secret Manager only (for Cloud Scheduler)"
echo "3) Both (recommended)"
read -p "Choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Storing in macOS Keychain..."
        store_keychain "DATABENTO_API_KEY" "databento"
        store_keychain "FRED_API_KEY" "fred"
        store_keychain "SCRAPECREATORS_API_KEY" "scrapecreators"
        store_keychain "GLIDE_API_KEY" "glide"
        ;;
    2)
        echo ""
        echo "Storing in Secret Manager..."
        store_secret "databento-api-key"
        store_secret "fred-api-key"
        store_secret "scrapecreators-api-key"
        store_secret "glide-api-key"
        ;;
    3)
        echo ""
        echo "Storing in both Keychain and Secret Manager..."
        echo ""
        echo "=== macOS Keychain ==="
        store_keychain "DATABENTO_API_KEY" "databento"
        store_keychain "FRED_API_KEY" "fred"
        store_keychain "SCRAPECREATORS_API_KEY" "scrapecreators"
        store_keychain "GLIDE_API_KEY" "glide"
        echo ""
        echo "=== Secret Manager ==="
        store_secret "databento-api-key"
        store_secret "fred-api-key"
        store_secret "scrapecreators-api-key"
        store_secret "glide-api-key"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "✅ API key storage complete!"
echo ""
echo "To retrieve from Keychain:"
echo "  security find-generic-password -s DATABENTO_API_KEY -w"
echo ""
echo "To retrieve from Secret Manager:"
echo "  gcloud secrets versions access latest --secret=databento-api-key"

