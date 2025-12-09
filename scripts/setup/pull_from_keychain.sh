#!/bin/bash
# Pull API keys from macOS Keychain and populate .env.local
# Usage: ./pull_from_keychain.sh

set -e

PROJECT_DIR="/Volumes/Satechi Hub/CBI-V15"
ENV_FILE="$PROJECT_DIR/.env.local"

echo "🔑 Pulling API Keys from macOS Keychain"
echo "========================================"
echo ""

# Ensure target env file exists and back it up if present
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ Backed up existing .env.local"
else
    touch "$ENV_FILE"
    echo "✅ Created .env.local"
fi

# Function to get from Keychain and update .env.local
pull_and_update() {
    local key_name=$1
    local service_name=$2
    
    echo "🔍 Looking for $key_name in Keychain..."
    
    # Try to find in Keychain
    if security find-generic-password -s "$key_name" &> /dev/null; then
        value=$(security find-generic-password -s "$key_name" -w 2>/dev/null || echo "")
        
        if [ -n "$value" ]; then
            # Update .env.local
            if grep -q "^${key_name}=" "$ENV_FILE" 2>/dev/null; then
                # Key exists, update it
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    sed -i '' "s|^${key_name}=.*|${key_name}=${value}|" "$ENV_FILE"
                else
                    sed -i "s|^${key_name}=.*|${key_name}=${value}|" "$ENV_FILE"
                fi
                echo "  ✅ Updated $key_name in .env.local"
            else
                # Key doesn't exist, append it
                echo "${key_name}=${value}" >> "$ENV_FILE"
                echo "  ✅ Added $key_name to .env.local"
            fi
        else
            echo "  ⚠️  $key_name found but value is empty"
        fi
    else
        echo "  ⚠️  $key_name not found in Keychain"
        echo "     Try searching in Keychain Access app for: $service_name"
    fi
}

echo ""
echo "Pulling keys from Keychain..."
echo ""

# Core keys (must exist)
pull_and_update "MOTHERDUCK_TOKEN" "MotherDuck"
pull_and_update "OPENAI_API_KEY" "OpenAI"
pull_and_update "TRADINGECONOMICS_API_KEY" "TradingEconomics"
pull_and_update "PROFARMER_API_KEY" "ProFarmer"
pull_and_update "DATABENTO_API_KEY" "Databento"
pull_and_update "EIA_API_KEY" "EIA"

# Optional keys
pull_and_update "MOTHERDUCK_READ_SCALING_TOKEN" "MotherDuck"
pull_and_update "FRED_API_KEY" "FRED"
pull_and_update "NOAA_API_TOKEN" "NOAA"
pull_and_update "SCRAPECREATORS_API_KEY" "ScrapeCreators"
pull_and_update "TRIGGER_SECRET_KEY" "Trigger.dev"
pull_and_update "ANCHOR_API_KEY" "Anchor"

echo ""
echo "✅ Keychain pull complete!"
echo ""
echo "Next steps:"
echo "  1. Review .env.local: cat $ENV_FILE"
echo "  2. Verify keys: ./scripts/setup/verify_api_keys.sh"
echo "  3. Test Trigger.dev: npx trigger.dev@latest dev"
echo ""
echo "If keys are missing from Keychain:"
echo "  1. Open Keychain Access app (Cmd+Space → 'Keychain Access')"
echo "  2. Search for service name (e.g., 'ProFarmer', 'OpenAI')"
echo "  3. Double-click → Show password → Copy"
echo "  4. Run: ./scripts/setup/store_api_keys.sh"
