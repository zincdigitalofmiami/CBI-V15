# CBI-V15 Environment Setup
# Source this file to configure your shell: source .cbi-v15.zsh

# Project directory
export CBI_V15_ROOT="/Volumes/Satechi Hub/CBI-V15"
cd "$CBI_V15_ROOT"

# Load environment variables from .env file
if [ -f "$CBI_V15_ROOT/.env" ]; then
    set -a
    source "$CBI_V15_ROOT/.env"
    set +a
    echo "Loaded environment from .env"
else
    echo "Warning: .env file not found"
fi

# Project-specific settings (non-sensitive)
export GOOGLE_CLOUD_PROJECT=cbi-v15-forecasting
export MOTHERDUCK_DATABASE=cbi_v15

# Activate Python virtual environment if it exists
if [ -d "$CBI_V15_ROOT/.venv" ]; then
    source "$CBI_V15_ROOT/.venv/bin/activate"
    echo "Activated Python virtual environment"
fi

echo "CBI-V15 environment ready"
