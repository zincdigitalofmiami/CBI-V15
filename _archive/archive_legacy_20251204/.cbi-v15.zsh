# CBI-V15 Terminal Configuration
# Source this file in your ~/.zshrc: source "/Volumes/Satechi Hub/CBI-V15/.cbi-v15.zsh"

# Project Configuration
export CBI_V15_ROOT="/Volumes/Satechi Hub/CBI-V15"
export CBI_V15_PROJECT="cbi-v15"
export CBI_V15_LOCATION="us-central1"
export CBI_V15_DATAFORM_DIR="${CBI_V15_ROOT}/dataform"
export GOOGLE_CLOUD_PROJECT="${CBI_V15_PROJECT}"

# Python Configuration
export PYTHONPATH="${CBI_V15_ROOT}/src:${PYTHONPATH}"

# Navigate to project root
alias cbi='cd ${CBI_V15_ROOT}'
alias cbidf='cd ${CBI_V15_DATAFORM_DIR}'

# GCP Configuration
alias gcp-set='gcloud config set project ${CBI_V15_PROJECT}'
alias gcp-get='gcloud config get-value project'
alias gcp-auth='gcloud auth login'

# BigQuery Quick Access
# Note: src/utils renamed to src/cbi_utils to avoid conflicts with gcloud/bq internal modules
alias bq-ls='bq ls --project_id=${CBI_V15_PROJECT}'
alias bq-datasets='bq ls --project_id=${CBI_V15_PROJECT} --format=prettyjson'
alias bq-tables='bq ls --project_id=${CBI_V15_PROJECT} --dataset_id='

# Dataform Commands
alias df-compile='cd ${CBI_V15_DATAFORM_DIR} && npx dataform compile'
alias df-run='cd ${CBI_V15_DATAFORM_DIR} && npx dataform run'
alias df-test='cd ${CBI_V15_DATAFORM_DIR} && npx dataform test'
alias df-run-staging='cd ${CBI_V15_DATAFORM_DIR} && npx dataform run --tags staging'
alias df-run-features='cd ${CBI_V15_DATAFORM_DIR} && npx dataform run --tags features'
alias df-run-training='cd ${CBI_V15_DATAFORM_DIR} && npx dataform run --tags training'

# Setup Scripts
alias cbi-preflight='cd ${CBI_V15_ROOT} && ./scripts/setup/pre_flight_check.sh'
alias cbi-setup-gcp='cd ${CBI_V15_ROOT} && ./scripts/setup/setup_gcp_project.sh'
alias cbi-setup-iam='cd ${CBI_V15_ROOT} && ./scripts/setup/setup_iam_permissions.sh'
alias cbi-setup-bq='cd ${CBI_V15_ROOT} && ./scripts/setup/setup_bigquery_skeleton.sh'
alias cbi-setup-keys='cd ${CBI_V15_ROOT} && ./scripts/setup/store_api_keys.sh'
alias cbi-verify='cd ${CBI_V15_ROOT} && python3 scripts/setup/verify_connections.py'

# Ingestion Scripts
alias cbi-ingest-databento='cd ${CBI_V15_ROOT} && python3 src/ingestion/databento/collect_daily.py'
alias cbi-ingest-fred='cd ${CBI_V15_ROOT} && python3 src/ingestion/fred/collect_daily.py'
alias cbi-ingest-usda='cd ${CBI_V15_ROOT} && python3 src/ingestion/usda/collect_daily.py'
alias cbi-ingest-cftc='cd ${CBI_V15_ROOT} && python3 src/ingestion/cftc/collect_daily.py'

# Training Scripts
alias cbi-train-lightgbm='cd ${CBI_V15_ROOT} && python3 src/training/baselines/lightgbm_zl.py'
alias cbi-export-data='cd ${CBI_V15_ROOT} && python3 scripts/export/export_training_data.py'

# Validation Scripts
alias cbi-quality='cd ${CBI_V15_ROOT} && python3 scripts/validation/data_quality_checks.py'

# Helper Functions
cbi-status() {
    echo "🔍 CBI-V15 Status Check"
    echo "======================"
    echo ""
    echo "Project Root: ${CBI_V15_ROOT}"
    echo "GCP Project: $(gcloud config get-value project 2>/dev/null || echo 'Not set')"
    echo "Location: ${CBI_V15_LOCATION}"
    echo ""
    echo "Python: $(python3 --version 2>/dev/null || echo 'Not found')"
    echo "Node.js: $(node --version 2>/dev/null || echo 'Not found')"
    echo "gcloud: $(gcloud --version 2>&1 | head -1)"
    echo "bq: $(bq version 2>&1 | head -1)"
    echo ""
    echo "Current Directory: $(pwd)"
    echo ""
}

cbi-set-project() {
    gcloud config set project ${CBI_V15_PROJECT}
    echo "✅ GCP project set to ${CBI_V15_PROJECT}"
}

cbi-datasets() {
    echo "📊 BigQuery Datasets in ${CBI_V15_PROJECT}:"
    bq ls --project_id=${CBI_V15_PROJECT} --format=prettyjson | grep -E '"datasetId"|"location"' | head -20
}

cbi-quick-setup() {
    echo "🚀 CBI-V15 Quick Setup"
    echo "====================="
    echo ""
    echo "Running pre-flight check..."
    cd ${CBI_V15_ROOT} && ./scripts/setup/pre_flight_check.sh
    echo ""
    echo "✅ Pre-flight check complete"
    echo ""
    echo "Next steps:"
    echo "  1. cbi-setup-gcp    - Setup GCP project"
    echo "  2. cbi-setup-iam     - Setup IAM permissions"
    echo "  3. cbi-setup-bq      - Setup BigQuery skeleton"
    echo "  4. cbi-setup-keys    - Store API keys"
    echo "  5. cbi-verify        - Verify setup"
}

# Only auto-configure if we're in CBI-V15 directory or explicitly requested
# Skip auto-configuration if running non-interactively (e.g., from scripts)
if [[ -t 0 ]] && [[ -t 1 ]]; then
    CURRENT_DIR=$(pwd)
    if [[ "$CURRENT_DIR" == "${CBI_V15_ROOT}"* ]] || [[ -n "$CBI_V15_FORCE_LOAD" ]]; then
        # Auto-set GCP project on shell start (only in CBI-V15)
        if command -v gcloud &> /dev/null; then
            CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
            if [ "$CURRENT_PROJECT" != "${CBI_V15_PROJECT}" ]; then
                echo "🔧 Setting GCP project to ${CBI_V15_PROJECT}..." >&2
                gcloud config set project ${CBI_V15_PROJECT} &>/dev/null
            fi
        fi
        
        # Welcome message (only in CBI-V15, only if interactive)
        echo "✅ CBI-V15 terminal configuration loaded" >&2
        echo "   Type 'cbi-status' for project status" >&2
        echo "   Type 'cbi-quick-setup' for setup guide" >&2
    fi
fi

