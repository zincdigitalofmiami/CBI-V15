#!/bin/bash
# Setup SSH key for Dataform GitHub connection
# This prepares the SSH key that will be used to connect Dataform to GitHub

set -e

PROJECT_ID="cbi-v15"
SECRET_NAME="dataform-github-ssh-key"
KEY_NAME="dataform_github_ed25519"
KEY_PATH="$HOME/.ssh/${KEY_NAME}"

echo "🔐 Setting up SSH Key for Dataform GitHub Connection"
echo "===================================================="
echo ""

# Check if key already exists
if [ -f "${KEY_PATH}" ]; then
    echo "⚠️  SSH key already exists at ${KEY_PATH}"
    read -p "Do you want to use existing key? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        echo "Exiting. Delete the key first if you want to regenerate."
        exit 0
    fi
else
    # Generate SSH key
    echo "Generating SSH key pair..."
    ssh-keygen -t ed25519 -C "dataform-${PROJECT_ID}@gcp" -f "${KEY_PATH}" -N ""
    echo "✅ SSH key generated"
fi

# Display public key
echo ""
echo "📋 Public Key (add this to GitHub):"
echo "===================================="
cat "${KEY_PATH}.pub"
echo ""
echo "===================================="
echo ""

# Instructions for adding to GitHub
echo "📝 Next Steps:"
echo ""
echo "1. Add Public Key to GitHub:"
echo "   - Go to: https://github.com/settings/ssh/new"
echo "   - Title: Dataform CBI-V15"
echo "   - Key: Copy the public key above"
echo "   - Click 'Add SSH key'"
echo ""
read -p "Press Enter after adding the key to GitHub..."

# Store private key in Secret Manager
echo ""
echo "2. Storing private key in Secret Manager..."
if gcloud secrets describe "${SECRET_NAME}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "   Secret exists, adding new version..."
    cat "${KEY_PATH}" | gcloud secrets versions add "${SECRET_NAME}" \
        --data-file=- \
        --project="${PROJECT_ID}"
else
    echo "   Creating new secret..."
    cat "${KEY_PATH}" | gcloud secrets create "${SECRET_NAME}" \
        --data-file=- \
        --project="${PROJECT_ID}" \
        --replication-policy="automatic"
fi

# Grant Dataform service account access
echo ""
echo "3. Granting Dataform service account access..."
SERVICE_ACCOUNT="service-$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')@gcp-sa-dataform.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="${PROJECT_ID}"

echo ""
echo "✅ SSH key setup complete!"
echo ""
echo "4. Connect in Dataform UI:"
echo "   - Go to: https://console.cloud.google.com/dataform"
echo "   - Select repository (or create CBI-V15)"
echo "   - Settings → Connect to GitHub"
echo "   - SSH URL: git@github.com:zincdigital/CBI-V15.git"
echo "   - Secret: ${SECRET_NAME}"
echo "   - Root Directory: dataform/"
echo "   - Click 'Connect'"
echo ""

