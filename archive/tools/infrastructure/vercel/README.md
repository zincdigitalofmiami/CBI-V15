# Vercel Infrastructure - Terraform

Infrastructure-as-Code for deploying the CBI-V15 dashboard to Vercel.

## Prerequisites

1. **Terraform** installed (v1.0+)
2. **Vercel account** with GitHub connected
3. **Vercel API token** from https://vercel.com/account/tokens

## Setup

### 1. Configure Variables

```bash
cd infrastructure/vercel

# Copy example and fill in your tokens
cp terraform.tfvars.example terraform.tfvars

# Edit with your actual tokens
nano terraform.tfvars
```

### 2. Set Vercel API Token

```bash
# Export Vercel token for Terraform provider
export VERCEL_API_TOKEN="your_vercel_api_token_here"
```

### 3. Initialize Terraform

```bash
terraform init
```

### 4. Review Plan

```bash
terraform plan
```

### 5. Deploy

```bash
terraform apply
```

## What This Creates

- ✅ Vercel project: `cbi-v15-dashboard`
- ✅ Environment variables:
  - `MOTHERDUCK_TOKEN` (production + preview)
  - `DATABENTO_API_KEY` (optional, production only)
  - `NODE_ENV=production`
- ✅ Default domain: `cbi-v15-dashboard.vercel.app`
- ✅ GitHub integration for automatic deployments

## Managing Environment Variables

To update environment variables:

```bash
# Edit terraform.tfvars with new values
nano terraform.tfvars

# Apply changes
terraform apply
```

## Custom Domain (Optional)

To add a custom domain, edit `main.tf`:

```hcl
resource "vercel_project_domain" "custom" {
  project_id = vercel_project.cbi_v15_dashboard.id
  domain     = "dashboard.yourdomain.com"
}
```

## Outputs

After deployment, Terraform will output:
- `project_id` - Vercel project ID
- `deployment_url` - Dashboard URL

## Tear Down

To remove all resources:

```bash
terraform destroy
```

## Security Notes

- ⚠️ Never commit `terraform.tfvars` (excluded in `.gitignore`)
- ⚠️ Keep `VERCEL_API_TOKEN` in environment, not in files
- ✅ All secrets marked as `sensitive = true` in Terraform
