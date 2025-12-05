terraform {
  required_providers {
    vercel = {
      source  = "vercel/vercel"
      version = "~> 1.0"
    }
  }
}

provider "vercel" {
  # API token will be set via VERCEL_API_TOKEN environment variable
  # Get your token from: https://vercel.com/account/tokens
}

variable "motherduck_token" {
  description = "MotherDuck service account token"
  type        = string
  sensitive   = true
}

variable "databento_api_key" {
  description = "Databento API key for market data ingestion"
  type        = string
  sensitive   = true
  default     = ""
}

# Vercel Project
resource "vercel_project" "cbi_v15_dashboard" {
  name      = "cbi-v15-dashboard"
  framework = "nextjs"

  # Git repository (update with your repo details)
  git_repository = {
    type = "github"
    repo = "zincdigitalofmiami/CBI-V15"
  }

  # Build settings
  build_command     = "cd dashboard && npm run build"
  output_directory  = "dashboard/.next"
  install_command   = "cd dashboard && npm install"
  root_directory    = "/"

  # Environment variables (available in all environments)
  environment = [
    {
      key    = "MOTHERDUCK_TOKEN"
      value  = var.motherduck_token
      target = ["production", "preview", "development"]
    },
    {
      key    = "NODE_ENV"
      value  = "production"
      target = ["production"]
    }
  ]
}

# Production environment variables
resource "vercel_project_environment_variable" "motherduck_token_prod" {
  project_id = vercel_project.cbi_v15_dashboard.id
  key        = "MOTHERDUCK_TOKEN"
  value      = var.motherduck_token
  target     = ["production", "preview"]
  sensitive  = true
}

# Optional: Databento API key (for future ingestion endpoints)
resource "vercel_project_environment_variable" "databento_key" {
  count      = var.databento_api_key != "" ? 1 : 0
  project_id = vercel_project.cbi_v15_dashboard.id
  key        = "DATABENTO_API_KEY"
  value      = var.databento_api_key
  target     = ["production"]
  sensitive  = true
}

# Project domain (optional - Vercel provides default)
resource "vercel_project_domain" "cbi_dashboard_domain" {
  project_id = vercel_project.cbi_v15_dashboard.id
  domain     = "${vercel_project.cbi_v15_dashboard.name}.vercel.app"
}

# Outputs
output "project_id" {
  description = "Vercel project ID"
  value       = vercel_project.cbi_v15_dashboard.id
}

output "deployment_url" {
  description = "Default deployment URL"
  value       = "https://${vercel_project.cbi_v15_dashboard.name}.vercel.app"
}
