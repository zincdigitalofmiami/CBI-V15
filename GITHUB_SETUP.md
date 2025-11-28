# GitHub Repository Setup Instructions

**Date**: November 28, 2025

---

## Step 1: Create Repository on GitHub

1. Go to: https://github.com/new
2. Repository name: `CBI-V15`
3. Description: `Institutional-grade ZL (soybean oil futures) price forecasting using Dataform ETL, Mac M4 training, and BigQuery storage`
4. Visibility: **Private** (recommended) or Public
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

---

## Step 2: Connect Local Repository to GitHub

After creating the repo on GitHub, run:

```bash
cd /Users/zincdigital/CBI-V15

# Add remote (replace YOUR_USERNAME if different)
git remote add origin https://github.com/zincdigitalofmiami/CBI-V15.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 3: Verify

1. Go to: https://github.com/zincdigitalofmiami/CBI-V15
2. Verify all files are present
3. Check that README.md displays correctly

---

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
cd /Users/zincdigital/CBI-V15

# Create repo and push in one command
gh repo create CBI-V15 --private --source=. --remote=origin --push
```

---

## Troubleshooting

### If you get authentication errors:
```bash
# Use GitHub CLI to authenticate
gh auth login

# Or use SSH instead of HTTPS
git remote set-url origin git@github.com:zincdigitalofmiami/CBI-V15.git
```

### If you need to force push (not recommended):
```bash
git push -u origin main --force
```

---

**Last Updated**: November 28, 2025

