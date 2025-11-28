# Quick Start: Push to GitHub

**Date**: November 28, 2025

---

## Step 1: Create Repository on GitHub

**Go to**: https://github.com/new

**Fill in**:
- Repository name: `CBI-V15`
- Description: `Institutional-grade ZL (soybean oil futures) price forecasting using Dataform ETL, Mac M4 training, and BigQuery storage`
- Visibility: **Private** (recommended)
- **IMPORTANT**: Do NOT check "Add a README file", "Add .gitignore", or "Choose a license" (we already have these)

**Click**: "Create repository"

---

## Step 2: Push Local Code

After creating the repo, run:

```bash
cd /Users/zincdigital/CBI-V15
./push_to_github.sh
```

**OR manually**:

```bash
cd /Users/zincdigital/CBI-V15
git remote add origin https://github.com/zincdigitalofmiami/CBI-V15.git
git branch -M main
git push -u origin main
```

---

## Step 3: Verify

Go to: https://github.com/zincdigitalofmiami/CBI-V15

You should see:
- ✅ README.md
- ✅ All 25+ files
- ✅ Folder structure

---

## Troubleshooting

### Authentication Error?
```bash
# Use SSH instead
git remote set-url origin git@github.com:zincdigitalofmiami/CBI-V15.git
git push -u origin main
```

### Remote Already Exists?
```bash
# Remove and re-add
git remote remove origin
git remote add origin https://github.com/zincdigitalofmiami/CBI-V15.git
git push -u origin main
```

---

**Status**: Local repo ready, waiting for GitHub repo creation

