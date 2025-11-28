# Dataform GitHub Connection - Ready to Connect ✅

**Date**: November 28, 2025  
**Status**: ✅ SSH Key Setup Complete

---

## ✅ Completed Setup

1. ✅ SSH key pair generated
2. ✅ Private key stored in Secret Manager (`dataform-github-ssh-key`)
3. ⚠️ Dataform service account will be created automatically when Dataform is first used
4. ⚠️ **Public key needs to be added to GitHub** (see below)

---

## 📋 Next Steps

### Step 1: Add Public Key to GitHub

**Your Public SSH Key:**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC1lQKFcHsbV9u+nHIYo/BjSBAEHpG1A4OBDvPk4NdrA dataform-cbi-v15@gcp
```

**Add to GitHub:**
1. Go to: https://github.com/settings/ssh/new
2. **Title**: `Dataform CBI-V15`
3. **Key**: Paste the public key above
4. Click **"Add SSH key"**

---

### Step 2: Connect in Dataform UI

1. **Go to Dataform Console**:
   - https://console.cloud.google.com/dataform?project=cbi-v15

2. **Create Repository** (if doesn't exist):
   - Click **"Create Repository"**
   - **Name**: `CBI-V15`
   - **Location**: `us-central1`
   - Click **"Create"**

3. **Connect to GitHub**:
   - Go to **Settings** → **Git Remote Settings**
   - Click **"Connect to GitHub"**
   - Select **SSH** connection method
   - Enter:
     - **SSH URL**: `git@github.com:zincdigital/CBI-V15.git`
     - **Secret**: `dataform-github-ssh-key`
     - **Branch**: `main`
     - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
   - Click **"Connect"**

---

## ✅ Verification

After connecting, verify:

1. **Repository Connected**:
   - "Create repository" button should disappear
   - Repository name shows: `CBI-V15`

2. **Files Visible**:
   - Navigate to `definitions/01_raw/`
   - Should see SQL files: `databento_daily.sqlx`, `fred_macro.sqlx`, etc.

3. **Compilation Works**:
   - Click **"Compile"** button
   - Should show: "Compiled 18 action(s)"
   - May show 2 warnings about UDF includes (non-critical)

---

## 🔧 Troubleshooting

**"Repository not found"**:
- Verify SSH key is added to GitHub
- Check repository name: `zincdigital/CBI-V15`
- Verify repository is accessible

**"Secret not found"**:
- Secret name: `dataform-github-ssh-key`
- Verify it exists: `gcloud secrets list --project=cbi-v15`

**"Files not visible"**:
- **Check Root Directory**: Must be `dataform/` (not `/` or empty)
- Refresh the page
- Verify files exist in GitHub

**"Permission denied"**:
- Verify SSH key is added to GitHub
- Check service account has Secret Manager access
- Re-run: `./scripts/setup/setup_dataform_github_ssh.sh`

---

## 📝 Quick Reference

**Public Key Location**: `~/.ssh/dataform_github_ed25519.pub`  
**Secret Name**: `dataform-github-ssh-key`  
**GitHub SSH URL**: `git@github.com:zincdigital/CBI-V15.git`  
**Root Directory**: `dataform/` ⚠️ **CRITICAL**

---

**Status**: ✅ **READY TO CONNECT**

SSH key setup complete. Add the public key to GitHub, then connect in the Dataform UI.

