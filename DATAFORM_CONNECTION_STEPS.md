# Dataform Connection - Step-by-Step Guide

**After SSH key is added to GitHub**

---

## Step 1: Add SSH Key to GitHub ✅

**Your Public Key:**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC1lQKFcHsbV9u+nHIYo/BjSBAEHpG1A4OBDvPk4NdrA dataform-cbi-v15@gcp
```

1. Go to: **https://github.com/settings/ssh/new**
2. **Title**: `Dataform CBI-V15`
3. **Key**: Paste the key above
4. Click **"Add SSH key"**

**Verify**: Run `ssh -T git@github.com` - should show "successfully authenticated"

---

## Step 2: Connect Dataform in Google Cloud Console

### 2.1 Open Dataform
**Link**: https://console.cloud.google.com/dataform?project=cbi-v15

### 2.2 Create Repository (if doesn't exist)
1. Click **"Create Repository"**
2. **Name**: `CBI-V15`
3. **Location**: `us-central1`
4. Click **"Create"**

### 2.3 Connect to GitHub
1. Go to **Settings** → **Git Remote Settings**
2. Click **"Connect to GitHub"** or **"Edit"**
3. **Connection Method**: Select **SSH**
4. Fill in:
   - **SSH URL**: `git@github.com:zincdigital/CBI-V15.git`
   - **Secret**: `dataform-github-ssh-key` (from dropdown)
   - **Branch**: `main`
   - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
5. Click **"Connect"** or **"Save"**

---

## Step 3: Verify Connection

### Check Repository Status
- "Create repository" button should disappear
- Repository name shows: `CBI-V15`
- Git remote shows: `git@github.com:zincdigital/CBI-V15.git`

### Check Files Visible
1. Navigate to `definitions/01_raw/`
2. Should see: `databento_daily.sqlx`, `fred_macro.sqlx`, etc.
3. Navigate to `definitions/02_staging/`
4. Should see: `market_daily.sqlx`, `fred_macro_clean.sqlx`, etc.

### Test Compilation
1. Click **"Compile"** button in Dataform UI
2. Should show: **"Compiled 18 action(s)"**
3. May show 2 warnings about UDF includes (non-critical)

---

## Step 4: First Dataform Run (After Data Ingestion)

Once you have data in raw tables:

1. **Run Staging Layer**:
   - Click **"Run"** → Select **"staging"** tag
   - Or: `npx dataform run --tags staging` (from CLI)

2. **Run Features Layer**:
   - Click **"Run"** → Select **"features"** tag
   - Or: `npx dataform run --tags features` (from CLI)

3. **Run Assertions**:
   - Click **"Test"** button
   - Or: `npx dataform test` (from CLI)

---

## ✅ Success Indicators

- ✅ Repository connected
- ✅ Files visible in UI
- ✅ Compilation successful
- ✅ Can run transformations
- ✅ No connection errors

---

## 🔧 Troubleshooting

**"Repository not found"**:
- Verify SSH key is added to GitHub
- Check repository name: `zincdigital/CBI-V15`
- Test SSH: `ssh -T git@github.com`

**"Files not visible"**:
- **Check Root Directory**: Must be `dataform/` (not `/`)
- Refresh the page
- Verify files exist in GitHub

**"Permission denied"**:
- Verify SSH key is on GitHub
- Check secret exists: `gcloud secrets list --project=cbi-v15`
- Re-add SSH key if needed

**"Compilation errors"**:
- Check Root Directory is `dataform/`
- Verify `dataform.json` exists
- Check file paths in SQL files

---

**After Connection**: Dataform is ready for ETL operations!

