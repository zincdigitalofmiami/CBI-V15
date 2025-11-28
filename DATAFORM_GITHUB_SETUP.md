# Dataform GitHub Repository Setup

**Issue**: Dataform UI shows "Create repository" button  
**Solution**: Connect Dataform to the GitHub repository

---

## Steps to Connect Dataform to GitHub

### 1. Ensure Repository Exists on GitHub

The repository `zincdigitalofmiami/CBI-V15` needs to exist on GitHub.

**Option A: Create via GitHub Web UI**
1. Go to https://github.com/new
2. Repository name: `CBI-V15`
3. Owner: `zincdigital`
4. Set as Public or Private
5. **DO NOT** initialize with README (we already have files)
6. Click "Create repository"

**Option B: Create via GitHub CLI** (if you have `gh` installed)
```bash
gh repo create zincdigitalofmiami/CBI-V15 --public --source=. --remote=origin --push
```

### 2. Push Local Repository to GitHub

Once the repository exists on GitHub:

```bash
cd /Users/zincdigital/CBI-V15
git push -u origin main
```

**If authentication fails**, use a Personal Access Token:
1. Go to https://github.com/settings/tokens
2. Generate new token (classic) with `repo` scope
3. Use token as password when pushing

### 3. Connect Dataform to GitHub Repository

In Google Cloud Dataform UI:

1. Go to Dataform → Your Repository
2. Click "Connect Repository" or "Link GitHub Repository"
3. Select: `zincdigitalofmiami/CBI-V15`
4. Branch: `main`
5. Root Directory: `dataform/` (if Dataform is in a subdirectory)
6. Click "Connect"

### 4. Verify Connection

After connecting:
- Dataform should show the repository name instead of "Create repository"
- You should see your SQL files in the Dataform UI
- Compilation should work from the UI

---

## Current Status

- ✅ Local repository initialized
- ✅ Dataform structure created
- ✅ All files committed locally
- ⚠️ Need to push to GitHub
- ⚠️ Need to connect in Dataform UI

---

## Quick Push Command (After Repo Created)

```bash
cd /Users/zincdigital/CBI-V15
git push -u origin main
```

If you need to use SSH instead:
```bash
git remote set-url origin git@github.com:zincdigitalofmiami/CBI-V15.git
git push -u origin main
```

