# Push CBI-V15 to GitHub

## Current Status
- ✅ Local repository: Ready (71 commits)
- ✅ Remote configured: `https://github.com/zincdigital/CBI-V15.git`
- ⚠️ Repository on GitHub: **Needs to be created**

## Steps

### 1. Create Repository on GitHub

**Via Web UI:**
1. Go to: https://github.com/new
2. Repository name: `CBI-V15`
3. Owner: `zincdigital`
4. **DO NOT** check "Initialize with README" (we have files)
5. Click "Create repository"

**Via GitHub CLI** (if authenticated):
```bash
gh repo create zincdigital/CBI-V15 --public --source=. --remote=origin --push
```

### 2. Push After Repository Created

```bash
cd /Users/zincdigital/CBI-V15
git push -u origin main
```

**If authentication fails:**
- Use Personal Access Token: https://github.com/settings/tokens
- Generate token with `repo` scope
- Use token as password when prompted

**Or use SSH:**
```bash
git remote set-url origin git@github.com:zincdigital/CBI-V15.git
git push -u origin main
```

### 3. Connect in Dataform UI

After repository exists and is pushed:
1. Go to Google Cloud Console → Dataform
2. Click "Connect Repository"
3. Select: `zincdigital/CBI-V15`
4. Branch: `main`
5. Root Directory: `dataform/`
6. Click "Connect"

---

**Ready to push**: 71 commits waiting

