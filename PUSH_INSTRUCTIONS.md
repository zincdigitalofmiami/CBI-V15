# Push Instructions for CBI-V15

**Repository URL**: https://github.com/zincdigitalofmiami/CBI-V15

---

## Current Status

✅ Local repository ready with 27 files committed  
✅ Remote configured: `git@github.com:zincdigitalofmiami/CBI-V15.git`

---

## Push to GitHub

### Option 1: Using SSH (Recommended)

If you have SSH keys set up with GitHub:

```bash
cd /Users/zincdigital/CBI-V15
git push -u origin main
```

### Option 2: Using HTTPS with Credential Helper

```bash
cd /Users/zincdigital/CBI-V15
git remote set-url origin https://github.com/zincdigitalofmiami/CBI-V15.git
git push -u origin main
```

You'll be prompted for:
- Username: `zincdigitalofmiami`
- Password: Use a **Personal Access Token** (not your GitHub password)

**Create Personal Access Token**:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy the token and use it as password

### Option 3: Using GitHub Desktop

1. Open GitHub Desktop
2. File → Add Local Repository
3. Select `/Users/zincdigital/CBI-V15`
4. Click "Publish repository"
5. Select "zincdigitalofmiami/CBI-V15"

---

## Verify Push

After pushing, verify at:
https://github.com/zincdigitalofmiami/CBI-V15

You should see:
- ✅ README.md
- ✅ All 27 files
- ✅ Complete folder structure

---

## Troubleshooting

### "Repository not found"
- Verify repo exists at: https://github.com/zincdigitalofmiami/CBI-V15
- Check you have access rights
- Ensure repo name matches exactly: `CBI-V15`

### Authentication Issues
- Check SSH keys: `ssh -T git@github.com`
- Or use Personal Access Token with HTTPS

---

**Last Updated**: November 28, 2025

