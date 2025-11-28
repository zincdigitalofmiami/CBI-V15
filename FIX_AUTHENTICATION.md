# Fix GitHub Authentication

**Issue**: Authentication failed in GitHub Desktop

---

## Solution 1: Switch to HTTPS (Recommended for GitHub Desktop)

I've switched the remote to HTTPS. Now:

1. **In GitHub Desktop**:
   - Repository → Repository Settings → Remote
   - Verify it shows: `https://github.com/zincdigitalofmiami/CBI-V15.git`
   - If not, update it manually

2. **Try Publishing Again**:
   - Click "Publish repository"
   - GitHub Desktop will prompt for credentials
   - Use your GitHub username and a **Personal Access Token** (not password)

3. **Create Personal Access Token** (if needed):
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Name: `CBI-V15 Desktop`
   - Select scopes: ✅ `repo` (full control)
   - Click "Generate token"
   - **Copy the token** (you won't see it again!)
   - Use this token as your password in GitHub Desktop

---

## Solution 2: Fix SSH Authentication

If you prefer SSH:

1. **Add SSH Key to ssh-agent**:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   # Or: ssh-add ~/.ssh/id_rsa
   ```

2. **Verify SSH Key is on GitHub**:
   - Go to: https://github.com/settings/keys
   - Make sure your SSH key is listed
   - If not, add it: https://github.com/settings/ssh/new

3. **Test SSH Connection**:
   ```bash
   ssh -T git@github.com
   ```
   Should see: "Hi zincdigitalofmiami! You've successfully authenticated..."

4. **Switch Remote Back to SSH**:
   ```bash
   cd /Users/zincdigital/CBI-V15
   git remote set-url origin git@github.com:zincdigitalofmiami/CBI-V15.git
   ```

---

## Solution 3: Login to GitHub Desktop

1. **Check Login Status**:
   - GitHub Desktop → Preferences → Accounts
   - Verify you're logged in as `zincdigitalofmiami`

2. **Re-authenticate**:
   - Sign out and sign back in
   - Use Personal Access Token if prompted

---

## Quick Fix: Use Command Line Instead

If GitHub Desktop continues to have issues:

```bash
cd /Users/zincdigital/CBI-V15

# Use HTTPS with token
git remote set-url origin https://github.com/zincdigitalofmiami/CBI-V15.git

# Push (will prompt for username/token)
git push -u origin main
```

When prompted:
- Username: `zincdigitalofmiami`
- Password: Use Personal Access Token (create at https://github.com/settings/tokens)

---

**Current Remote**: Switched to HTTPS (better for GitHub Desktop)

