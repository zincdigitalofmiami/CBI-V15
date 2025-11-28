# Dataform GitHub Connection - Quick Instructions

## ✅ Setup Complete

- ✅ SSH key generated: `~/.ssh/dataform_github_ed25519`
- ✅ Private key stored in Secret Manager: `dataform-github-ssh-key`
- ⚠️ **Add public key to GitHub** (next step)

---

## Step 1: Add Public Key to GitHub

**Your Public Key:**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC1lQKFcHsbV9u+nHIYo/BjSBAEHpG1A4OBDvPk4NdrA dataform-cbi-v15@gcp
```

1. Go to: **https://github.com/settings/ssh/new**
2. **Title**: `Dataform CBI-V15`
3. **Key**: Paste the key above
4. Click **"Add SSH key"**

---

## Step 2: Connect in Dataform UI

1. **Open Dataform Console**:
   - https://console.cloud.google.com/dataform?project=cbi-v15

2. **Create Repository** (if needed):
   - Click **"Create Repository"**
   - Name: `CBI-V15`
   - Location: `us-central1`
   - Click **"Create"**

3. **Connect to GitHub**:
   - Go to **Settings** → **Git Remote Settings**
   - Click **"Connect to GitHub"**
   - **Connection Method**: SSH
   - **SSH URL**: `git@github.com:zincdigital/CBI-V15.git`
   - **Secret**: `dataform-github-ssh-key`
   - **Branch**: `main`
   - **Root Directory**: `dataform/` ⚠️ **CRITICAL**
   - Click **"Connect"**

---

## ✅ Verify Connection

After connecting:
- ✅ "Create repository" button disappears
- ✅ SQL files visible in `definitions/` folders
- ✅ Can compile from UI
- ✅ Shows "Compiled 18 action(s)"

---

**That's it!** Dataform is now connected to GitHub.

