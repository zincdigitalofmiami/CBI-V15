# Add SSH Key to GitHub - Quick Instructions

## Your Public SSH Key

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC1lQKFcHsbV9u+nHIYo/BjSBAEHpG1A4OBDvPk4NdrA dataform-cbi-v15@gcp
```

## Quick Add (3 Steps)

### Step 1: Go to GitHub SSH Settings
**Link**: https://github.com/settings/ssh/new

### Step 2: Fill in the Form
- **Title**: `Dataform CBI-V15`
- **Key**: Paste the key above
- **Key type**: Authentication Key (default)

### Step 3: Click "Add SSH key"

---

## Verify Key Added

After adding, verify:
```bash
ssh -T git@github.com
```

Should show: "Hi zincdigital! You've successfully authenticated..."

---

## Then Connect Dataform

After SSH key is added:
1. Go to: https://console.cloud.google.com/dataform?project=cbi-v15
2. Create/Select repository: `CBI-V15`
3. Settings → Connect to GitHub
4. SSH URL: `git@github.com:zincdigital/CBI-V15.git`
5. Secret: `dataform-github-ssh-key`
6. Root Directory: `dataform/`
7. Click "Connect"

---

**That's it!** Once connected, Dataform will have access to your repository.

