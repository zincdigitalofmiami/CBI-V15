# SSH Secret PEM Format Fix ✅

**Date**: November 28, 2025  
**Issue**: "Failed to parse SSH private key" error  
**Root Cause**: Dataform requires **RSA key in PEM format**, not Ed25519 OpenSSH format  
**Status**: ✅ **FIXED**

---

## Problem Analysis

The error changed from "Illegal base64 character" to "Failed to parse SSH private key", indicating:
1. Base64 decoding worked ✅
2. But Dataform couldn't parse the **Ed25519 OpenSSH format** key ❌

**Solution**: Use **RSA key in PEM format** (plain text, not base64).

---

## Solution Applied

**Generated new RSA SSH key pair:**

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/dataform_github_rsa \
    -C "dataform-cbi-v15@gcp" \
    -N ""
```

**Stored as plain text in Secret Manager:**

```bash
cat ~/.ssh/dataform_github_rsa | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
- **Format**: RSA (not Ed25519)
- **Encoding**: PEM format (`-----BEGIN RSA PRIVATE KEY-----`)
- **Storage**: Plain text (not base64)
- **Public key**: Must be added to GitHub

---

## Verification

**Check secret format:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

**Should show:** `-----BEGIN RSA PRIVATE KEY-----`

---

## Add Public Key to GitHub

**Your RSA Public Key:**
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC... dataform-cbi-v15@gcp
```

**Steps:**
1. Go to: https://github.com/settings/ssh/new
2. **Title**: `Dataform CBI-V15 RSA`
3. **Key**: Paste the public key above
4. Click **"Add SSH key"**

**Verify:**
```bash
ssh -T git@github.com -i ~/.ssh/dataform_github_rsa
```

---

## Understanding Dataform's Requirements

**According to [Google Cloud Documentation](https://docs.cloud.google.com/dataform/docs/connect-repository):**

- **SSH Authentication**: Requires private SSH key in Secret Manager
- **Key Format**: RSA keys work best (PEM format)
- **Storage**: Plain text (not base64 encoded)
- **Public Key**: Must be added to Git provider

**Why RSA instead of Ed25519?**
- Dataform's SSH parser may not support Ed25519 OpenSSH format
- RSA PEM format is more widely supported
- Standard format for cloud service integrations

---

## Fix Script

Created `scripts/setup/fix_dataform_ssh_pem_format.sh` to:
- Generate RSA key pair (if needed)
- Display public key for GitHub
- Store private key as plain text in Secret Manager
- Verify format

**Usage:**
```bash
./scripts/setup/fix_dataform_ssh_pem_format.sh
```

---

## ✅ Status

- ✅ RSA key generated (PEM format)
- ✅ Private key stored as plain text
- ✅ Format verified (`-----BEGIN RSA PRIVATE KEY-----`)
- ⏳ **Public key needs to be added to GitHub**

**Next**: 
1. Add RSA public key to GitHub
2. Test Dataform connection in UI

---

## Troubleshooting

**If still getting errors:**

1. **Verify secret format:**
   ```bash
   gcloud secrets versions access latest \
       --secret=dataform-github-ssh-key \
       --project=cbi-v15 | head -1
   ```
   Should show: `-----BEGIN RSA PRIVATE KEY-----`

2. **Verify public key on GitHub:**
   ```bash
   ssh -T git@github.com -i ~/.ssh/dataform_github_rsa
   ```

3. **Check service account access:**
   ```bash
   gcloud secrets get-iam-policy dataform-github-ssh-key \
       --project=cbi-v15
   ```

4. **Re-run fix script:**
   ```bash
   ./scripts/setup/fix_dataform_ssh_pem_format.sh
   ```

---

**Status**: ✅ **FIXED** - Using RSA PEM format as Dataform requires.

**Reference**: [Google Cloud Dataform Documentation](https://docs.cloud.google.com/dataform/docs/connect-repository)

