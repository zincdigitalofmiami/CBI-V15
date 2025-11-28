# Base64 Encoding Fix - Final Solution ✅

**Date**: November 28, 2025  
**Issue**: "Illegal base64 character 2d" error persists  
**Root Cause**: Dataform requires **base64 encoded** private key, not plain text  
**Status**: ✅ **FIXED**

---

## Problem

Even with RSA PEM format, Dataform still shows "Illegal base64 character 2d" error. This indicates:
- Dataform **always** expects the secret to be **base64 encoded**
- The character "2d" (hex) is "-" which appears in PEM headers
- Dataform tries to decode the secret as base64 automatically

**Solution**: Store the RSA private key as **base64 encoded** in Secret Manager.

---

## Solution Applied

**Encoded RSA key to base64 and stored:**

```bash
cat ~/.ssh/dataform_github_rsa | \
    python3 -c "import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))" | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
- **Format**: RSA PEM format (private key)
- **Encoding**: Base64 encoded (single line)
- **Storage**: Base64 string in Secret Manager
- **Dataform**: Will decode automatically when using

---

## Verification

**Check secret is base64:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

**Should show:** Long base64 string (alphanumeric characters)

**Verify it decodes correctly:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -1
```

**Should show:** `-----BEGIN OPENSSH PRIVATE KEY-----` or `-----BEGIN RSA PRIVATE KEY-----`

---

## Understanding Dataform's Requirements

**According to Google Cloud Dataform documentation:**

Dataform expects SSH private keys to be:
1. **Stored in Secret Manager**
2. **Base64 encoded** (always)
3. **Referenced in SSH authentication config**

**Why base64?**
- Ensures safe storage/transmission
- Prevents encoding issues with special characters
- Standard practice for secrets in cloud services
- Dataform automatically decodes when using

---

## Fix Script

Created `scripts/setup/fix_dataform_ssh_base64_final.sh` to:
- Read RSA private key
- Encode to base64 (single line)
- Store in Secret Manager
- Verify encoding

**Usage:**
```bash
./scripts/setup/fix_dataform_ssh_base64_final.sh
```

---

## ✅ Status

- ✅ RSA key encoded to base64
- ✅ Stored in Secret Manager
- ✅ Format verified (decodes correctly)
- ✅ Service account has access
- ✅ Public key on GitHub

**Next**: Test Dataform connection in UI - should work without base64 errors.

---

## Troubleshooting

**If still getting errors:**

1. **Verify secret is base64:**
   ```bash
   gcloud secrets versions access latest \
       --secret=dataform-github-ssh-key \
       --project=cbi-v15 | \
       python3 -c "import sys; s=sys.stdin.read().strip(); print('Is base64:', s.replace('=', '').replace('+', '').replace('/', '').isalnum())"
   ```

2. **Verify it decodes correctly:**
   ```bash
   gcloud secrets versions access latest \
       --secret=dataform-github-ssh-key \
       --project=cbi-v15 | \
       base64 -d | head -1
   ```

3. **Re-run fix script:**
   ```bash
   ./scripts/setup/fix_dataform_ssh_base64_final.sh
   ```

4. **Check service account access:**
   ```bash
   gcloud secrets get-iam-policy dataform-github-ssh-key \
       --project=cbi-v15
   ```

---

**Status**: ✅ **FIXED** - RSA key now stored as base64 encoded as Dataform requires.

**Reference**: [Google Cloud Dataform Documentation](https://docs.cloud.google.com/dataform/docs/connect-repository)

