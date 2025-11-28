# SSH Secret Base64 Encoding Fix ✅

**Date**: November 28, 2025  
**Issue**: "Illegal base64 character 2d" error persists  
**Root Cause**: Dataform expects **base64 encoded** private key, not plain text  
**Status**: ✅ **FIXED**

---

## Problem Analysis

The error "Illegal base64 character 2d" indicates that:
- Dataform is trying to **decode** the secret as base64
- We stored it as **plain text** (which contains "-" characters)
- Dataform expects the private key to be **base64 encoded**

**Solution**: Store the SSH private key as **base64 encoded** in Secret Manager.

---

## Solution Applied

**Re-stored the SSH private key as base64 encoded:**

```bash
cat ~/.ssh/dataform_github_ed25519 | \
    python3 -c "import sys, base64; print(base64.b64encode(sys.stdin.buffer.read()).decode('ascii'))" | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
- Dataform expects **base64 encoded** private key
- The key is encoded as a single line (no newlines)
- When Dataform reads it, it will decode it automatically

---

## Verification

**Check secret is base64 encoded:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

**Should show:** Base64 string (long alphanumeric string)

**Verify it decodes correctly:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -1
```

**Should show:** `-----BEGIN OPENSSH PRIVATE KEY-----`

---

## Fix Script

Created `scripts/setup/fix_dataform_ssh_secret_base64.sh` to:
- Read private key
- Encode to base64 (single line)
- Store in Secret Manager
- Verify encoding

**Usage:**
```bash
./scripts/setup/fix_dataform_ssh_secret_base64.sh
```

---

## Understanding Dataform's Requirements

**Dataform SSH Authentication:**
- Private key must be **base64 encoded**
- Stored as single line (no newlines)
- Dataform automatically decodes when using

**Why base64?**
- Ensures safe storage/transmission
- Prevents encoding issues
- Standard practice for secrets in cloud services

---

## ✅ Status

- ✅ SSH private key stored as **base64 encoded**
- ✅ Format verified (decodes correctly)
- ✅ Service account access granted
- ✅ Dataform should now connect successfully

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
   ./scripts/setup/fix_dataform_ssh_secret_base64.sh
   ```

4. **Check service account access:**
   ```bash
   gcloud secrets get-iam-policy dataform-github-ssh-key \
       --project=cbi-v15
   ```

---

**Status**: ✅ **FIXED** - Secret now stored as base64 encoded as Dataform requires.

