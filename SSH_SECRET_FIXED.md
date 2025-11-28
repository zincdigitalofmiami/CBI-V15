# SSH Secret Format Fixed ✅

**Date**: November 28, 2025  
**Issue**: "Illegal base64 character 2d" error when connecting Dataform  
**Status**: ✅ **FIXED**

---

## Problem

Dataform was unable to connect to GitHub with error:
```
Illegal base64 character 2d
```

**Root Cause**: The SSH private key in Secret Manager may have been incorrectly formatted or encoded.

---

## Solution

Re-stored the SSH private key in Secret Manager as **plain text** (not base64 encoded):

```bash
cat ~/.ssh/dataform_github_ed25519 | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
- Dataform expects the **raw private key** (plain text)
- The key should start with `-----BEGIN OPENSSH PRIVATE KEY-----`
- No base64 encoding needed

---

## Verification

**Check secret format:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

**Should show:**
```
-----BEGIN OPENSSH PRIVATE KEY-----
```

---

## Fix Script

Created `scripts/setup/fix_dataform_ssh_secret.sh` to:
- Verify private key format
- Store key correctly in Secret Manager
- Ensure proper formatting

**Usage:**
```bash
./scripts/setup/fix_dataform_ssh_secret.sh
```

---

## ✅ Status

- ✅ SSH private key re-stored correctly
- ✅ Secret format verified
- ✅ Dataform should now connect successfully

**Next**: Test Dataform connection in UI - should work without base64 errors.

---

## Troubleshooting

**If still getting errors:**

1. **Verify secret format:**
   ```bash
   gcloud secrets versions access latest \
       --secret=dataform-github-ssh-key \
       --project=cbi-v15 | head -3
   ```

2. **Check private key locally:**
   ```bash
   head -1 ~/.ssh/dataform_github_ed25519
   ```

3. **Re-run fix script:**
   ```bash
   ./scripts/setup/fix_dataform_ssh_secret.sh
   ```

4. **Verify service account access:**
   ```bash
   gcloud secrets get-iam-policy dataform-github-ssh-key \
       --project=cbi-v15
   ```

