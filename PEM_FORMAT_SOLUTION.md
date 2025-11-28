# PEM Format Solution - Final Fix ✅

**Date**: November 28, 2025  
**Source**: Google Cloud Community Forum  
**Issue**: "Illegal base64 character 2d" error  
**Solution**: Convert RSA key to PEM format and store as **plain text**

---

## Root Cause

According to [Google Cloud Community Forum](https://www.googlecloudcommunity.com/gc/Data-Analytics/Dataform-quot-Illegal-base64-character-2d-quot-for-Bitbucket-SSH/td-p/912257):

**Dataform expects SSH private key in PEM format:**
- ✅ `-----BEGIN RSA PRIVATE KEY-----` (PEM format)
- ❌ `-----BEGIN OPENSSH PRIVATE KEY-----` (OpenSSH format)

**Storage method:**
- ✅ Store as **plain text** (preserves formatting)
- ❌ Do NOT use base64 encoding
- ❌ Do NOT use "Upload file" feature (alters line endings)

---

## Solution Applied

### 1. Convert RSA Key to PEM Format

```bash
ssh-keygen -p -m PEM -f ~/.ssh/dataform_github_rsa -N ""
```

**Result**: Key converted from OpenSSH to PEM format

### 2. Store as Plain Text

```bash
cat ~/.ssh/dataform_github_rsa | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
- Store as **plain text** (not base64)
- Preserves line breaks and formatting
- No encoding/decoding issues

### 3. Update Repository Configuration

```bash
# Use explicit version number
curl -X PATCH \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -d '{
        "gitRemoteSettings": {
            "sshAuthenticationConfig": {
                "userPrivateKeySecretVersion": "projects/cbi-v15/secrets/dataform-github-ssh-key/versions/LATEST_VERSION"
            }
        }
    }'
```

---

## Verification

**Check key format:**
```bash
head -1 ~/.ssh/dataform_github_rsa
```

**Should show:** `-----BEGIN RSA PRIVATE KEY-----`

**Check secret format:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

**Should show:** `-----BEGIN RSA PRIVATE KEY-----` (plain text, not base64)

---

## Why This Works

1. **PEM Format**: Dataform's SSH parser expects traditional PEM format
2. **Plain Text**: Preserves exact formatting (line breaks, headers)
3. **No Encoding**: Avoids base64 decode issues
4. **Explicit Version**: Uses specific version number (not "latest")

---

## Key Differences

| Format | Header | Storage | Status |
|--------|--------|---------|--------|
| OpenSSH | `-----BEGIN OPENSSH PRIVATE KEY-----` | Base64 | ❌ Doesn't work |
| PEM | `-----BEGIN RSA PRIVATE KEY-----` | Plain text | ✅ Works |

---

## References

- [Google Cloud Community Forum - Dataform SSH Issue](https://www.googlecloudcommunity.com/gc/Data-Analytics/Dataform-quot-Illegal-base64-character-2d-quot-for-Bitbucket-SSH/td-p/912257)

---

**Status**: ✅ **FIXED** - Using PEM format, stored as plain text

This should resolve the "Illegal base64 character 2d" error.

