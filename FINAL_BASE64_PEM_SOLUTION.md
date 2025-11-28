# Final Solution: Base64 Encoded PEM Format ✅

**Date**: November 28, 2025  
**Error**: "Illegal base64 character 2d"  
**Root Cause**: Dataform expects **base64 encoded** key, but we stored PEM as **plain text**  
**Solution**: Base64 encode the PEM format key

---

## The Real Issue

The error "Illegal base64 character 2d" means:
- **Character 2d** (hex) = `-` (dash)
- Dataform **IS trying to decode as base64**
- We stored PEM format as **plain text** (contains dashes in `-----BEGIN-----`)
- Dataform expects **base64 encoded** key

**Key Insight**: 
- ✅ PEM format is correct (`-----BEGIN RSA PRIVATE KEY-----`)
- ✅ But must be **base64 encoded** before storing
- ❌ Not plain text PEM

---

## Solution Applied

**Base64 encode the PEM format key:**

```bash
cat ~/.ssh/dataform_github_rsa | \
    python3 -c "import sys, base64; \
        key_data = sys.stdin.buffer.read(); \
        b64_key = base64.b64encode(key_data).decode('ascii'); \
        b64_key = b64_key.replace('\n', '').replace('\r', ''); \
        print(b64_key)" | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Key Points:**
1. **Format**: PEM (`-----BEGIN RSA PRIVATE KEY-----`)
2. **Encoding**: Base64 encoded (single line)
3. **Storage**: Base64 string in Secret Manager
4. **Dataform**: Will decode automatically

---

## Verification

**Check secret is base64:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -c 100
```

**Should show:** Base64 string (starts with `LS0tLS1CRUdJTi...`)

**Verify it decodes to PEM:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -1
```

**Should show:** `-----BEGIN RSA PRIVATE KEY-----`

---

## Why This Works

1. **PEM Format**: Dataform's SSH parser expects PEM format
2. **Base64 Encoded**: Dataform expects base64 encoded secret
3. **Single Line**: No newlines in base64 string
4. **Pure Base64**: Only A-Z, a-z, 0-9, +, /, = characters

**Process:**
1. Key: PEM format (`-----BEGIN RSA PRIVATE KEY-----`)
2. Encode: Base64 encode the PEM key
3. Store: Base64 string in Secret Manager
4. Dataform: Decodes base64 → Gets PEM key → Uses for SSH

---

## What We Learned

**The confusion:**
- ❌ Plain text PEM (has dashes, Dataform tries to decode as base64 → error)
- ❌ Base64 OpenSSH (wrong format, Dataform can't parse)
- ✅ **Base64 PEM** (correct format, correct encoding)

**Dataform Requirements:**
1. SSH key must be **PEM format** (`-----BEGIN RSA PRIVATE KEY-----`)
2. Secret must be **base64 encoded** (not plain text)
3. Base64 must be **single line** (no newlines)
4. Must be **pure base64** (only valid base64 characters)

---

## ✅ Status

- ✅ PEM format key: Converted
- ✅ Base64 encoded: Applied
- ✅ Format verified: Pure base64, decodes correctly
- ✅ Repository: Updated to latest version

**This should FINALLY resolve the error!**

---

**Test in UI**: https://console.cloud.google.com/dataform?project=cbi-v15

The "Illegal base64 character 2d" error should be resolved.

