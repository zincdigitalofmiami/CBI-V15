# Correct Solution - Base64 Encoded PEM Key ✅

**Based on Diagnostic Guidance**

---

## The Issue

**Error**: "Illegal base64 character 2d"
- Character `2d` (hex) = `-` (dash)
- Dataform **IS trying to base64-decode** the secret
- This means Dataform **EXPECTS base64 encoded** key

**Root Cause**: 
- We stored PEM key as **plain text** (contains dashes in `-----BEGIN-----`)
- Dataform tries to decode it as base64 → hits dash → error

---

## The Solution

**Store PEM key as base64 encoded:**

1. **Key Format**: PEM (`-----BEGIN RSA PRIVATE KEY-----`)
2. **Storage Format**: Base64 encoded (Dataform will decode it)
3. **Result**: Dataform decodes base64 → Gets PEM key → Uses for SSH

---

## Current Configuration (Version 19)

**Secret**: `dataform-github-ssh-key` (version 19)
- **Format**: Base64 encoded PEM key
- **Length**: 4324 characters
- **Decodes to**: `-----BEGIN RSA PRIVATE KEY-----`
- **Valid**: ✅ Yes (can be used for SSH)

**Repository Config**:
- Uses version 19
- SSH authentication configured
- Host public key set

---

## Verification

**Check secret is base64:**
```bash
gcloud secrets versions access 19 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -c 50
```

**Should show:** Base64 string (starts with `LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlKS0...`)

**Verify it decodes correctly:**
```bash
gcloud secrets versions access 19 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -1
```

**Should show:** `-----BEGIN RSA PRIVATE KEY-----`

**Test the decoded key:**
```bash
gcloud secrets versions access 19 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d > /tmp/test.pem
chmod 600 /tmp/test.pem
ssh-keygen -l -f /tmp/test.pem
```

**Should show:** Key fingerprint (proves it's valid)

---

## Why This Works

**Dataform's Process:**
1. Reads secret from Secret Manager (version 19)
2. **Decodes base64** → Gets PEM key
3. Uses PEM key for SSH authentication
4. Connects to GitHub

**Our Configuration:**
1. PEM key: `-----BEGIN RSA PRIVATE KEY-----` ✅
2. Base64 encode: `LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlKS0...` ✅
3. Store in Secret Manager ✅
4. Dataform decodes → Gets PEM → Works ✅

---

## Key Points

✅ **PEM Format**: Required (not OpenSSH format)
✅ **Base64 Encoded**: Required (Dataform decodes it)
✅ **Single Line**: Base64 should be single line (no newlines)
✅ **Pure Base64**: Only A-Z, a-z, 0-9, +, /, = characters

---

## ✅ Status

- ✅ PEM key: Converted to PEM format
- ✅ Base64 encoded: Applied correctly
- ✅ Decodes correctly: Verified
- ✅ Repository: Updated to version 19

**This should resolve the error!**

---

**Test in UI**: https://console.cloud.google.com/dataform?project=cbi-v15

The "Illegal base64 character 2d" error should be resolved because:
- Secret is base64 encoded (no dashes in base64 string)
- Dataform decodes it → Gets PEM key (with dashes, but that's OK - it's decoded)
- PEM key works for SSH authentication

