# Clean Base64 Solution - Final Fix ✅

**Following Diagnostic Steps**

---

## The Problem

**UI Shows:**
- Using version 20 (RAW PEM)
- Error: "Failed to parse SSH private key"
- Previous error: "Illegal base64 character 2d"

**Root Cause:**
- Version 20 is RAW PEM (contains dashes `-----BEGIN-----`)
- Dataform tries to base64-decode it → hits dash → error
- Need **standard base64** (not base64url, not raw PEM)

---

## Solution Applied

**Created clean base64 version using macOS-compatible method:**

```bash
# Create standard base64 (one line, no newlines) - macOS syntax
cat ~/.ssh/dataform_github_rsa | base64 | tr -d '\n' > /tmp/dataform_github.b64

# Store in Secret Manager
cat /tmp/dataform_github.b64 | \
    gcloud secrets versions add dataform-github-ssh-key \
    --data-file=- \
    --project=cbi-v15
```

**Note:** macOS `base64` doesn't support `-b0` flag, so use `base64 | tr -d '\n'` instead.

**Key Points:**
- ✅ Standard base64 (A-Z, a-z, 0-9, +, /, =)
- ✅ Single line (no newlines)
- ✅ Not base64url (no dashes/underscores)
- ✅ Decodes correctly to PEM

---

## Verification

**Check secret format:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -c 50
```

**Should show:** Base64 string (starts with `LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlKS0...`)

**Verify decode:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -2
```

**Should show:**
```
-----BEGIN RSA PRIVATE KEY-----
MIIJKAIBAAKCAgEAsKcKAWUDYFMBU+3dXvR3x9s+ipBqs1y98jJWYUCiVzrbLn4z
```

**Verify key fingerprint:**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | ssh-keygen -lf /dev/stdin
```

**Should show:** `4096 SHA256:...` (valid key fingerprint)

---

## Current Configuration

**Secret Version**: Latest (clean base64)
**Format**: Standard base64 (one line)
**Repository**: Updated to use latest version

---

## In Dataform UI

**Steps:**
1. **Refresh the page** (to pick up API changes)
2. **Click "Edit Git connection"**
3. **Verify Secret dropdown** shows `dataform-github-ssh-key`
4. **Click "Update"** button
5. **Test connection** - should work now!

**Or:**
- The UI may auto-refresh and pick up the new version
- Just click "Update" to save
- Test connection

---

## Why This Works

**Standard Base64:**
- Uses: A-Z, a-z, 0-9, +, /, =
- **No dashes** (`-`) or underscores (`_`)
- Dataform's decoder can handle it

**Base64url (doesn't work):**
- Uses: A-Z, a-z, 0-9, `-`, `_`
- Has dashes → triggers "Illegal base64 character 2d" error

**Raw PEM (doesn't work):**
- Contains: `-----BEGIN RSA PRIVATE KEY-----`
- Has dashes → Dataform tries to decode → error

---

## ✅ Status

- ✅ Clean base64 created (standard base64, not base64url)
- ✅ Single line (no newlines)
- ✅ Decodes correctly: Verified
- ✅ Key fingerprint: Valid
- ✅ Repository: Updated to latest version

**This should FINALLY resolve the error!**

---

**Test in UI**: Refresh page → Click "Update" → Test connection

The error should be resolved because:
- Secret is **standard base64** (no dashes)
- Dataform decodes it → Gets PEM key
- PEM key is valid and works

