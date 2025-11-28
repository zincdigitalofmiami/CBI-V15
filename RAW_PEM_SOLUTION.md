# RAW PEM Solution - The Correct Format ✅

**Dataform expects RAW PEM, NOT base64 encoded!**

---

## The Real Problem

**What We Thought:**
- Dataform wants base64 encoded PEM
- We stored: `LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlKS0...` (base64)

**What Dataform Actually Wants:**
- Dataform wants **RAW PEM** (multi-line, with BEGIN/END headers)
- Should be: 
  ```
  -----BEGIN RSA PRIVATE KEY-----
  MIIJKAIBAAKCAgEAsKcKAWUDYFMBU+3dXvR3x9s+ipBqs1y98jJWYUCiVzrbLn4z
  ...
  -----END RSA PRIVATE KEY-----
  ```

**Why It Failed:**
- Base64 decoding worked ✅
- But then PEM parsing failed ❌
- Because Dataform expects RAW PEM, not base64-wrapped PEM

---

## Solution Applied

**Created clean RAW PEM file:**

```bash
# Copy the key file
cp ~/.ssh/dataform_github_rsa /tmp/dataform_github.pem
chmod 600 /tmp/dataform_github.pem

# Verify format
head -1 /tmp/dataform_github.pem  # Should show: -----BEGIN RSA PRIVATE KEY-----
tail -1 /tmp/dataform_github.pem  # Should show: -----END RSA PRIVATE KEY-----

# Sanity check
ssh-keygen -lf /tmp/dataform_github.pem  # Should show fingerprint

# Store RAW PEM directly (NOT base64 encoded!)
gcloud secrets versions add dataform-github-ssh-key \
    --data-file=/tmp/dataform_github.pem \
    --project=cbi-v15
```

**Key Points:**
- ✅ RAW PEM (multi-line format)
- ✅ Has `-----BEGIN RSA PRIVATE KEY-----` header
- ✅ Has `-----END RSA PRIVATE KEY-----` footer
- ✅ Proper line breaks (64 chars per line)
- ✅ Stored directly (NOT base64 encoded)

---

## Verification

**Check stored version:**
```bash
LATEST_VER=$(gcloud secrets versions list dataform-github-ssh-key \
    --project=cbi-v15 --limit=1 --format="value(name)" | awk -F'/' '{print $NF}')

# Should show BEGIN header
gcloud secrets versions access $LATEST_VER \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1

# Should show END footer
gcloud secrets versions access $LATEST_VER \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | tail -1
```

**Expected Output:**
```
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
```

---

## Current Configuration

**Secret Version**: Latest (RAW PEM format)
**Format**: Multi-line PEM with BEGIN/END headers
**Repository**: Updated to use latest version

---

## In Dataform UI

**Steps:**
1. **Refresh the page** (to pick up API changes)
2. **Click "Edit Git connection"**
3. **Verify Secret dropdown** shows `dataform-github-ssh-key`
4. **Click "Update"** button
5. **Test connection** - should work now!

**Why This Works:**
- Dataform reads the secret
- It's RAW PEM (not base64)
- Dataform parses it directly
- No decoding step needed
- Key is valid and works ✅

---

## Format Comparison

**❌ Wrong (Base64 Encoded):**
```
LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlKS0...
```
- Dataform tries to decode → Gets PEM → Tries to parse → Fails

**✅ Correct (RAW PEM):**
```
-----BEGIN RSA PRIVATE KEY-----
MIIJKAIBAAKCAgEAsKcKAWUDYFMBU+3dXvR3x9s+ipBqs1y98jJWYUCiVzrbLn4z
...
-----END RSA PRIVATE KEY-----
```
- Dataform reads directly → Parses PEM → Works ✅

---

## ✅ Status

- ✅ RAW PEM created (multi-line format)
- ✅ Proper BEGIN/END headers
- ✅ Stored directly (NOT base64 encoded)
- ✅ Key fingerprint verified
- ✅ Repository updated to latest version

**This is the correct format - Dataform should accept it now!**

---

**Test in UI**: Refresh page → Click "Update" → Test connection

The error should be resolved because:
- Secret is **RAW PEM** (what Dataform expects)
- No base64 decoding needed
- PEM parsing should work directly

