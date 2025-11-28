# Testing Both Formats - RAW PEM vs Base64

**Based on Diagnostic Guidance**

---

## The Issue

**Error**: "Illegal base64 character 2d"
- Dataform is trying to base64-decode the secret
- If secret contains dashes (`-----BEGIN-----`), it's not valid base64
- **Question**: Does Dataform expect RAW PEM or base64-encoded PEM?

---

## Testing Both Formats

### Version 19: Base64 Encoded PEM
- **Format**: Base64 string (no dashes)
- **Decodes to**: `-----BEGIN RSA PRIVATE KEY-----`
- **Status**: Available for testing

### Latest Version: RAW PEM
- **Format**: Plain text PEM (has dashes)
- **Starts with**: `-----BEGIN RSA PRIVATE KEY-----`
- **Status**: Just created for testing

---

## What to Test

**In Dataform UI:**
1. Try connection with latest version (RAW PEM)
2. If error persists, switch to version 19 (base64)
3. See which format works

**The format that works is what Dataform expects.**

---

## Current Configuration

**Repository is using**: Latest version (RAW PEM format)

**To switch to base64 version:**
```bash
curl -X PATCH \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15?updateMask=gitRemoteSettings" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -d '{
        "gitRemoteSettings": {
            "sshAuthenticationConfig": {
                "userPrivateKeySecretVersion": "projects/cbi-v15/secrets/dataform-github-ssh-key/versions/19"
            }
        }
    }'
```

---

## Verification Commands

**Check what version repository is using:**
```bash
curl -X GET \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" | \
    python3 -m json.tool | grep "userPrivateKeySecretVersion"
```

**Test version 19 (base64):**
```bash
gcloud secrets versions access 19 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    base64 -d | head -1
```

**Test latest version (raw PEM):**
```bash
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -1
```

---

## Next Steps

1. **Test in UI** with current version (RAW PEM)
2. **If error persists**: Switch to version 19 (base64)
3. **Report which format works**

---

**Status**: Testing RAW PEM format first (latest version)

