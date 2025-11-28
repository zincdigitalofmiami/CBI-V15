# Dataform Connection Troubleshooting

**Issue**: "Illegal base64 character 2d" error persists

---

## What We've Tried

1. ✅ Plain text Ed25519 key
2. ✅ Base64 Ed25519 key  
3. ✅ Plain text RSA key
4. ✅ Base64 RSA key (with newlines)
5. ✅ Base64 RSA key (single line, no newlines)
6. ✅ Pure base64 format (A-Z, a-z, 0-9, +, /, = only)
7. ✅ Explicit version number (not "latest")

---

## Current Configuration

**Secret Format**: Pure base64 encoded RSA key
- Single line
- No newlines
- No dashes
- Only base64 characters

**Repository Config**:
- Uses explicit version number
- Host public key set
- SSH authentication configured

---

## Next Steps to Debug

**If error persists, please provide:**

1. **Exact error message** from Dataform UI
2. **When it occurs** (during connection test, compilation, etc.)
3. **Any additional error details**

**To check current state:**

```bash
# Check secret format
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -c 100

# Verify pure base64
gcloud secrets versions access latest \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | \
    grep -qE '^[A-Za-z0-9+/=]+$' && echo "Pure base64" || echo "Has invalid chars"

# Check repository config
curl -X GET \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" | \
    python3 -m json.tool | grep -A 10 "sshAuthenticationConfig"
```

---

## Possible Issues

1. **Dataform caching old secret** - May need to wait or refresh
2. **Secret version mismatch** - Using explicit version now
3. **Service account permissions** - Already verified
4. **Host public key format** - Already set correctly
5. **Dataform UI bug** - May need to disconnect/reconnect

---

**Please share the exact error message for further debugging.**

