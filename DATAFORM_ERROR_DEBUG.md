# Dataform Connection Error Debug

**Status**: Still investigating "Illegal base64 character 2d" error

---

## What We've Verified

✅ **Secret Format**: PEM (`-----BEGIN RSA PRIVATE KEY-----`)
✅ **Storage**: Plain text (version 16)
✅ **GitHub SSH**: Key works when tested directly
✅ **Service Account**: Has access
✅ **Repository Config**: Updated to use version 16

---

## Current Configuration

**Secret Version**: 16
**Format**: PEM (plain text)
**Key Type**: RSA 4096-bit
**On GitHub**: ✅ Yes

**Repository Config**:
- Git URL: `git@github.com:zincdigital/CBI-V15.git`
- Branch: `main`
- Secret Version: `16`
- Host Public Key: Set

---

## Possible Issues

1. **Dataform UI Cache**: May be caching old secret
   - **Solution**: Disconnect and reconnect in UI

2. **Secret Reading**: Dataform may be reading secret differently
   - **Solution**: Verify secret format matches exactly

3. **Workspace Settings**: npm package option may be separate
   - **Note**: npm package config is separate from SSH connection

4. **Timing**: Dataform may need time to refresh
   - **Solution**: Wait a few minutes, then retry

---

## Next Steps

**Please provide:**
1. **Exact error message** from Dataform UI
2. **When it occurs** (connection test, compilation, etc.)
3. **Screenshot** if possible

**To help debug:**
```bash
# Check secret format
gcloud secrets versions access 16 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 | head -5

# Test secret directly
gcloud secrets versions access 16 \
    --secret=dataform-github-ssh-key \
    --project=cbi-v15 > /tmp/test.key
chmod 600 /tmp/test.key
ssh -T git@github.com -i /tmp/test.key
```

---

## About npm Package Option

The npm package configuration option in Dataform is **separate** from the SSH connection. It's for:
- Installing private npm packages during Dataform runs
- Configuring npm registry access
- Not related to Git/SSH connection

**Focus**: Fix SSH connection first, then configure npm packages separately if needed.

---

**Need exact error message to proceed with fix.**

