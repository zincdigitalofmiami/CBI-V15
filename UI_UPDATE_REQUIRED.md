# UI Update Required - Use Version 19

**Based on UI Screenshot Analysis**

---

## Current Situation

**UI Shows:**
- Secret Version: 20 (RAW PEM format)
- Error: "Illegal base64 character 2d"
- Status: Connection failing

**Root Cause:**
- Dataform **IS trying to base64-decode** the secret
- Version 20 is RAW PEM (has dashes `-----BEGIN-----`)
- Dataform tries to decode it → hits dash → error

**Solution:**
- Use version 19 (base64 encoded PEM)
- Repository API updated to version 19
- **UI needs to be refreshed/updated**

---

## What to Do in UI

### Step 1: Refresh the Page
- The repository API has been updated to version 19
- Refresh the Dataform UI page
- The secret version should update automatically

### Step 2: Verify Secret Version
- Check the "Secret" dropdown
- Should show: `dataform-github-ssh-key`
- Version should be: 19 (or "latest" which points to 19)

### Step 3: Re-select Secret (if needed)
- Open the "Secret" dropdown
- Select `dataform-github-ssh-key` again
- This ensures UI picks up version 19

### Step 4: Click "Update"
- Click the blue "Update" button
- This saves the configuration

### Step 5: Test Connection
- The error should be resolved
- Connection should work

---

## API Status

**Repository Configuration (via API):**
- Secret Version: 19 ✅
- Format: Base64 encoded PEM ✅
- Decodes correctly: Verified ✅

**The UI may need to:**
- Refresh to pick up API changes
- Re-select the secret to ensure version 19 is used

---

## Verification

**Check what API says:**
```bash
curl -X GET \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" | \
    python3 -m json.tool | grep "userPrivateKeySecretVersion"
```

**Should show:** `versions/19`

---

## If Error Persists

**Try in UI:**
1. Click "Edit Git connection"
2. Open "Secret" dropdown
3. Select `dataform-github-ssh-key` (should use version 19)
4. Click "Update"
5. Test connection

**Or disconnect and reconnect:**
1. Click "Delete connection"
2. Click "Edit Git connection" again
3. Configure with same settings
4. Secret should use latest version (19)

---

**Status**: API updated to version 19. UI may need refresh or re-selection of secret.

