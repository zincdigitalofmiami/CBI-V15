# Dataform Console URL Guide

**Issue**: URL not found or redirects to sign-in

---

## Correct Dataform URLs

### Option 1: Main Dataform Page
```
https://console.cloud.google.com/dataform?project=cbi-v15
```

### Option 2: Dataform Repositories List
```
https://console.cloud.google.com/dataform/repositories?project=cbi-v15
```

### Option 3: Direct Repository Link
```
https://console.cloud.google.com/dataform/repositories/CBI-V15?project=cbi-v15&location=us-central1
```

---

## If URL Shows Sign-In Page

**Possible causes:**
1. Not authenticated with Google Cloud
2. Wrong Google account
3. No access to project `cbi-v15`

**Solutions:**

### 1. Authenticate with gcloud
```bash
gcloud auth login
gcloud config set project cbi-v15
```

### 2. Verify Project Access
```bash
gcloud projects describe cbi-v15
```

### 3. Check Your Google Account
- Ensure you're signed in with the correct Google account
- The account must have access to `cbi-v15` project

---

## Alternative: Use gcloud CLI

**List repositories:**
```bash
gcloud dataform repositories list \
    --project=cbi-v15 \
    --location=us-central1
```

**Get repository details:**
```bash
gcloud dataform repositories describe CBI-V15 \
    --project=cbi-v15 \
    --location=us-central1
```

---

## Verify Project Setup

**Check project exists:**
```bash
gcloud projects describe cbi-v15
```

**Check Dataform API enabled:**
```bash
gcloud services list --enabled \
    --project=cbi-v15 \
    --filter="name:dataform"
```

**Check repository via API:**
```bash
curl -X GET \
    "https://dataform.googleapis.com/v1beta1/projects/cbi-v15/locations/us-central1/repositories/CBI-V15" \
    -H "Authorization: Bearer $(gcloud auth print-access-token)"
```

---

## Quick Access Steps

1. **Sign in to Google Cloud Console:**
   - Go to: https://console.cloud.google.com
   - Sign in with your Google account

2. **Select Project:**
   - Click project selector (top bar)
   - Select `cbi-v15`

3. **Navigate to Dataform:**
   - Search for "Dataform" in top search bar
   - Or go directly to: https://console.cloud.google.com/dataform?project=cbi-v15

4. **Select Repository:**
   - Click on `CBI-V15` repository
   - Go to Settings → Connect with Git

---

**If still having issues, verify:**
- Project exists: `gcloud projects describe cbi-v15`
- You have access: Check IAM permissions
- Dataform API enabled: Check services list

