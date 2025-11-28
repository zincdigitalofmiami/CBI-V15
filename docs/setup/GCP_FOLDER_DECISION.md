# GCP Folder Decision - CONFIRMED ✅

**Date**: November 28, 2025  
**Decision**: ✅ **YES** - CBI-V15 will be created under "App Development" folder

---

## Confirmation

### Current Organization:
- ✅ **cbi-v14** is already under "App Development" folder (`568609080192`)
- ✅ **CBI-V15** will be created under the same folder for consistency

### Folder Details:
- **Folder Name**: App Development
- **Folder ID**: `568609080192`
- **Location**: Same as cbi-v14 (for organizational consistency)

---

## Updated Setup

The `setup_gcp_project.sh` script has been updated to:
1. ✅ Create `cbi-v15` under App Development folder
2. ✅ Move existing project to App Development folder if it exists elsewhere
3. ✅ Verify folder placement after creation

---

## Project Creation Command

**Automatic** (via script):
```bash
./scripts/setup/setup_gcp_project.sh
```

**Manual** (if needed):
```bash
gcloud projects create cbi-v15 \
  --name="CBI-V15 Soybean Oil Forecasting" \
  --folder=568609080192
```

---

## Verification

After project creation, verify folder placement:
```bash
gcloud projects describe cbi-v15 --format="get(parent)"
```

**Expected output**: `id=568609080192;type=folder`

---

## Status

✅ **CONFIRMED** - CBI-V15 will be created under "App Development" folder  
✅ **CONSISTENT** - Same folder as cbi-v14  
✅ **READY** - Setup script updated

---

**Last Updated**: November 28, 2025

