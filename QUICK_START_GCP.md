# Quick Start: GCP Setup

**Date**: November 28, 2025  
**Project**: `cbi-v15`

---

## 🚀 One-Command Setup (Recommended)

```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_gcp_project.sh
```

**What happens**:
1. Creates GCP project `cbi-v15` (if needed)
2. Enables all required APIs
3. Creates 8 BigQuery datasets (us-central1 only)
4. Creates service account
5. Grants permissions

**You'll need to**:
- Link billing account when prompted
- Run API key storage script after

---

## 📋 Step-by-Step

### 1. Run GCP Setup
```bash
cd /Users/zincdigital/CBI-V15
./scripts/setup/setup_gcp_project.sh
```

### 2. Store API Keys
```bash
./scripts/setup/store_api_keys.sh
```
Choose **Option 3** (Both Keychain + Secret Manager)

### 3. Verify Everything Works
```bash
python scripts/setup/verify_connections.py
```

### 4. Initialize Dataform
```bash
cd dataform
npm install -g @dataform/cli
npm install
dataform init
dataform compile
```

---

## ✅ Success Checklist

- [ ] GCP project `cbi-v15` created
- [ ] Billing account linked
- [ ] All APIs enabled
- [ ] 8 BigQuery datasets created (us-central1)
- [ ] Service account created
- [ ] API keys stored (Keychain + Secret Manager)
- [ ] Verification script passes
- [ ] Dataform initialized

---

## 📚 Full Documentation

See [docs/setup/GCP_SETUP.md](docs/setup/GCP_SETUP.md) for detailed instructions.

---

## 🆘 Troubleshooting

**"Permission denied"**
```bash
gcloud auth login
```

**"Billing not enabled"**
- Link billing: https://console.cloud.google.com/billing

**"Dataset already exists"**
- This is OK - script skips existing datasets

---

**Ready to start?** Run the setup script above! 🚀

