# CBI-V15 Setup - Quick Reference

**Start Here**: `COMPLETE_SETUP_EXECUTION_GUIDE.md`

---

## 🚀 Quick Start

```bash
cd /Users/zincdigital/CBI-V15

# Run complete setup
./scripts/setup/pre_flight_check.sh
./scripts/setup/setup_gcp_project.sh
./scripts/setup/setup_iam_permissions.sh
./scripts/setup/setup_bigquery_skeleton.sh
./scripts/setup/store_api_keys.sh
python3 scripts/setup/verify_connections.py
cd dataform && npm install && dataform compile
```

---

## 📚 Documentation

- **Complete Guide**: `COMPLETE_SETUP_EXECUTION_GUIDE.md`
- **Checklist**: `EXECUTION_CHECKLIST.md`
- **IAM Guide**: `docs/setup/IAM_PERMISSIONS_GUIDE.md`
- **BigQuery Guide**: `docs/setup/BIGQUERY_SETUP_EXECUTION.md`

---

## ✅ Status

**100% Ready** - All scripts and documentation complete

---

**Last Updated**: November 28, 2025

