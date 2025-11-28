# CBI-V15 Quick Start Guide

**Get started in 5 minutes**

---

## 🚀 Quick Start (5 Steps)

### Step 1: Connect Dataform to GitHub
1. Go to: [Google Cloud Console → Dataform](https://console.cloud.google.com/dataform)
2. Click **"Connect Repository"**
3. Enter:
   - Repository: `zincdigital/CBI-V15`
   - Branch: `main`
   - **Root Directory: `dataform/`** ⚠️ Critical
4. Click **"Connect"**

### Step 2: Store API Keys
```bash
./scripts/setup/store_api_keys.sh
```
Enter your API keys when prompted (Databento, ScrapeCreators, etc.)

### Step 3: Test Connections
```bash
python3 scripts/ingestion/test_connections.py
```
Should show: ✅ BigQuery connected

### Step 4: First Data Ingestion
```bash
python3 src/ingestion/databento/collect_daily.py
```
This collects price data and loads it to BigQuery.

### Step 5: Run Dataform
```bash
cd dataform
npx dataform compile  # Verify compilation
npx dataform run --tags staging  # Build staging tables
npx dataform run --tags features  # Build feature tables
```

---

## ✅ That's It!

You now have:
- ✅ Data flowing into BigQuery
- ✅ Staging tables built
- ✅ Feature tables built
- ✅ Ready for model training

---

## 📚 Next Steps

- **Export Training Data**: `python3 scripts/export/export_training_data.py`
- **Train Models**: `python3 src/training/baselines/lightgbm_zl.py`
- **Set Up Automation**: See `DEPLOYMENT_GUIDE.md`

---

## 🆘 Troubleshooting

**Dataform won't connect?**
- Verify Root Directory is `dataform/` (not `/`)
- Check GitHub repository access
- Refresh the Dataform page

**API keys not found?**
- Run `./scripts/setup/store_api_keys.sh` again
- Check Keychain: `security find-generic-password -s DATABENTO_API_KEY`

**Ingestion fails?**
- Verify API keys are stored
- Check BigQuery permissions
- Review error logs

---

**Need Help?** See `DEPLOYMENT_GUIDE.md` for detailed instructions.
