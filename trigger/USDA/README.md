# USDA Ingestion

USDA agricultural data ingestion (WASDE, export sales, crop progress).

## Scripts

- `Scripts/usda_fas_exports.ts` - Weekly export sales (⚠️ Needs creation)
- `Scripts/usda_wasde.ts` - Monthly WASDE reports (⚠️ Needs creation)
- `Scripts/usda_nass_quickstats.ts` - Crop progress data (⚠️ Needs creation)

## Target Tables

- `raw.usda_export_sales` - Weekly export sales
- `raw.usda_wasde` - Supply/demand estimates
- `raw.usda_nass` - Crop progress & conditions

## Schedule

- Export Sales: Weekly (Thursday after release)
- WASDE: Monthly (report date)
- Crop Progress: Weekly (Monday after release)
