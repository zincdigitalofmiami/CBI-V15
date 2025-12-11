# Ingestion (Redirect)

Canonical ingestion now lives in `trigger/<Source>/Scripts/` (Python runners + Trigger.dev jobs). This folder remains only as a pointer; no active ingestion logic should live here.

## Canonical Locations

| Source | Path |
|--------|------|
| DataBento | `trigger/DataBento/Scripts/collect_daily.py` |
| FRED | `trigger/FRED/Scripts/collect_fred_*` |
| EIA/EPA | `trigger/EIA_EPA/Scripts/collect_eia_biofuels.py` |
| ScrapeCreators | `trigger/ScrapeCreators/Scripts/collect_news_buckets.py` |
| ProFarmer | `trigger/ProFarmer/Scripts/profarmer_all_urls.ts` / `profarmer_anchor.py` |
| CFTC | `trigger/CFTC/Scripts/ingest_cot.py` |
| USDA | `trigger/USDA/Scripts/ingest_export_sales.py`, `ingest_wasde.py` |
| Weather | `trigger/Weather/Scripts/ingest_weather.py` |
| Cross-source buckets | `trigger/Scripts/collect_all_buckets.py` |

## Notes

- Update any schedulers or docs to point to the trigger paths above.
- Leave this directory free of ingestion logic (only stubs/README files).
