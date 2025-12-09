# Bucket-Level News Ingestion

**Purpose:** Direct scraping of premium and public news sources, organized by Big 8 buckets.

**Complementary to:** ScrapeCreators API pipeline (`src/ingestion/scrapecreators/`)

---

## Architecture

### Two-Tier News System:

**Tier 1: ScrapeCreators API** (`src/ingestion/scrapecreators/`)
- Google News Search API
- Truth Social API
- Twitter/X API
- Keyword-based aggregation
- → Stores in `raw.scrapecreators_news_buckets`

**Tier 2: Direct Scraping** (`src/ingestion/buckets/`)
- ProFarmer (premium curated)
- Agrimoney, CONAB, Reuters
- Farm Bureau, State Ag Depts
- Source-specific HTML parsing
- → Stores in `raw.bucket_news`

---

## Folder Structure

```
src/ingestion/buckets/
├── news/
│   └── profarmer_anchor.py      # ProFarmer premium scraper (ANCHOR)
├── china/
│   └── collect_china_news.py    # Agrimoney, CONAB, Reuters
├── tariff/
│   └── collect_tariff_news.py   # Immigration, Farm Bureau, State Ag
├── biofuel/
│   └── collect_biofuel_news.py  # Clean Fuels Alliance, EPA
├── weather/
│   └── collect_weather_news.py  # AgWeb, Farm Progress
├── energy/
│   └── collect_energy_news.py   # Agrimoney grains/oilseeds
├── crush/
│   └── collect_crush_news.py    # NOPA
├── fx/
│   └── collect_fx_news.py       # Reuters FX/commodities
├── fed/
│   └── collect_fed_news.py      # Migration Policy, labor market
└── collect_all_buckets.py       # Master orchestrator
```

---

## ProFarmer Anchor (Premium Curated)

**Why "Anchor"?**
- Primary curated news feed
- High trust score (0.95)
- Daily editions cover all market-moving events
- Maps to multiple Big 8 buckets

**Sections:**
| Section | URL | Edition Type | Timing |
|---------|-----|--------------|--------|
| First Thing Today | `/news/first-thing-today` | `pre_open` | Before market |
| Ahead of the Open | `/news/ahead-of-the-open` | `pre_open` | Before market |
| After the Bell | `/news/after-the-bell` | `post_close` | After market |
| Agriculture News | `/news/agriculture-news` | `intraday` | During day |
| Newsletters | `/newsletters` | `newsletter` | Various |

**Authentication:**
- Requires `PROFARMER_USERNAME` and `PROFARMER_PASSWORD` in `.env`
- Uses `requests.Session()` to maintain login state

**Usage:**
```python
from src.ingestion.buckets.news.profarmer_anchor import fetch_profarmer_articles

articles = fetch_profarmer_articles(days_back=7)
# Returns list of dicts with: article_id, headline, content, author, source, url, edition_type
```

---

## Bucket Collectors

### China Bucket
**Sources:**
- Agrimoney China (`https://www.agrimoney.com/news/china/`)
- CONAB Brazil (`https://www.conab.gov.br/ultimas-noticias`)
- ABIOVE Brazil (`https://abiove.org.br/en/statistics/`)
- Reuters Commodities (China-filtered)

**Keywords:** China, Sinograin, COFCO, soja (Portuguese), exportação

### Tariff Bucket
**Sources:**
- Immigration Impact (`https://immigrationimpact.com/`)
- SPLC Immigrant Justice (`https://www.splcenter.org/issues/immigrant-justice`)
- Farm Bureau (`https://www.fb.org/newsroom/`)
- State Ag Depts (CA, TX, FL, GA)

**Keywords:** trade, tariff, export, import, H-2A visa, farm labor

### Biofuel Bucket
**Sources:**
- Clean Fuels Alliance (`https://cleanfuels.org/`)
- EPA RFS updates
- State biofuel mandates

### Weather Bucket
**Sources:**
- AgWeb Soybeans (`https://www.agweb.com/news/crops/soybeans`)
- Farm Progress (`https://www.farmprogress.com/soybeans`)
- Agriculture.com Markets

### Energy Bucket
**Sources:**
- Agrimoney Grains/Oilseeds (`https://www.agrimoney.com/news/grains-oilseeds/`)
- World Grain (`https://www.world-grain.com/`)

---

## Running the Pipeline

### Test Individual Bucket:
```bash
# China bucket
python src/ingestion/buckets/china/collect_china_news.py

# Tariff bucket
python src/ingestion/buckets/tariff/collect_tariff_news.py

# ProFarmer (requires credentials)
python src/ingestion/buckets/news/profarmer_anchor.py
```

### Run All Buckets:
```bash
export $(grep -v '^#' .env | xargs)
python src/ingestion/buckets/collect_all_buckets.py
```

---

## Database Schema

**Table:** `raw.bucket_news`

**Columns:**
- `date` - Article date
- `article_id` - MD5 hash of URL (primary key)
- `headline` - Article title
- `content` - Article body
- `author` - Author name (if available)
- `source` - Source name ("ProFarmer", "Agrimoney China", etc.)
- `source_trust_score` - 0.0-1.0 (premium sources = 0.95)
- `url` - Source URL (for deduplication)
- `bucket_name` - Big 8 bucket ("china", "tariff", etc.)
- `edition_type` - ProFarmer edition type (NULL for non-ProFarmer)
- `published_at` - Publication timestamp
- `created_at` - Ingestion timestamp

**Deduplication:**
- By URL (keeps highest trust score)
- `INSERT OR IGNORE` on re-runs

---

## Configuration

**File:** `config/ingestion/bucket_sources.yaml`

Maps each source to buckets and defines scraping parameters.

---

## Next Steps

**TODO - Remaining Buckets:**
- [ ] Biofuel bucket collector
- [ ] Weather bucket collector
- [ ] Energy bucket collector
- [ ] Crush bucket collector
- [ ] FX bucket collector
- [ ] Fed bucket collector

**TODO - Enhancements:**
- [ ] Full article body scraping (currently just snippets)
- [ ] Playwright for JavaScript-heavy sites
- [ ] Rate limiting/politeness delays
- [ ] Error retry logic
- [ ] Monitoring/alerting

---

**Last Updated:** December 9, 2024

