# Trigger Adapters

Shared utilities and adapters for ingestion jobs.

## Purpose

Reusable helpers for common ingestion patterns:
- HTTP/API clients (auth, pagination, rate limiting)
- ScrapeCreators integration
- Browser/headless/anchor automation flows
- PDF extraction and HTML normalization

## Usage

Import from this folder in your ingestion scripts:

```typescript
import { httpClient } from "../Adapters/http_client";
import { scrapeCreatorsAdapter } from "../Adapters/scrapecreators";
```
