# API Endpoints

**Status:** Production  
**Last Updated:** December 3, 2025

## MotherDuck API Routes

All endpoints connect to MotherDuck via `MOTHERDUCK_TOKEN` environment variable.

### `/api/motherduck/forecast`

**Method:** GET

**Query Parameters:**
- `horizon` - Required: `1W`, `1M`, `3M`, `6M`, `12M`
- `date` - Optional: specific forecast date (default: latest)

**Response:**
```json
{
  "success": true,
  "data": {
    "forecast_date": "2025-12-03",
    "target_date": "2025-12-10",
    "horizon": "1W",
    "forecast_value": 45.23,
    "lower_90": 43.12,
    "upper_90": 47.34,
    "model_name": "Ensemble (AutoETS 0.5 + ARIMA 0.3 + Theta 0.2)",
    "confidence": 0.87
  }
}
```

### `/api/motherduck/buckets`

**Method:** GET

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "bucket": "biofuel",
      "score": 0.72,
      "sentiment": "bullish",
      "mape": 3.21,
      "last_update": "2025-12-03T17:05:00Z"
    },
    ...
  ]
}
```

### `/api/motherduck/shap`

**Method:** GET

**Query Parameters:**
- `horizon` - Required: `1W`, `1M`, `3M`, `6M`, `12M`
- `date` - Optional: specific date (default: latest)

**Response:**
```json
{
  "success": true,
  "data": {
    "date": "2025-12-03",
    "horizon": "1M",
    "drivers": [
      {"feature": "eia_rin_price_d4", "shap_value": 11.2, "rank": 1},
      {"feature": "weather_argentina_drought_zscore", "shap_value": 6.8, "rank": 2},
      ...
    ]
  }
}
```

### `/api/motherduck/live-zl`

**Method:** GET

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "ZL",
    "date": "2025-12-03",
    "close": 45.67,
    "volume": 12345,
    "change_pct": 1.23,
    "last_update": "2025-12-03T16:00:00Z"
  }
}
```

### `/api/quant-reports`

**Method:** GET

**Auth:** Public (Chris-visible)

**Response:** Data quality and forecast performance metrics

### `/api/quant-admin`

**Method:** POST (PIN validation)

**Auth:** 4-digit PIN required

**Body:**
```json
{"pin": "1234"}
```

**Response:** Session token (24-hour cookie)

## Rate Limits

- Forecast endpoints: 100 req/min
- Live ZL: 1000 req/min (cached 5 minutes)
- SHAP: 50 req/min (compute-heavy)

## Error Responses

```json
{
  "success": false,
  "error": "Error message",
  "code": "MOTHERDUCK_CONNECTION_ERROR"
}
```

