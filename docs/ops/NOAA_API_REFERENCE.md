# NOAA Weather API Reference
**Last Updated:** December 16, 2025  
**API Docs:** https://weather-gov.github.io/api/general-faqs  
**OpenAPI Spec:** https://api.weather.gov/openapi.json

---

## Overview

api.weather.gov provides REST-style JSON access to National Weather Service data:
- Current conditions
- Forecasts (7-day, hourly)
- Weather alerts
- Gridded data (2.5km resolution)
- Historical observations

**Authentication:** User-Agent header required (API key system coming)

---

## Getting Forecasts (3-Step Process)

### Step 1: Get Grid Point
```
GET https://api.weather.gov/points/{lat},{lon}
```

**Example:** Washington Monument (38.8894, -77.0352)
```
GET https://api.weather.gov/points/38.8894,-77.0352
```

**Returns:** Grid coordinates + forecast URLs

### Step 2: Extract Forecast URL
From response JSON:
```json
{
  "properties": {
    "forecast": "https://api.weather.gov/gridpoints/LWX/96,70/forecast",
    "forecastHourly": "https://api.weather.gov/gridpoints/LWX/96,70/forecast/hourly"
  }
}
```

### Step 3: Get Forecast
```
GET https://api.weather.gov/gridpoints/LWX/96,70/forecast
```

**Returns:** 7-day forecast in JSON or DWML format

---

## Key Endpoints

### Observations
```
GET /stations/{stationId}/observations/latest
GET /stations/{stationId}/observations
```

### Alerts
```
GET /alerts/active
GET /alerts/active/zone/{zoneId}
```

### Gridpoints
```
GET /gridpoints/{wfo}/{x},{y}
GET /gridpoints/{wfo}/{x},{y}/forecast
GET /gridpoints/{wfo}/{x},{y}/forecast/hourly
```

---

## CBI-V15 Integration

**Current Implementation:**
- `src/ingestion/weather/collect_all_weather.py` - Weather data collector
- `raw.weather_noaa` - Storage table (600 rows currently)
- Locations: US Corn Belt, Brazil (via INMET), Argentina (via SMN)

**Daily Collection:**
```bash
python src/ingestion/weather/collect_all_weather.py
```

---

## Best Practices

1. **Cache grid mappings** - Points rarely change
2. **Use User-Agent header** - Required for all requests
3. **Respect Cache-Control** - API sends proper caching headers
4. **Don't cache bust** - No random query params
5. **GeoJSON format** - Most responses use RFC 7946

---

## Rate Limits

No official rate limits documented, but:
- Use reasonable request frequency
- Cache responses per Cache-Control headers
- Include User-Agent for tracking

---

**Reference:** https://weather-gov.github.io/api/general-faqs
