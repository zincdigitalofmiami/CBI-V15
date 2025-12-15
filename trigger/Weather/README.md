# Weather Ingestion

Weather and climate data from NOAA, INMET (Brazil), SMN (Argentina).

## Scripts

- `Scripts/noaa_weather.ts` - NOAA station data and GFS forecasts (⚠️ Needs creation)
- `Scripts/inmet_brazil.ts` - Brazil weather station data (⚠️ Needs creation)
- `Scripts/smn_argentina.ts` - Argentina weather data (⚠️ Needs creation)

## Target Tables

- `raw.weather_station_obs` - Station observations
- `raw.weather_model_fields` - Forecast model data

## Schedule

- Daily at 2 AM UTC










