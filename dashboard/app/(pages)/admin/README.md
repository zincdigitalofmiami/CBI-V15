# Admin / Business Configuration (`/admin`)

## Purpose

Central place for business-level configuration:

- Base procurement volume for $ impact calculations.
- Risk thresholds for red/yellow/green zones on gauges.
- Feature/section visibility toggles.

## Key Components

### 1. App Settings Form

- Base volume (lbs)
- Currency (USD)
- Default contract (ZL front month)

### 2. Risk Thresholds

Sliders or numeric inputs for green/yellow/red per key score:

- China Demand: Green > 70, Yellow 40-70, Red < 40
- Tariff Risk: Green < 30, Yellow 30-60, Red > 60
- Weather Risk: Green < 25, Yellow 25-50, Red > 50

### 3. Visibility Toggles

Enable/disable certain dashboard tiles:

- Show/hide Chris's Four Factors
- Show/hide Big-8 heatmap
- Show/hide Vegas Intel link in nav

## Data Sources (MotherDuck)

- `reference.app_config` (key/value settings).
- Read-only views:
  - `reference.feature_catalog`
  - `reference.model_registry`

## Configuration Schema

### `reference.app_config`

```sql
CREATE TABLE reference.app_config (
  key TEXT PRIMARY KEY,
  value TEXT,
  value_type TEXT, -- 'string', 'number', 'boolean'
  description TEXT,
  updated_at TIMESTAMP
);
```

### Example Rows

```sql
INSERT INTO reference.app_config VALUES
  ('base_volume_lbs', '1000000', 'number', 'Base procurement volume for $ impact calcs'),
  ('currency', 'USD', 'string', 'Display currency'),
  ('china_threshold_green', '70', 'number', 'China demand green threshold'),
  ('show_vegas_intel', 'true', 'boolean', 'Show Vegas Intel in nav');
```

## API Routes

### GET /api/config

Fetch all config values.

### POST /api/config

Update config values (requires auth).

```ts
// Request
{
  "key": "base_volume_lbs",
  "value": "1500000"
}

// Response
{
  "success": true,
  "updated": {
    "key": "base_volume_lbs",
    "value": "1500000",
    "updated_at": "2025-12-05T14:30:00Z"
  }
}
```

## Notes

- Writes should go through API route with auth.
- Keep this page separate from Quant Admin (no model knobs here).
- Changes take effect immediately (no restart required).

## Visual Design

### DashdarkX Theme

- **Background:** `rgb(0, 0, 0)` - pure black
- **Upload cards:** `border-zinc-800` with `font-extralight` titles
- **Refresh buttons:** all `font-extralight`
- **System status:** `font-extralight` throughout
- **Badges:** all include `font-extralight`

### Form Elements

- Inputs: `bg-zinc-950 border-zinc-800`
- Sliders: Custom styled with zinc track
- Toggles: Green for enabled, zinc for disabled

### Threshold Visualization

| Zone   | Color              | Example      |
| ------ | ------------------ | ------------ |
| Green  | `bg-green-500/20`  | Safe zone    |
| Yellow | `bg-yellow-500/20` | Caution zone |
| Red    | `bg-red-500/20`    | Alert zone   |
