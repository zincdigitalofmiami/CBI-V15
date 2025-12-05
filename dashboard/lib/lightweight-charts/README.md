# Lightweight Charts Package

Reusable chart components for the CBI-V15 Dashboard, built on [Lightweight Charts v5](https://tradingview.github.io/lightweight-charts/).

## Installation

Lightweight Charts is already installed:

```bash
npm install lightweight-charts
```

## Components

### ZLForecastChart

Historical ZL price + multi-horizon forecasts with confidence bands.

**Usage:**

```tsx
import { ZLForecastChart } from '@/lib/lightweight-charts/ZLForecastChart';

<ZLForecastChart
  historical={historicalData}
  forecasts={{
    '1W': { point: 45.2, lower: 44.1, upper: 46.3 },
    '1M': { point: 46.5, lower: 44.8, upper: 48.2 },
    '3M': { point: 48.1, lower: 45.2, upper: 51.0 },
    '6M': { point: 49.3, lower: 45.8, upper: 52.8 },
  }}
  buyZone={{ min: 44, max: 46 }}
  riskZone={{ min: 50, max: 55 }}
/>
```

**Props:**
- `historical`: Array of `{ time: number, value: number }`
- `forecasts`: Object with horizons as keys, each with `point`, `lower`, `upper`
- `buyZone`: Optional `{ min, max }` for shaded buy zone
- `riskZone`: Optional `{ min, max }` for shaded risk zone

---

### SentimentHeatmap

Big-8 bucket sentiment heatmap over time.

**Usage:**

```tsx
import { SentimentHeatmap } from '@/lib/lightweight-charts/SentimentHeatmap';

<SentimentHeatmap
  buckets={['crush', 'china', 'fx', 'fed', 'tariff', 'biofuel', 'energy', 'vol']}
  data={heatmapData}
  lookbacks={['1d', '5d', '20d']}
/>
```

**Props:**
- `buckets`: Array of bucket names
- `data`: 2D array of scores
- `lookbacks`: Array of time window labels

---

### HedgeLadder

Volume ladder chart for hedge strategy visualization.

**Usage:**

```tsx
import { HedgeLadder } from '@/lib/lightweight-charts/HedgeLadder';

<HedgeLadder
  data={[
    { horizon: '1W', volume: 100000, expiry: '2025-12-12' },
    { horizon: '1M', volume: 250000, expiry: '2026-01-15' },
    { horizon: '3M', volume: 500000, expiry: '2026-03-15' },
  ]}
/>
```

**Props:**
- `data`: Array of `{ horizon, volume, expiry }`

---

## Chart Configuration

### Dark Theme (Default)

All charts use a consistent dark theme matching the dashboard:

```ts
{
  layout: {
    background: { color: '#0a0e1a' },
    textColor: '#9ca3af'
  },
  grid: {
    vertLines: { color: '#1f2937' },
    horzLines: { color: '#1f2937' }
  }
}
```

### Responsive Sizing

Charts automatically resize on window resize:

```ts
useEffect(() => {
  const handleResize = () => {
    chart.applyOptions({
      width: container.clientWidth,
      height: container.clientHeight,
    });
  };
  
  window.addEventListener('resize', handleResize);
  return () => window.removeEventListener('resize', handleResize);
}, []);
```

## Best Practices

### 1. Dynamic Imports

Use dynamic imports to avoid SSR issues:

```tsx
'use client';

useEffect(() => {
  import('lightweight-charts').then(({ createChart }) => {
    const chart = createChart(containerRef.current!, options);
    // ...
  });
}, []);
```

### 2. Cleanup

Always remove charts on unmount:

```tsx
return () => {
  chart?.remove();
};
```

### 3. Data Format

Lightweight Charts expects Unix timestamps (seconds):

```ts
const time = Math.floor(new Date(dateString).getTime() / 1000);
```

### 4. Performance

- Limit data points to what's visible (~2000 points max)
- Use `setData()` for initial load, `update()` for incremental
- Debounce resize handlers

## Examples

### Basic Line Chart

```tsx
'use client';

import { useEffect, useRef } from 'react';

export function BasicChart({ data }: { data: { time: number; value: number }[] }) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    import('lightweight-charts').then(({ createChart }) => {
      const chart = createChart(chartRef.current!, {
        layout: { background: { color: '#0a0e1a' }, textColor: '#9ca3af' },
        width: chartRef.current!.clientWidth,
        height: 400,
      });

      const lineSeries = chart.addLineSeries({
        color: '#ffffff',
        lineWidth: 2,
      });

      lineSeries.setData(data);

      return () => chart.remove();
    });
  }, [data]);

  return <div ref={chartRef} className="w-full h-[400px]" />;
}
```

### Candlestick Chart

```tsx
const candleSeries = chart.addCandlestickSeries({
  upColor: '#22c55e',
  downColor: '#ef4444',
  borderVisible: false,
  wickUpColor: '#22c55e',
  wickDownColor: '#ef4444',
});

candleSeries.setData(ohlcData);
```

### Area Chart with Gradient

```tsx
const areaSeries = chart.addAreaSeries({
  topColor: 'rgba(34, 197, 94, 0.4)',
  bottomColor: 'rgba(34, 197, 94, 0.0)',
  lineColor: '#22c55e',
  lineWidth: 2,
});
```

## Troubleshooting

### Chart Not Rendering

- Ensure container has explicit width/height
- Check that data is in correct format (Unix timestamps)
- Verify dynamic import is used (avoid SSR)

### Performance Issues

- Reduce data points
- Use `setData()` instead of multiple `update()` calls
- Debounce resize handlers

### TypeScript Errors

```bash
npm install --save-dev @types/lightweight-charts
```

## Resources

- [Lightweight Charts Docs](https://tradingview.github.io/lightweight-charts/)
- [Examples](https://tradingview.github.io/lightweight-charts/docs/examples)
- [API Reference](https://tradingview.github.io/lightweight-charts/docs/api)
