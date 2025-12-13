# Chart Package

**All charting logic, wrappers, and visualizations live under this folder.**

This is the canonical home for all chart components and utilities. Do not create chart components elsewhere.

---

## Structure

```
Chart Package/
├── lightweight-charts/     # Core chart hooks/components based on lightweight-charts lib
│   ├── index.ts
│   └── LivePriceMini.tsx
├── nivo/                   # Nivo chart wrappers
│   ├── Gauge.tsx
│   ├── TimeSeriesCharts.tsx
│   ├── WeatherChoropleth.tsx
│   └── index.ts
├── tradingview-widgets/    # TradingView widgets integrations
│   ├── EconomicCalendarWidget.tsx
│   ├── HeatmapEmbed.tsx
│   ├── NewsFeedWidget.tsx
│   ├── SymbolOverviewCard.tsx
│   ├── TechnicalGaugeWidget.tsx
│   ├── TradingViewWidget.tsx
│   └── index.ts
├── TradingViewGauge.tsx   # Shared gauge component
└── Idea Images/           # Design references
```

---

## Usage

Import chart components from this folder:

```typescript
// Lightweight Charts
import { LivePriceMini } from "@/app/Chart Package/lightweight-charts/LivePriceMini";

// Nivo Charts
import { TimeSeriesCharts } from "@/app/Chart Package/nivo/TimeSeriesCharts";
import { Gauge } from "@/app/Chart Package/nivo/Gauge";

// TradingView Widgets
import { TradingViewWidget } from "@/app/Chart Package/tradingview-widgets/TradingViewWidget";
import { EconomicCalendarWidget } from "@/app/Chart Package/tradingview-widgets/EconomicCalendarWidget";

// Shared Components
import TradingViewGauge from "@/app/Chart Package/TradingViewGauge";
```

---

## Principles

1. **Single Source of Truth** - All charting logic lives here
2. **No Duplicates** - Do not create chart components in `components/` or `lib/`
3. **Thin Wrappers Only** - Keep wrapper components minimal; core logic here

---

## Adding New Charts

1. Determine the chart library (lightweight-charts, Nivo, TradingView)
2. Create component in appropriate subfolder
3. Export from `index.ts` in that subfolder
4. Document usage in this README

---

**Last Updated:** December 10, 2025









