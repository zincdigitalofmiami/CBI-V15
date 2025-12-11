'use client';

import { memo, useMemo } from 'react';
import TradingViewWidget from './TradingViewWidget';

interface HeatmapEmbedProps {
  width?: number | string;
  height?: number;
  market?: 'forex' | 'crypto' | 'stock';
}

/**
 * TradingView Forex/Crypto Heatmap widget for macro sentiment visualization.
 * Dark themed grid showing percentage changes across currency pairs.
 */
function HeatmapEmbed({
  width = '100%',
  height = 400,
  market = 'forex',
}: HeatmapEmbedProps) {
  const config = useMemo(
    () => ({
      exchanges: [],
      dataSource: market === 'forex' ? 'ForexCross' : 'Crypto',
      grouping: 'sector',
      blockSize: 'market_cap_basic',
      blockColor: 'change',
      locale: 'en',
      symbolUrl: '',
      colorTheme: 'dark',
      hasTopBar: false,
      isDataSet498Enabled: true,
      isZoomEnabled: true,
      hasSymbolTooltip: true,
      width: '100%',
      height: '100%',
    }),
    [market]
  );

  // Forex heatmap uses a different widget
  const scriptSrc =
    market === 'forex'
      ? 'https://s3.tradingview.com/external-embedding/embed-widget-forex-heat-map.js'
      : 'https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js';

  return (
    <div className="bg-[#131722] rounded-lg border border-[#2a2f3e] overflow-hidden">
      <TradingViewWidget
        scriptSrc={scriptSrc}
        config={config}
        height={height}
        width={width}
      />
    </div>
  );
}

export default memo(HeatmapEmbed);
