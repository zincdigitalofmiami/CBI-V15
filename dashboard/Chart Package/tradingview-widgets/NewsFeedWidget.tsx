'use client';

import { memo, useMemo } from 'react';
import TradingViewWidget from './TradingViewWidget';

interface NewsFeedWidgetProps {
  width?: number | string;
  height?: number;
  feedMode?: 'all_symbols' | 'symbol' | 'market';
  symbol?: string;
}

/**
 * TradingView Top Stories/Timeline widget.
 * Live market news headlines in a scrollable list.
 */
function NewsFeedWidget({
  width = '100%',
  height = 400,
  feedMode = 'all_symbols',
  symbol,
}: NewsFeedWidgetProps) {
  const config = useMemo(
    () => ({
      feedMode,
      ...(feedMode === 'symbol' && symbol ? { symbol } : {}),
      market: 'commodity',
      isTransparent: false,
      displayMode: 'regular',
      width: '100%',
      height: '100%',
      locale: 'en',
      colorTheme: 'dark',
    }),
    [feedMode, symbol]
  );

  return (
    <div className="bg-[#131722] rounded-lg border border-[#2a2f3e] overflow-hidden">
      <TradingViewWidget
        scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js"
        config={config}
        height={height}
        width={width}
      />
    </div>
  );
}

export default memo(NewsFeedWidget);
