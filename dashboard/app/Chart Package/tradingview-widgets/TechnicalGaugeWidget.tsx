'use client';

import { memo, useMemo } from 'react';
import TradingViewWidget from './TradingViewWidget';

interface TechnicalGaugeWidgetProps {
  symbol?: string;
  width?: number | string;
  height?: number;
  showIntervalTabs?: boolean;
  displayMode?: 'single' | 'multiple';
}

/**
 * TradingView Technical Analysis widget in gauge mode.
 * Shows buy/sell sentiment based on oscillators and moving averages.
 */
function TechnicalGaugeWidget({
  symbol = 'CBOT:ZL1!',
  width = '100%',
  height = 400,
  showIntervalTabs = true,
  displayMode = 'multiple',
}: TechnicalGaugeWidgetProps) {
  const config = useMemo(
    () => ({
      interval: '1D',
      width: '100%',
      height: '100%',
      isTransparent: false,
      showIntervalTabs,
      displayMode,
      locale: 'en',
      colorTheme: 'dark',
      symbol,
    }),
    [symbol, showIntervalTabs, displayMode]
  );

  return (
    <div className="bg-[#131722] rounded-lg p-4 border border-[#2a2f3e]">
      <TradingViewWidget
        scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js"
        config={config}
        height={height}
        width={width}
      />
    </div>
  );
}

export default memo(TechnicalGaugeWidget);
