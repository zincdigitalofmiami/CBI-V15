'use client';

import { memo, useMemo } from 'react';
import TradingViewWidget from './TradingViewWidget';

interface EconomicCalendarWidgetProps {
  width?: number | string;
  height?: number;
  countryFilter?: string[];
  importanceFilter?: string;
}

/**
 * TradingView Economic Calendar widget.
 * Shows upcoming macro events with impact ratings.
 */
function EconomicCalendarWidget({
  width = '100%',
  height = 400,
  countryFilter = ['US', 'CA', 'BR', 'CN'],
  importanceFilter = '-1,0,1', // low, medium, high
}: EconomicCalendarWidgetProps) {
  const config = useMemo(
    () => ({
      width: '100%',
      height: '100%',
      isTransparent: false,
      locale: 'en',
      colorTheme: 'dark',
      importanceFilter,
      countryFilter: countryFilter.join(','),
    }),
    [countryFilter, importanceFilter]
  );

  return (
    <div className="bg-[#131722] rounded-lg border border-[#2a2f3e] overflow-hidden">
      <TradingViewWidget
        scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-events.js"
        config={config}
        height={height}
        width={width}
      />
    </div>
  );
}

export default memo(EconomicCalendarWidget);
