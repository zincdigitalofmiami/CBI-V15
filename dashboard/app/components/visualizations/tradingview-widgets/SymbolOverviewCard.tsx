"use client";

import { memo, useMemo } from "react";
import TradingViewWidget from "./TradingViewWidget";

interface SymbolOverviewCardProps {
  symbol?: string;
  symbolName?: string;
  width?: number | string;
  height?: number;
  showVolume?: boolean;
  showMA?: boolean;
}

/**
 * TradingView Symbol Overview widget configured for dark theme area chart.
 * No candlesticks - line/area style only.
 */
function SymbolOverviewCard({
  symbol = "CBOT:ZL1!",
  symbolName = "Soybean Oil",
  width = "100%",
  height = 300,
  showVolume = false,
  showMA = false,
}: SymbolOverviewCardProps) {
  const config = useMemo(
    () => ({
      symbols: [[symbol, symbolName]],
      chartOnly: false,
      width: "100%",
      height: "100%",
      locale: "en",
      colorTheme: "dark",
      autosize: true,
      showVolume,
      showMA,
      hideDateRanges: false,
      hideMarketStatus: false,
      hideSymbolLogo: false,
      scalePosition: "right",
      scaleMode: "Normal",
      fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
      fontSize: "10",
      noTimeScale: false,
      valuesTracking: "1",
      changeMode: "price-and-percent",
      chartType: "area",
      lineWidth: 2,
      lineType: 0,
      lineColor: "rgba(34, 197, 94, 1)",
      topColor: "rgba(34, 197, 94, 0.3)",
      bottomColor: "rgba(34, 197, 94, 0.0)",
      dateRanges: ["1d|1", "1m|30", "3m|60", "12m|1D", "all|1W"],
    }),
    [symbol, symbolName, showVolume, showMA],
  );

  return (
    <div className="bg-[#131722] rounded-lg p-4 border border-[#2a2f3e]">
      <TradingViewWidget
        scriptSrc="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js"
        config={config}
        height={height}
        width={width}
      />
    </div>
  );
}

export default memo(SymbolOverviewCard);
