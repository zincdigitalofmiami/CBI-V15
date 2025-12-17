"use client";

import { memo, useEffect, useRef } from "react";

interface TradingViewWidgetProps {
  scriptSrc: string;
  config: Record<string, unknown>;
  containerId?: string;
  height?: number | string;
  width?: number | string;
}

/**
 * Base wrapper component for TradingView embed widgets.
 * Handles script injection, cleanup, and dark theme configuration.
 */
function TradingViewWidget({
  scriptSrc,
  config,
  containerId,
  height = 400,
  width = "100%",
}: TradingViewWidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptRef = useRef<HTMLScriptElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear any existing content
    containerRef.current.innerHTML = "";

    // Create the widget container div that TradingView expects
    const widgetContainer = document.createElement("div");
    widgetContainer.className = "tradingview-widget-container__widget";
    containerRef.current.appendChild(widgetContainer);

    // Create and inject the script
    const script = document.createElement("script");
    script.src = scriptSrc;
    script.async = true;
    script.type = "text/javascript";

    // Merge default dark theme with provided config
    const mergedConfig = {
      colorTheme: "dark",
      isTransparent: false,
      width: "100%",
      height: "100%",
      ...config,
    };

    script.innerHTML = JSON.stringify(mergedConfig);
    containerRef.current.appendChild(script);
    scriptRef.current = script;

    return () => {
      // Cleanup on unmount
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    };
  }, [scriptSrc, config]);

  return (
    <div
      ref={containerRef}
      id={containerId}
      className="tradingview-widget-container rounded-lg overflow-hidden"
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        width: typeof width === "number" ? `${width}px` : width,
      }}
    />
  );
}

export default memo(TradingViewWidget);
