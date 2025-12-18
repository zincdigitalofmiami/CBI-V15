"use client";

import { memo, useEffect, useRef, useId } from "react";

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
  const widgetRef = useRef<HTMLDivElement>(null);
  const uniqueId = useId().replace(/:/g, "_");
  const actualContainerId = containerId || `tv_widget_${uniqueId}`;

  useEffect(() => {
    if (!containerRef.current) return;

    // Clear any existing content
    containerRef.current.innerHTML = "";

    // Create the widget container div that TradingView expects
    const widgetContainer = document.createElement("div");
    widgetContainer.className = "tradingview-widget-container__widget";
    widgetContainer.id = actualContainerId;
    widgetContainer.style.height = "100%";
    widgetContainer.style.width = "100%";
    containerRef.current.appendChild(widgetContainer);

    // Create and inject the script
    const script = document.createElement("script");
    script.src = scriptSrc;
    script.async = true;
    script.type = "text/javascript";

    // Merge default dark theme with provided config
    // For advanced chart, we need container_id
    const mergedConfig = {
      colorTheme: "dark",
      isTransparent: false,
      width: "100%",
      height: "100%",
      container_id: actualContainerId,
      ...config,
    };

    script.innerHTML = JSON.stringify(mergedConfig);
    containerRef.current.appendChild(script);

    return () => {
      // Cleanup on unmount
      if (containerRef.current) {
        containerRef.current.innerHTML = "";
      }
    };
  }, [scriptSrc, config, actualContainerId]);

  return (
    <div
      ref={containerRef}
      className="tradingview-widget-container rounded-lg overflow-hidden"
      style={{
        height: typeof height === "number" ? `${height}px` : height,
        width: typeof width === "number" ? `${width}px` : width,
      }}
    />
  );
}

export default memo(TradingViewWidget);
