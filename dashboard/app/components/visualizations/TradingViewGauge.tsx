"use client";

import { useEffect, useRef } from "react";

interface TradingViewGaugeProps {
  value: number;
  min?: number;
  max?: number;
  label?: string;
  shapImpact?: number;
  color?: string;
  size?: number;
}

export default function TradingViewGauge({
  value,
  min = 0,
  max = 100,
  label = "",
  shapImpact,
  color = "#3b82f6",
  size = 120,
}: TradingViewGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2 - 10;
    const strokeWidth = 3;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // Calculate angle (180 degrees = full semicircle)
    const normalizedValue = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const angle = normalizedValue * Math.PI; // 0 to 180 degrees

    // Draw background arc (dark gray, unfilled portion)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 0, false);
    ctx.strokeStyle = "rgba(100, 100, 100, 0.3)";
    ctx.lineWidth = strokeWidth;
    ctx.stroke();

    // Draw filled arc (colored segment)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, Math.PI - angle, true);
    ctx.strokeStyle = color;
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Draw value indicator dot
    const dotAngle = Math.PI - angle;
    const dotX = centerX + radius * Math.cos(dotAngle);
    const dotY = centerY - radius * Math.sin(dotAngle);
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }, [value, min, max, color, size]);

  return (
    <div className="flex flex-col items-center">
      <canvas ref={canvasRef} width={size} height={size} className="block" />
      {label && <div className="mt-2 text-xs text-slate-400 font-light">{label}</div>}
      <div className="mt-1 text-sm font-medium text-slate-200">{value.toFixed(1)}</div>
      {shapImpact !== undefined && (
        <div className="mt-1 text-xs text-slate-500">
          SHAP: {shapImpact > 0 ? "+" : ""}
          {shapImpact.toFixed(2)}
        </div>
      )}
    </div>
  );
}
