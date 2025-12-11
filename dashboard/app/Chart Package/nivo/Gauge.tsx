"use client";

interface GaugeProps {
  value: number; // 0 = strong sell, 1 = strong buy
  label: string;
  subtitle?: string;
  minLabel?: string;
  maxLabel?: string;
}

export function Gauge({
  value,
  label,
  subtitle,
  minLabel = "Strong sell",
  maxLabel = "Strong buy",
}: GaugeProps) {
  const clamped = Math.min(1, Math.max(0, value));
  const angle = -120 + clamped * 240;
  const radius = 80;
  const centerX = 100;
  const centerY = 100;

  const pointerLength = radius - 14;
  const pointerRad = (angle * Math.PI) / 180;
  const pointerX = centerX + pointerLength * Math.cos(pointerRad);
  const pointerY = centerY + pointerLength * Math.sin(pointerRad);

  const labelText =
    clamped <= 0.2
      ? "Strong sell"
      : clamped <= 0.4
      ? "Sell"
      : clamped < 0.6
      ? "Neutral"
      : clamped < 0.8
      ? "Buy"
      : "Strong buy";

  return (
    <div className="rounded-2xl bg-black/80 border border-slate-800 p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-slate-300">{label}</div>
        {subtitle && (
          <div className="text-[10px] uppercase tracking-wide text-slate-500">
            {subtitle}
          </div>
        )}
      </div>
      <svg width={200} height={120} viewBox="0 0 200 120">
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="50%" stopColor="#eab308" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>

        <path
          d="M 40 100 A 70 70 0 0 1 160 100"
          fill="none"
          stroke="#111827"
          strokeWidth={14}
          strokeLinecap="round"
        />

        <path
          d="M 40 100 A 70 70 0 0 1 160 100"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth={10}
          strokeLinecap="round"
        />

        <text x={40} y={112} fontSize={9} fill="#6b7280" textAnchor="start">
          {minLabel}
        </text>
        <text x={100} y={112} fontSize={9} fill="#6b7280" textAnchor="middle">
          Neutral
        </text>
        <text x={160} y={112} fontSize={9} fill="#6b7280" textAnchor="end">
          {maxLabel}
        </text>

        <line
          x1={centerX}
          y1={centerY}
          x2={pointerX}
          y2={pointerY}
          stroke="#e5e7eb"
          strokeWidth={3}
          strokeLinecap="round"
        />
        <circle cx={centerX} cy={centerY} r={4} fill="#e5e7eb" />

        <text
          x={centerX}
          y={centerY - 4}
          fontSize={11}
          fill="#e5e7eb"
          textAnchor="middle"
        >
          {labelText}
        </text>
      </svg>
    </div>
  );
}

