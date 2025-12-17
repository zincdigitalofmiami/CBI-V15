"use client";

type Props = {
  value?: number; // 0..1
  label?: string;
};

export default function ConfidenceBadge({ value = 0.72, label = "Prediction Confidence" }: Props) {
  const pct = Math.round((value ?? 0) * 100);
  const color = pct >= 75 ? "text-emerald-400" : pct >= 55 ? "text-yellow-400" : "text-red-400";
  const ring =
    pct >= 75 ? "ring-emerald-500/30" : pct >= 55 ? "ring-yellow-500/30" : "ring-red-500/30";

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#0c1222] ring-1 ${ring}`}
    >
      <span className="text-[10px] uppercase tracking-wider text-gray-400">{label}</span>
      <span className={`text-sm font-medium ${color}`}>{pct}%</span>
    </div>
  );
}
