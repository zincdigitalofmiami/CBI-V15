export type BandsPoint = { time: number; value: number; upper: number; lower: number };

/**
 * Helper to derive simple symmetric bands around a series using a constant percentage.
 * (We use this for UI until full volatility-derived σ bands are available.)
 */
export function buildPercentBands(
  data: { time: number; value: number }[],
  pct: number,
): BandsPoint[] {
  const p = Math.max(0, pct);
  return data.map((d) => ({
    time: d.time,
    value: d.value,
    upper: d.value * (1 + p),
    lower: d.value * (1 - p),
  }));
}

