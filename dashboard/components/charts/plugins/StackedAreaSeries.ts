export type StackedAreaPoint = { time: number; values: Record<string, number> };

export type StackedAreaStack = {
  id: string;
  name?: string;
  color: string;
  data: { time: number; value: number }[];
};

/**
 * Convert wide "values per key" points into stacked cumulative series suitable for
 * rendering with built-in `AreaSeries`.
 *
 * Notes:
 * - This is a pragmatic implementation that achieves the visual "stacked area" effect
 *   without relying on custom renderers/primitives.
 * - Order matters: earlier keys are stacked first (bottom), later keys are cumulative.
 */
export function buildStackedAreaSeries(
  points: StackedAreaPoint[],
  keysInOrder: { id: string; name?: string; color: string }[],
): StackedAreaStack[] {
  const cumulativeByTime = new Map<number, number>();

  return keysInOrder.map((k) => {
    const seriesData = points.map((p) => {
      const prev = cumulativeByTime.get(p.time) ?? 0;
      const v = p.values[k.id] ?? 0;
      const next = prev + v;
      cumulativeByTime.set(p.time, next);
      return { time: p.time, value: next };
    });
    return { id: k.id, name: k.name, color: k.color, data: seriesData };
  });
}

