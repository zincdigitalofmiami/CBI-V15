'use client';

type Big8Signal = {
  bucket: string; // Crush, China, FX, Fed, Tariff, Biofuel, Energy, Volatility
  direction: 'bullish' | 'bearish' | 'neutral';
  strength: number; // 0..1
  confidence: number; // 0..1
};

const DEFAULT_SIGNALS: Big8Signal[] = [
  { bucket: 'Crush', direction: 'neutral', strength: 0.5, confidence: 0.6 },
  { bucket: 'China', direction: 'bearish', strength: 0.6, confidence: 0.7 },
  { bucket: 'FX', direction: 'bearish', strength: 0.4, confidence: 0.5 },
  { bucket: 'Fed', direction: 'neutral', strength: 0.3, confidence: 0.5 },
  { bucket: 'Tariff', direction: 'bearish', strength: 0.7, confidence: 0.8 },
  { bucket: 'Biofuel', direction: 'bullish', strength: 0.6, confidence: 0.7 },
  { bucket: 'Energy', direction: 'bullish', strength: 0.5, confidence: 0.6 },
  { bucket: 'Volatility', direction: 'bearish', strength: 0.5, confidence: 0.6 },
];

export default function Big8Panel({ signals = DEFAULT_SIGNALS }: { signals?: Big8Signal[] }) {
  return (
    <section className="w-full bg-[#0a0e1a] border-t border-[#1a1f2e] py-8">
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-white text-xl font-light mb-4">Big 8 Driver Panel</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {signals.map((s) => (
            <div
              key={s.bucket}
              className="rounded-md border border-[#1a1f2e] p-3 flex flex-col items-center gap-2 bg-[#0c1222]"
              title={`${s.bucket}: ${s.direction} • strength ${Math.round(s.strength * 100)}% • confidence ${
                Math.round(s.confidence * 100)
              }%`}
            >
              <span className="text-xs text-gray-400">{s.bucket}</span>
              <span
                className={
                  'text-sm font-medium ' +
                  (s.direction === 'bullish'
                    ? 'text-emerald-400'
                    : s.direction === 'bearish'
                    ? 'text-red-400'
                    : 'text-gray-400')
                }
              >
                {s.direction.toUpperCase()}
              </span>
              <div className="w-full h-2 bg-[#11172a] rounded">
                <div
                  className={
                    'h-2 rounded ' +
                    (s.direction === 'bullish' ? 'bg-emerald-500' : s.direction === 'bearish' ? 'bg-red-500' : 'bg-gray-500')
                  }
                  style={{ width: `${Math.round(s.strength * 100)}%` }}
                />
              </div>
              <span className="text-[10px] text-gray-500">Conf {Math.round(s.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
