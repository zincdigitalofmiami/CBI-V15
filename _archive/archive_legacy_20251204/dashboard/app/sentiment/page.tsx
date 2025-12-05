"use client";

export default function SentimentPage() {
  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-thin tracking-wide mb-2">
          Sentiment & Policy Pulse
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          Daily ScrapeC buckets, FinBERT scores, and Trump / China policy feeds
          summarized for ZL.
        </p>

        <div className="grid gap-6 md:grid-cols-2">
          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Architecture
            </h3>
            <ul className="text-[13px] text-slate-400 space-y-1">
              <li>• ScrapeC buckets: biofuel policy, China demand, tariffs, Trump social</li>
              <li>• FinBERT sentiment mapped to ZL (bullish / bearish / neutral)</li>
              <li>• Daily aggregation into news_bucketed & sentiment_buckets</li>
              <li>• Future: features.sentiment_features_daily → daily_ml_matrix</li>
            </ul>
          </section>

          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              What will be here
            </h3>
            <ul className="text-[13px] text-slate-400 space-y-1">
              <li>• 7d / 30d net sentiment for supply, biofuels, trade, macro</li>
              <li>• Trump policy shock index (trade China, biofuels, ZL‑specific)</li>
              <li>• Rolling correlation of sentiment vs ZL returns</li>
              <li>• SHAP impact of sentiment features in the 1m model</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}

