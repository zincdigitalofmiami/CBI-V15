"use client";

export default function LegislationPage() {
  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-thin tracking-wide mb-2">
          Legislation & Policy Risk
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          The laws and rules that move soybean oil: RFS, 45Z, tariffs, LCFS and UCO.
        </p>

        <div className="grid gap-6 md:grid-cols-2 text-[13px] text-slate-300">
          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Biofuels Mandates
            </h3>
            <ul className="space-y-1">
              <li>• U.S. Renewable Fuel Standard (RFS) – RVO volumes, D4/D6 RINs.</li>
              <li>• Section 45Z – Clean Fuel Production Credit (2025–2027).</li>
              <li>• California LCFS – state‑level carbon price for renewable diesel.</li>
            </ul>
          </section>

          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              Trade & Tariffs
            </h3>
            <ul className="space-y-1">
              <li>• Section 301 tariffs and exclusions on China trade.</li>
              <li>• UCO import policy – bans / tariffs vs Chinese used cooking oil.</li>
              <li>• Retaliatory tariffs on U.S. soy exports.</li>
            </ul>
          </section>

          <section className="bg-[#020617] border border-slate-800 rounded-xl p-5 md:col-span-2">
            <h3 className="text-sm font-medium text-slate-200 mb-2">
              What will be tracked here
            </h3>
            <ul className="space-y-1">
              <li>• Daily RVO / RIN curve from EIA + EPA announcements.</li>
              <li>• Policy shock index from ScrapeC news + Trump social feeds.</li>
              <li>• Cross‑impact on ZL: SHAP share for policy vs weather vs macro.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}

