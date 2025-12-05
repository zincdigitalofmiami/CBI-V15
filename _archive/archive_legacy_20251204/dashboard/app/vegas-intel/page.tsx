"use client";

export default function VegasIntelPage() {
  return (
    <div className="min-h-screen bg-[#020617] text-slate-100">
      <div className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-3xl font-thin tracking-wide mb-2">
          Vegas Intel
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          Storytelling for PMs and risk committees: what the options market,
          flows, and policy tape are saying about ZL.
        </p>

        <section className="bg-[#020617] border border-slate-800 rounded-2xl p-6 mb-6">
          <h3 className="text-sm font-semibold text-slate-200 mb-3">
            What Vegas Intel Delivers
          </h3>
          <ul className="text-[13px] text-slate-300 space-y-1">
            <li>• Plain‑English explanations of why ZL is moving today.</li>
            <li>• Options‑implied odds on big moves (skew, CVOL, term structure).</li>
            <li>• Cross‑asset tells from ES / CL / HO / FX and macro spreads.</li>
            <li>• Policy and news overlays: RFS, 45Z, tariffs, UCO, China demand.</li>
          </ul>
        </section>

        <section className="bg-[#020617] border border-slate-800 rounded-2xl p-6 mb-6">
          <h3 className="text-sm font-semibold text-slate-200 mb-3">
            How PMs Use It
          </h3>
          <div className="grid gap-4 md:grid-cols-3 text-[13px] text-slate-300">
            <div>
              <h4 className="font-medium text-slate-100 mb-1">Direction</h4>
              <p>
                Daily bias and confidence bands sourced from the ZL forecast
                stack (LightGBM + neural overlays).
              </p>
            </div>
            <div>
              <h4 className="font-medium text-slate-100 mb-1">Conviction</h4>
              <p>
                Sharpe / Sortino, hit rates, and drawdown stats presented as
                simple “risk budget” language.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-slate-100 mb-1">Narrative</h4>
              <p>
                One paragraph that ties together biofuels, weather, FX, and
                tariffs into a trade story.
              </p>
            </div>
          </div>
        </section>

        <section className="bg-[#020617] border border-slate-800 rounded-2xl p-6">
          <h3 className="text-sm font-semibold text-slate-200 mb-3">
            Status
          </h3>
          <p className="text-[13px] text-slate-300">
            Vegas Intel is a presentation layer on top of the live CBI‑V15
            models. As the options and sentiment feeds are wired in, this page
            will populate with real skew, CVOL curves, and “what changed since
            yesterday” cards. For now it serves as a clear spec for PMs and
            sales.
          </p>
        </section>
      </div>
    </div>
  );
}

