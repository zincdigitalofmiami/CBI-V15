'use client';

import dynamic from 'next/dynamic';
import { useState } from 'react';

// Dynamic imports for TradingView widgets (client-side only)
const NewsFeedWidget = dynamic(
  () => import('@/app/components/visualizations/tradingview-widgets/NewsFeedWidget'),
  { ssr: false, loading: () => <WidgetLoader /> }
);

const EconomicCalendarWidget = dynamic(
  () => import('@/app/components/visualizations/tradingview-widgets/EconomicCalendarWidget'),
  { ssr: false, loading: () => <WidgetLoader /> }
);

function WidgetLoader() {
  return (
    <div className="h-[400px] bg-[#131722] rounded-lg border border-[#2a2f3e] animate-pulse flex items-center justify-center">
      <span className="text-gray-500 text-sm">Loading widget...</span>
    </div>
  );
}

// Big-4 Driver Cards
const DRIVERS = [
  {
    id: 'lobbying',
    title: 'Lobbying & Policy',
    icon: '🏛️',
    example: '$1.8M biofuel donations → 70% odds 45Z credit → +25% SAF demand',
    impact: '+2.3%',
    confidence: 72,
  },
  {
    id: 'saf',
    title: 'SAF Policy',
    icon: '✈️',
    example: 'India 2% SAF target by 2030 → +5% U.S. soybean oil exports',
    impact: '+1.8%',
    confidence: 65,
  },
  {
    id: 'weather',
    title: 'Weather Risk',
    icon: '🌦️',
    example: '65% La Niña probability → -4% Brazil soybean yield expected',
    impact: '+3.1%',
    confidence: 78,
  },
  {
    id: 'global',
    title: 'Global Events',
    icon: '🌍',
    example: 'China Q4 purchase volumes tracking +12% YoY, renewable fuel mandates expanding',
    impact: '+1.5%',
    confidence: 58,
  },
];

export default function NeuralQuantPage() {
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'ai'; content: string }[]>([
    {
      role: 'ai',
      content:
        "I'm the Crystal Ball AI. Ask me complex questions like: 'How will SAF policies, lobbying, and weather affect soybean oil prices in 2026?'",
    },
  ]);

  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    // Add user message
    setChatHistory((prev) => [...prev, { role: 'user', content: chatInput }]);

    // Simulated AI response (placeholder for actual AI integration)
    setTimeout(() => {
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'ai',
          content: `Analyzing correlations for: "${chatInput}"... This feature will connect to the Crystal Ball AI engine for predictive insights based on the Big-4 drivers.`,
        },
      ]);
    }, 1000);

    setChatInput('');
  };

  return (
    <main className="min-h-screen bg-[#0a0e1a] p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">🔮</span>
            <h1 className="text-3xl font-light text-white">Neural Quant</h1>
          </div>
          <p className="text-gray-400">
            Crystal Ball AI — Drivers of Drivers | Advanced Correlation & Predictive Insights
          </p>
        </header>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column: Big-4 Drivers + Chat */}
          <div className="lg:col-span-2 space-y-6">
            {/* Big-4 Driver Cards */}
            <section>
              <h2 className="text-xl font-light text-white mb-4 flex items-center gap-2">
                <span className="text-[#22c55e]">●</span> Big-4 Unconventional Correlations
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                {DRIVERS.map((driver) => (
                  <div
                    key={driver.id}
                    className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-5 hover:border-[#3a3f4e] transition-colors"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-2xl">{driver.icon}</span>
                        <h3 className="text-white font-medium">{driver.title}</h3>
                      </div>
                      <span className="text-[#22c55e] text-lg font-semibold">{driver.impact}</span>
                    </div>
                    <p className="text-gray-400 text-sm mb-3">{driver.example}</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-[#0a0e1a] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-[#22c55e] to-[#10b981] rounded-full"
                          style={{ width: `${driver.confidence}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500">{driver.confidence}% conf</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Conversational AI Interface */}
            <section className="bg-[#1a1f2e] border border-[#2a2f3e] rounded-lg p-5">
              <h2 className="text-xl font-light text-white mb-4 flex items-center gap-2">
                <span className="text-[#a855f7]">●</span> Reverse Google Search
              </h2>
              <p className="text-gray-500 text-sm mb-4">
                Proactively synthesizes vast information to answer complex, forward-looking questions.
              </p>

              {/* Chat History */}
              <div className="bg-[#0a0e1a] rounded-lg p-4 h-64 overflow-y-auto mb-4 space-y-3">
                {chatHistory.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-lg ${
                        msg.role === 'user'
                          ? 'bg-[#3b82f6] text-white'
                          : 'bg-[#1a1f2e] border border-[#2a2f3e] text-gray-300'
                      }`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))}
              </div>

              {/* Chat Input */}
              <form onSubmit={handleChatSubmit} className="flex gap-3">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask: How will SAF policies affect ZL prices in 2026?"
                  className="flex-1 bg-[#0a0e1a] border border-[#2a2f3e] rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-[#3b82f6]"
                />
                <button
                  type="submit"
                  className="px-6 py-3 bg-gradient-to-r from-[#3b82f6] to-[#8b5cf6] text-white rounded-lg font-medium hover:opacity-90 transition-opacity"
                >
                  Ask
                </button>
              </form>
            </section>
          </div>

          {/* Right Column: News & Calendar */}
          <div className="space-y-6">
            <section>
              <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
                🗞️ Market News
              </h2>
              <NewsFeedWidget height={350} feedMode="all_symbols" />
            </section>

            <section>
              <h2 className="text-lg font-light text-white mb-3 flex items-center gap-2">
                📅 Economic Calendar
              </h2>
              <EconomicCalendarWidget height={350} countryFilter={['US', 'BR', 'CN']} />
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}
