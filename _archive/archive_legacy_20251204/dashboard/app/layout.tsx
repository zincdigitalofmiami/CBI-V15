import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "U.S. Oil Solutions – Intelligence Platform",
  description: "CBI-V15 Dashboard for Soybean Oil (ZL) and macro drivers.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const navItems = [
    { href: "/", label: "Dashboard" },
    { href: "/sentiment", label: "Sentiment" },
    { href: "/strategy", label: "Strategy" },
    { href: "/legislation", label: "Legislation" },
    { href: "/vegas-intel", label: "Vegas Intel" },
    { href: "/admin", label: "Admin" },
  ];

  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#020617] text-slate-100`}
      >
        <div className="min-h-screen flex">
          {/* Sidebar */}
          <aside className="w-64 border-r border-slate-800 bg-[#020617] flex flex-col">
            <div className="px-5 pt-5 pb-4 border-b border-slate-800">
              <h1 className="text-lg font-thin tracking-wide text-slate-100">
                U.S. Oil Solutions
              </h1>
              <p className="text-[11px] text-slate-500 mt-1 font-light">
                Intelligence Platform • CBI‑V15
              </p>
            </div>
            <nav className="flex-1 px-3 py-4 space-y-1 text-sm">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-2 px-3 py-2 rounded-md text-slate-300 hover:bg-slate-800/60 hover:text-slate-50 transition-colors font-light"
                >
                  <span>{item.label}</span>
                </Link>
              ))}
            </nav>
            <div className="px-5 py-3 border-t border-slate-800 text-[10px] text-slate-500">
              <p>Data: Databento • FRED • USDA • EIA • NOAA</p>
            </div>
          </aside>

          {/* Main content */}
          <main className="flex-1 overflow-y-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}
