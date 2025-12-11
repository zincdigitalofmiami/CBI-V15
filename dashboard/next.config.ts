import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // DuckDB native package for server-side API routes
  serverExternalPackages: ["duckdb"],
  webpack: (config) => {
    // Externalize DuckDB to avoid bundling native binaries
    config.externals.push({
      "duckdb": "commonjs duckdb",
    });
    return config;
  },
  // COOP/COEP headers required for WASM (MotherDuck WASM client)
  // These enable SharedArrayBuffer and other WASM features in browser/serverless
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "require-corp",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
