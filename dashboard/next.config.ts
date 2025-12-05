import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["duckdb"],
  webpack: (config) => {
    config.externals.push({
      "duckdb": "commonjs duckdb",
    });
    return config;
  },
};

export default nextConfig;
