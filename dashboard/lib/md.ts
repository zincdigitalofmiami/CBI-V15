// MotherDuck connection for Next.js API routes
// Uses backend proxy API for MotherDuck queries

const PROXY_URL = process.env.MOTHERDUCK_PROXY_URL || "http://localhost:8000";

/**
 * Query MotherDuck via backend proxy API
 *
 * @param sql SQL query to execute
 * @returns Array of row objects with column names as keys
 *
 * @example
 * ```typescript
 * const rows = await queryMotherDuck('SELECT * FROM raw.databento_futures_ohlcv_1d LIMIT 10');
 * ```
 */
export async function queryMotherDuck(sql: string): Promise<Record<string, unknown>[]> {
  try {
    const response = await fetch(`${PROXY_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ sql }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();

    if (!result.success) {
      throw new Error(result.error || "Query failed");
    }

    return result.data;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`MotherDuck query failed: ${message}`);
  }
}
