// MotherDuck connection for Next.js API routes
//
// IMPORTANT:
// - Vercel serverless environments do not have a localhost proxy.
// - We query MotherDuck using the WASM client, which requires a runtime that
//   provides Web Worker APIs. Therefore, any API route importing this helper
//   must run in the Edge runtime (`export const runtime = "edge"`).
// - To avoid `next build` crashes, we dynamically import the WASM client so it
//   is not evaluated during the Node.js build step.

const MOTHERDUCK_DB = process.env.MOTHERDUCK_DB || "cbi_v15";
const MOTHERDUCK_ATTACH_ALIAS = "md_db";

function getToken(): string | null {
  // Prefer canonical name; support Vercel MotherDuck integration naming; accept a few legacy aliases.
  const token =
    process.env.MOTHERDUCK_TOKEN ||
    process.env.motherduck_storage_MOTHERDUCK_TOKEN ||
    process.env.MOTHERDUCK_STORAGE_TOKEN ||
    process.env.motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN ||
    process.env.MOTHERDUCK_READ_SCALING_TOKEN;
  return token || null;
}

type MDConnection = {
  isInitialized: () => Promise<void>;
  evaluateQuery: (sql: string) => Promise<{ data: { toRows: () => Record<string, unknown>[] } }>;
};

let _connPromise: Promise<MDConnection> | null = null;

async function getConnection(): Promise<MDConnection> {
  if (_connPromise) return _connPromise;

  _connPromise = (async () => {
    const token = getToken();
    if (!token) {
      throw new Error(
        "MotherDuck token is not defined (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN in Vercel env vars)",
      );
    }

    // Dynamic import to avoid Node build-time Worker errors.
    const mod = await import("@motherduck/wasm-client");
    const conn = (mod as any).MDConnection.create({ mdToken: token }) as MDConnection;

    await conn.isInitialized();

    // Attach DB and set default
    try {
      await conn.evaluateQuery(`ATTACH 'md:${MOTHERDUCK_DB}' AS ${MOTHERDUCK_ATTACH_ALIAS}`);
    } catch (attachError: any) {
      const message = attachError?.message || String(attachError);
      const alreadyAttached =
        message.includes("already exists") ||
        message.includes("already attached") ||
        message.includes("Catalog Error: Database with name");
      if (!alreadyAttached) throw attachError;
    }

    await conn.evaluateQuery(`USE ${MOTHERDUCK_ATTACH_ALIAS}`);
    return conn;
  })();

  return _connPromise;
}

/**
 * Query MotherDuck directly via native DuckDB (MotherDuck `:md:` connection)
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
    const conn = await getConnection();
    const result = await conn.evaluateQuery(sql);
    return result.data.toRows();
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`MotherDuck query failed: ${message}`);
  }
}
