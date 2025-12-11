// MotherDuck connection for Next.js API routes (server-side)
// 
// Uses native DuckDB with md: connection string for Vercel serverless compatibility.
// WASM client (lib/motherduck.ts) is available for client-side components in the browser.
//
// For API routes (server-side): Use native DuckDB (this file)
// For client components (browser): Use WASM client (lib/motherduck.ts)

import duckdb from 'duckdb';

const MOTHERDUCK_TOKEN = process.env.MOTHERDUCK_TOKEN;
const MOTHERDUCK_DB = process.env.MOTHERDUCK_DB || 'cbi_v15';

// Connection pool for reuse across requests
let connectionPool: duckdb.Database | null = null;

function getConnection(): duckdb.Connection {
    if (!MOTHERDUCK_TOKEN) {
        throw new Error("MOTHERDUCK_TOKEN is not defined in environment variables");
    }

    // Create database instance if not exists (lightweight, can be reused)
    if (!connectionPool) {
        const connectionString = `md:${MOTHERDUCK_DB}?motherduck_token=${MOTHERDUCK_TOKEN}`;
        connectionPool = new duckdb.Database(connectionString);
    }

    // Create a new connection for each query (connections are lightweight)
    return connectionPool.connect();
}

/**
 * Query MotherDuck using native DuckDB (for server-side API routes)
 * 
 * This uses the native DuckDB Node.js client with MotherDuck's md: protocol.
 * Works in Vercel serverless functions and Next.js API routes.
 * 
 * @param sql SQL query to execute
 * @returns Array of row objects with column names as keys
 * 
 * @example
 * ```typescript
 * const rows = await queryMotherDuck('SELECT * FROM forecasts.zl_predictions LIMIT 10');
 * // Returns: [{ as_of_date: '2025-12-10', horizon: '1w', price_p50: 51.2, ... }, ...]
 * ```
 */
export async function queryMotherDuck(sql: string): Promise<Record<string, unknown>[]> {
    const conn = getConnection();
    
    return new Promise((resolve, reject) => {
        try {
            conn.all(sql, (err: Error | null, rows: unknown[]) => {
                conn.close();
                
                if (err) {
                    reject(new Error(`MotherDuck query failed: ${err.message}`));
                    return;
                }
                
                // DuckDB returns rows as array of objects with column names as keys
                resolve(rows as Record<string, unknown>[]);
            });
        } catch (error: any) {
            conn.close();
            reject(new Error(`MotherDuck query execution failed: ${error.message || error}`));
        }
    });
}
