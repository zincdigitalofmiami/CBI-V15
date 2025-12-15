// MotherDuck connection for Next.js API routes
// Uses @motherduck/wasm-client for Vercel serverless compatibility
// DO NOT use native DuckDB - it doesn't work on Vercel!

import { MotherDuckClient } from './motherduck';

/**
 * Query MotherDuck using WASM client (Vercel compatible)
 * 
 * @param sql SQL query to execute
 * @returns Array of row objects with column names as keys
 * 
 * @example
 * ```typescript
 * const rows = await queryMotherDuck('SELECT * FROM forecasts.zl_predictions LIMIT 10');
 * ```
 */
export async function queryMotherDuck(sql: string): Promise<Record<string, unknown>[]> {
    try {
        const result = await MotherDuckClient.query(sql);
        return result.data.toRows() as Record<string, unknown>[];
    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`MotherDuck query failed: ${message}`);
    }
}
