// MotherDuck WASM Client
// Provides full DuckDB functionality in Vercel serverless environments
// Uses @motherduck/wasm-client for browser/serverless compatibility

import { MDConnection } from '@motherduck/wasm-client';

const MOTHERDUCK_DB = process.env.MOTHERDUCK_DB || 'cbi_v15';

export class MotherDuckClient {
    private static instance: MDConnection | null = null;

    private constructor() { }

    /**
     * Get or create MotherDuck WASM connection
     * Uses singleton pattern to reuse connection across requests
     */
    public static async getConnection(): Promise<MDConnection> {
        if (!MotherDuckClient.instance) {
            const token = process.env.MOTHERDUCK_TOKEN;
            if (!token) {
                throw new Error('MOTHERDUCK_TOKEN is not defined in environment variables');
            }

            try {
                MotherDuckClient.instance = MDConnection.create({
                    mdToken: token,
                });

                await MotherDuckClient.instance.isInitialized();
            } catch (error: any) {
                throw new Error(`Failed to initialize MotherDuck WASM client: ${error.message || error}`);
            }
        }

        return MotherDuckClient.instance;
    }

    /**
     * Execute SQL query against MotherDuck database
     * 
     * The WASM client requires ATTACH to connect to a specific database.
     * Queries should use fully qualified table names (schema.table).
     * 
     * @param sql SQL query string (use schema.table format, e.g., 'SELECT * FROM forecasts.zl_predictions')
     * @returns Query result object with { data: { toRows: () => Record<string, unknown>[] } }
     * 
     * @example
     * ```typescript
     * const result = await MotherDuckClient.query('SELECT * FROM forecasts.zl_predictions LIMIT 10');
     * const rows = result.data.toRows();
     * ```
     */
    public static async query(sql: string) {
        const connection = await MotherDuckClient.getConnection();
        
        try {
            // ATTACH database (token is handled by connection-level mdToken)
            // ATTACH is idempotent - safe to call multiple times
            // Use fully qualified names (schema.table) in queries
            try {
                await connection.evaluateQuery(`ATTACH 'md:${MOTHERDUCK_DB}' AS md_db`);
            } catch (attachError: any) {
                // If ATTACH fails (e.g., already attached), continue with query
                // Some errors are expected if database is already attached
                if (!attachError.message?.includes('already exists') && 
                    !attachError.message?.includes('already attached')) {
                    // Re-throw if it's a different error
                    throw attachError;
                }
            }
            
            // Execute the actual query
            const result = await connection.evaluateQuery(sql);
            return result;
        } catch (error: any) {
            throw new Error(`MotherDuck query execution failed: ${error.message || error}`);
        }
    }

    /**
     * Close the connection (useful for cleanup)
     */
    public static async close(): Promise<void> {
        if (MotherDuckClient.instance) {
            try {
                await MotherDuckClient.instance.close();
            } catch (error) {
                // Ignore close errors
            }
            MotherDuckClient.instance = null;
        }
    }
}
