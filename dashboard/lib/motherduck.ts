// MotherDuck WASM Client
// Provides full DuckDB functionality in Vercel serverless environments
// Uses @motherduck/wasm-client for browser/serverless compatibility

import { MDConnection } from "@motherduck/wasm-client";

const MOTHERDUCK_DB = process.env.MOTHERDUCK_DB || "cbi_v15";
const MOTHERDUCK_ATTACH_ALIAS = "md_db";

export class MotherDuckClient {
  private static instance: MDConnection | null = null;
  private static isAttached: boolean = false;

  private constructor() {}

  /**
   * Attach the MotherDuck database and make it the default.
   *
   * We attach as `md_db` then `USE md_db` so callers can query with `schema.table`
   * (e.g. `raw.databento_futures_ohlcv_1d`) without needing a database prefix.
   */
  private static async ensureAttached(connection: MDConnection): Promise<void> {
    if (MotherDuckClient.isAttached) {
      return;
    }

    try {
      await connection.evaluateQuery(`ATTACH 'md:${MOTHERDUCK_DB}' AS ${MOTHERDUCK_ATTACH_ALIAS}`);
    } catch (attachError: any) {
      const message = attachError?.message || String(attachError);
      const alreadyAttached =
        message.includes("already exists") ||
        message.includes("already attached") ||
        message.includes("Catalog Error: Database with name");
      if (!alreadyAttached) {
        throw attachError;
      }
    }

    // Make the attached database the default so `schema.table` resolves correctly.
    await connection.evaluateQuery(`USE ${MOTHERDUCK_ATTACH_ALIAS}`);
    MotherDuckClient.isAttached = true;
  }

  /**
   * Get or create MotherDuck WASM connection
   * Uses singleton pattern to reuse connection across requests
   */
  public static async getConnection(): Promise<MDConnection> {
    if (!MotherDuckClient.instance) {
      const token =
        process.env.MOTHERDUCK_TOKEN || process.env.motherduck_storage_MOTHERDUCK_TOKEN;
      if (!token) {
        throw new Error(
          "MotherDuck token is not defined (expected MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN)",
        );
      }

      try {
        MotherDuckClient.instance = MDConnection.create({
          mdToken: token,
        });

        await MotherDuckClient.instance.isInitialized();
        await MotherDuckClient.ensureAttached(MotherDuckClient.instance);
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
      await MotherDuckClient.ensureAttached(connection);

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
