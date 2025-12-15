/**
 * MotherDuck Client - Shared Database Connection
 * 
 * Provides connection pooling and batch insert utilities for MotherDuck.
 * Used by all Trigger.dev jobs for data ingestion.
 */

import * as duckdb from "duckdb";

export class MotherDuckClient {
  private db: duckdb.Database | null = null;
  private connection: duckdb.Connection | null = null;

  constructor() {
    const token = process.env.MOTHERDUCK_TOKEN;
    const dbName = process.env.MOTHERDUCK_DB || "cbi_v15";

    if (!token) {
      throw new Error("MOTHERDUCK_TOKEN environment variable not set");
    }

    this.db = new duckdb.Database(`:md:${dbName}?motherduck_token=${token}`);
  }

  /**
   * Get or create connection
   */
  private async getConnection(): Promise<duckdb.Connection> {
    if (this.connection) {
      return this.connection;
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error("Database not initialized"));
        return;
      }

      this.db.connect((err, conn) => {
        if (err) {
          reject(err);
          return;
        }
        this.connection = conn;
        resolve(conn);
      });
    });
  }

  /**
   * Execute a SQL query
   */
  async query<T = any>(sql: string, params?: any[]): Promise<T[]> {
    const conn = await this.getConnection();

    return new Promise((resolve, reject) => {
      conn.all(sql, params || [], (err, rows) => {
        if (err) {
          reject(err);
          return;
        }
        resolve(rows as T[]);
      });
    });
  }

  /**
   * Insert batch of records into a table
   * Uses INSERT OR IGNORE for idempotency
   */
  async insertBatch(table: string, records: Record<string, any>[]): Promise<number> {
    if (records.length === 0) {
      return 0;
    }

    const conn = await this.getConnection();

    // Create temp table from records
    const columns = Object.keys(records[0]);
    const values = records.map(r => 
      `(${columns.map(c => this.formatValue(r[c])).join(", ")})`
    ).join(", ");

    const stamp = Date.now();
    const tempName = `staging_${stamp}`;

    const createTempSql = `
      CREATE TEMP TABLE ${tempName} AS 
      SELECT * FROM (VALUES ${values}) AS t(${columns.join(", ")})
    `;

    const insertSql = `
      INSERT OR IGNORE INTO ${table}
      SELECT * FROM ${tempName}
    `;

    return new Promise((resolve, reject) => {
      // Create temp table
      conn.run(createTempSql, (err) => {
        if (err) {
          reject(err);
          return;
        }

        // Insert from temp table
        conn.run(insertSql, function(err) {
          if (err) {
            reject(err);
            return;
          }
          resolve(this.changes || 0);
        });
      });
    });
  }

  /**
   * Format value for SQL
   */
  private formatValue(value: any): string {
    if (value === null || value === undefined) {
      return "NULL";
    }
    if (typeof value === "string") {
      return `'${value.replace(/'/g, "''")}'`;
    }
    if (typeof value === "number") {
      return value.toString();
    }
    if (typeof value === "boolean") {
      return value ? "TRUE" : "FALSE";
    }
    if (value instanceof Date) {
      return `'${value.toISOString()}'`;
    }
    return `'${JSON.stringify(value)}'`;
  }

  /**
   * Close connection
   */
  async close(): Promise<void> {
    if (this.connection) {
      await new Promise<void>((resolve, reject) => {
        this.connection!.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
      this.connection = null;
    }

    if (this.db) {
      await new Promise<void>((resolve, reject) => {
        this.db!.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      });
      this.db = null;
    }
  }
}

