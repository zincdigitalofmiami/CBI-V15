import duckdb from "duckdb";

// Use a single connection pool or open/close per request?
// For serverless (Vercel), we typically open a connection per request or use a global object to cache it if possible (but Vercel lambda freezes).
// The user instruction says: "MD connections must be opened per-request and closed automatically."

export async function queryMotherDuck(sql: string) {
    // Ensure token is present
    if (!process.env.MOTHERDUCK_TOKEN) {
        throw new Error("MOTHERDUCK_TOKEN is not defined");
    }

    // Connect to MotherDuck
    // Note: We use the 'md:' prefix and the token param.
    // We target the 'cbi_v15' database as requested.
    const db = new duckdb.Database(`md:cbi_v15?motherduck_token=${process.env.MOTHERDUCK_TOKEN}`);

    try {
        // Execute query
        return new Promise((resolve, reject) => {
            db.all(sql, (err, res) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(res);
                }
                // Close connection after query
                db.close();
            });
        });
    } catch (error) {
        // Ensure close in case of sync error (unlikely with duckdb node api structure but good practice)
        try { db.close(); } catch { }
        throw error;
    }
}
