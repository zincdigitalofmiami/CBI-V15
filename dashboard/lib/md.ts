// MotherDuck connection using WASM client for Vercel serverless compatibility
// Note: The native DuckDB binary doesn't work on Vercel due to GLIBC requirements

const MOTHERDUCK_TOKEN = process.env.MOTHERDUCK_TOKEN;

interface QueryResult {
    data: {
        toRows: () => Record<string, unknown>[];
    };
}

export async function queryMotherDuck(sql: string): Promise<Record<string, unknown>[]> {
    if (!MOTHERDUCK_TOKEN) {
        throw new Error("MOTHERDUCK_TOKEN is not defined");
    }

    // Use MotherDuck's HTTP API which is compatible with serverless
    const response = await fetch('https://api.motherduck.com/v1/sql', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${MOTHERDUCK_TOKEN}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            database: 'cbi_v15',
            sql: sql,
        }),
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`MotherDuck query failed: ${error}`);
    }

    const result = await response.json();

    // MotherDuck API returns { data: { columns: [...], rows: [[...], ...] } }
    if (result.data && result.data.columns && result.data.rows) {
        const columns = result.data.columns;
        return result.data.rows.map((row: unknown[]) => {
            const obj: Record<string, unknown> = {};
            columns.forEach((col: string, i: number) => {
                obj[col] = row[i];
            });
            return obj;
        });
    }

    return [];
}
