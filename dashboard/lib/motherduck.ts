import { MDConnection } from '@motherduck/wasm-client';

export class MotherDuckClient {
    private static instance: MDConnection | null = null;

    private constructor() { }

    public static async getConnection(): Promise<MDConnection> {
        if (!MotherDuckClient.instance) {
            const token = process.env.MOTHERDUCK_TOKEN;
            if (!token) {
                throw new Error('MOTHERDUCK_TOKEN is not defined');
            }

            MotherDuckClient.instance = MDConnection.create({
                mdToken: token,
            });

            await MotherDuckClient.instance.isInitialized();
        }

        return MotherDuckClient.instance;
    }

    public static async query(sql: string) {
        const connection = await MotherDuckClient.getConnection();
        return connection.evaluateQuery(sql);
    }
}
