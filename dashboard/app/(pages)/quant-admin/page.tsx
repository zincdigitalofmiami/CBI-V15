
export default function QuantAdminPage() {
    return (
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Quant Admin</h1>
            <p className="mb-4">
                This is a restored stub for the Quant Admin dashboard.
                Real-time training metrics and model performance will appear here once the full pipeline is active.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="border p-4 rounded shadow">
                    <h2 className="font-semibold">Latest Run</h2>
                    <p>No runs found.</p>
                </div>
                <div className="border p-4 rounded shadow">
                    <h2 className="font-semibold">System Status</h2>
                    <p>MotherDuck: Connected</p>
                    <p>Ingestion: Pending</p>
                </div>
            </div>
        </div>
    );
}
