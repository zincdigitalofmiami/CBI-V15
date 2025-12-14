"""
Seed: 33 Canonical Symbols
Loads the official symbol list into reference.symbols
"""
import duckdb
from datetime import datetime

# 33 canonical symbols for V15
SYMBOLS = {
    # Agricultural (11)
    "ZL": {"name": "Soybean Oil", "exchange": "CME", "category": "agricultural", "primary": True},
    "ZS": {"name": "Soybeans", "exchange": "CME", "category": "agricultural"},
    "ZM": {"name": "Soybean Meal", "exchange": "CME", "category": "agricultural"},
    "ZC": {"name": "Corn", "exchange": "CME", "category": "agricultural"},
    "ZW": {"name": "Wheat (Chicago)", "exchange": "CME", "category": "agricultural"},
    "KE": {"name": "Wheat (Kansas City)", "exchange": "CME", "category": "agricultural"},
    "ZO": {"name": "Oats", "exchange": "CME", "category": "agricultural"},
    "CT": {"name": "Cotton", "exchange": "ICE", "category": "agricultural"},
    "KC": {"name": "Coffee", "exchange": "ICE", "category": "agricultural"},
    "SB": {"name": "Sugar", "exchange": "ICE", "category": "agricultural"},
    "CC": {"name": "Cocoa", "exchange": "ICE", "category": "agricultural"},
    
    # Energy (4)
    "CL": {"name": "Crude Oil (WTI)", "exchange": "CME", "category": "energy"},
    "HO": {"name": "Heating Oil", "exchange": "CME", "category": "energy"},
    "RB": {"name": "RBOB Gasoline", "exchange": "CME", "category": "energy"},
    "NG": {"name": "Natural Gas", "exchange": "CME", "category": "energy"},
    
    # Metals (5)
    "GC": {"name": "Gold", "exchange": "CME", "category": "metals"},
    "SI": {"name": "Silver", "exchange": "CME", "category": "metals"},
    "HG": {"name": "Copper", "exchange": "CME", "category": "metals"},
    "PA": {"name": "Palladium", "exchange": "CME", "category": "metals"},
    "PL": {"name": "Platinum", "exchange": "CME", "category": "metals"},
    
    # Treasuries (3)
    "ZN": {"name": "10-Year T-Note", "exchange": "CME", "category": "treasuries"},
    "ZB": {"name": "30-Year T-Bond", "exchange": "CME", "category": "treasuries"},
    "ZF": {"name": "5-Year T-Note", "exchange": "CME", "category": "treasuries"},
    
    # FX (9)
    "6A": {"name": "Australian Dollar", "exchange": "CME", "category": "fx"},
    "6B": {"name": "British Pound", "exchange": "CME", "category": "fx"},
    "6C": {"name": "Canadian Dollar", "exchange": "CME", "category": "fx"},
    "6E": {"name": "Euro", "exchange": "CME", "category": "fx"},
    "6J": {"name": "Japanese Yen", "exchange": "CME", "category": "fx"},
    "6M": {"name": "Mexican Peso", "exchange": "CME", "category": "fx"},
    "6N": {"name": "New Zealand Dollar", "exchange": "CME", "category": "fx"},
    "6S": {"name": "Swiss Franc", "exchange": "CME", "category": "fx"},
    "DX": {"name": "US Dollar Index", "exchange": "ICE", "category": "fx"},
    
    # Palm (1)
    "FCPO": {"name": "Palm Oil (Malaysia)", "exchange": "BMD", "category": "agricultural"},
}


def seed_symbols(conn: duckdb.DuckDBPyConnection) -> int:
    """Insert symbols into reference.symbols table."""
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reference.symbols (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            exchange VARCHAR,
            category VARCHAR,
            is_primary BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert symbols
    count = 0
    for symbol, meta in SYMBOLS.items():
        conn.execute("""
            INSERT INTO reference.symbols (symbol, name, exchange, category, is_primary)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                exchange = EXCLUDED.exchange,
                category = EXCLUDED.category,
                is_primary = EXCLUDED.is_primary
        """, [symbol, meta["name"], meta["exchange"], meta["category"], meta.get("primary", False)])
        count += 1
    
    return count


if __name__ == "__main__":
    import os
    
    # Connect to MotherDuck or local DuckDB
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        conn = duckdb.connect(f"md:cbi_v15?motherduck_token={token}")
    else:
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    count = seed_symbols(conn)
    print(f"Seeded {count} symbols")
    conn.close()

