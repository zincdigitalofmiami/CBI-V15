#!/usr/bin/env python3
"""
MotherDuck API Proxy
Simple FastAPI server that proxies queries from the Next.js dashboard to MotherDuck
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import duckdb
import os
import json
from decimal import Decimal
from datetime import date, datetime
from typing import List, Dict, Any

app = FastAPI(title="MotherDuck Proxy API")

# Enable CORS for local dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

class QueryRequest(BaseModel):
    sql: str

class QueryResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    rowCount: int
    error: str | None = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "motherduck_connected": bool(MOTHERDUCK_TOKEN),
        "database": MOTHERDUCK_DB
    }

def convert_value(v):
    """Convert database values to JSON-serializable types"""
    if isinstance(v, Decimal):
        return float(v)
    elif isinstance(v, (date, datetime)):
        return v.isoformat()
    return v

@app.post("/query")
async def query(req: QueryRequest):
    """Execute SQL query against MotherDuck"""
    if not MOTHERDUCK_TOKEN:
        raise HTTPException(status_code=500, detail="MOTHERDUCK_TOKEN not configured")

    try:
        conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        result = conn.execute(req.sql).fetchall()

        # Get column names
        columns = [desc[0] for desc in conn.description]

        # Convert to list of dicts with proper type conversion
        data = [
            {k: convert_value(v) for k, v in zip(columns, row)}
            for row in result
        ]

        conn.close()

        return JSONResponse(content={
            "success": True,
            "data": data,
            "rowCount": len(data),
            "error": None
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": [],
                "rowCount": 0,
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
