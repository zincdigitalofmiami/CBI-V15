#!/usr/bin/env python3
"""
Collect Vegas Events from Glide API
Daily ingestion at 3 AM UTC
"""

import sys
from pathlib import Path
from datetime import datetime
import requests
from google.cloud import bigquery
from google.cloud import secretmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

PROJECT_ID = "cbi-v15"
DATASET_ID = "raw"
TABLE_ID = "vegas_events"

def get_api_key():
    """Get Glide API key from Secret Manager"""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/glide-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def collect_vegas_events():
    """Collect Vegas events from Glide API"""
    api_key = get_api_key()
    
    # Glide API endpoint for events
    url = "https://api.glide.app/api/v1/events"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"Collecting Vegas events from Glide API...")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data or "data" not in data:
            print("⚠️  No events data returned")
            return []
        
        events = data.get("data", [])
        print(f"✅ Collected {len(events)} events")
        
        return events
        
    except Exception as e:
        print(f"❌ Error collecting events: {e}")
        raise

def save_to_bigquery(events):
    """Save events to BigQuery"""
    if not events:
        print("⚠️  No events to save")
        return
    
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    
    # Prepare data
    rows = []
    for event in events:
        row = {
            "event_id": event.get("id"),
            "event_name": event.get("name"),
            "event_date": event.get("date"),
            "event_type": event.get("type"),
            "venue": event.get("venue"),
            "description": event.get("description"),
            "created_at": datetime.now().isoformat(),
            "raw_data": str(event)  # Store full JSON as string
        }
        rows.append(row)
    
    # Insert rows
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        print(f"❌ Errors inserting rows: {errors}")
        raise Exception(f"BigQuery insert errors: {errors}")
    
    print(f"✅ Saved {len(rows)} events to BigQuery")

def main():
    """Main function"""
    print("=" * 60)
    print("Vegas Events Collection")
    print("=" * 60)
    print()
    
    try:
        # Collect events
        events = collect_vegas_events()
        
        # Save to BigQuery
        if events:
            save_to_bigquery(events)
        
        print()
        print("✅ Vegas events collection complete!")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

