import json
from datetime import datetime
from pathlib import Path

from utils import PolymarketSDK

# Keywords to exclude if found in the event slug
exclude_keywords = ["btc", "eth", "xrp", "sol"]

def pull_all_events() -> list[dict]:
    """Fetch all events from Polymarket, excluding those with specific keywords in slug."""
    client = PolymarketSDK()
    
    all_events = []
    offset = 0 
    limit = 500  # Max batch size
    
    print(f"Pulling events from Polymarket (excluding slugs with: {', '.join(exclude_keywords)})...")
    
    while True:
        batch = client.list_events(limit=limit, offset=offset, include_markets=False)
        if not batch:
            break
        
        for event in batch:
            slug = (event.slug or "").lower()
            # Exclude if any keyword is present in the slug
            if not any(kw in slug for kw in exclude_keywords):
                all_events.append({
                    "id": event.id,
                    "title": event.title,
                    "slug": event.slug,
                    "volume": event.volume,
                    "created_at": event.created_at.strftime("%Y-%m-%d") if event.created_at else None,
                })
        
        print(f"  Fetched {len(batch)} events at offset {offset}")
        
        if len(batch) < limit:
            break
        offset += limit
    
    # Sort by creation date descendingly
    all_events.sort(key=lambda x: x["created_at"] or "", reverse=True)
    
    print(f"Total events: {len(all_events)}")
    return all_events


def main():
    # Pull events
    events = pull_all_events()
    
    # Save to data directory
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "events.json"
    with open(output_file, "w") as f:
        json.dump(events, f, indent=2)
    
    print(f"Saved {len(events)} events to {output_file}")


if __name__ == "__main__":
    main()
