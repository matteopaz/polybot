"""
Score events using the insider_event_score_parallel function.

Reads events from data/events.json and creates a mapping from event id to insider score.
"""

import json
from pathlib import Path
from utils import insider_event_score_parallel

VOLUME_THRESHOLD = 25000 # Minimum volume to consider an event

def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    events_file = data_dir / "events.json"
    output_file = data_dir / "event_scores.json"
    
    # Load events
    with open(events_file) as f:
        events = json.load(f)

    try:
        with open(output_file) as f:
            scored_events = json.load(f)
            id_to_score = {int(k): v for k, v in scored_events.items()}
    except FileNotFoundError:
        id_to_score = {}
    
    # Extract relevant events
    event_titles = []
    event_ids = []
    for event in events:
        if event["volume"] and event["volume"] < VOLUME_THRESHOLD:
            continue
        if event["id"] in id_to_score:
            continue
        event_titles.append(event["title"])
        event_ids.append(event["id"])


    print(f"Scoring {len(event_titles)} events with volume >= ${VOLUME_THRESHOLD:,}...")
    
    # Score all events in parallel
    scores = insider_event_score_parallel(event_titles)
    
    # Create id -> score mapping
    id_to_score = {eid: score for eid, score in zip(event_ids, scores)}
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(id_to_score, f, indent=2)
    
    print(f"Saved {len(id_to_score)} scores to {output_file}")


if __name__ == "__main__":
    main()
