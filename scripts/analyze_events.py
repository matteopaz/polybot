"""
Analyze events data - plot histogram of market volumes.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    events_file = data_dir / "events.json"
    
    # Load events
    with open(events_file) as f:
        events = json.load(f)
    
    # Extract volumes (filter out None values)
    volumes = [e["volume"] for e in events if e.get("volume") is not None]
    
    print(f"Loaded {len(volumes)} events with volume data")
    print(f"Volume range: ${min(volumes):,.2f} - ${max(volumes):,.2f}")
    print(f"Median volume: ${np.median(volumes):,.2f}")
    print(f"Mean volume: ${np.mean(volumes):,.2f}")
    
    # Plot histogram with log scale (volumes span many orders of magnitude)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale histogram
    axes[0].hist(volumes, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Volume ($)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Market Volumes")
    
    # Log scale histogram (more informative for wide-ranging data)
    log_volumes = np.log10(np.array(volumes) + 1)  # +1 to handle zeros
    axes[1].hist(log_volumes, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("log₁₀(Volume + 1)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Market Volumes (Log Scale)")
    
    plt.tight_layout()
    
    # Save figure
    output_file = data_dir / "volume_histogram.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved histogram to {output_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
