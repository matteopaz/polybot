import numpy as np
from utils.embed import embed_texts
import json
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt

excluded_words = [""]

def relate(event_titles: list[str]):
    embeddings = embed_texts(event_titles) # (N, D) array
    relation = embeddings @ embeddings.T  # (N, N) array
    return relation

if __name__ == "__main__":

    with open("data/events.json") as f:
        events = json.load(f)

    titles = [e["title"] for e in events]

    relation_matrix = relate(titles)

    mean_rel = np.mean(relation_matrix)
    median_rel = np.median(relation_matrix)
    min_rel = np.min(relation_matrix)
    max_rel = np.max(relation_matrix)
    std_rel = np.std(relation_matrix)
    print(f"Relation matrix stats: mean={mean_rel}, median={median_rel}, min={min_rel}, max={max_rel}, std={std_rel}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(relation_matrix, cmap="viridis")
    plt.title("Event Relation Heatmap")
    plt.xlabel("Event Index")
    plt.ylabel("Event Index")
    plt.tight_layout()
    plt.show()