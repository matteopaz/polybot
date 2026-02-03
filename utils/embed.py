import os
import pickle
import atexit
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from dotenv import load_dotenv
from joblib import Parallel, delayed
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/embeddings_cache.pkl")

_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            _cache = pickle.load(f)
    except EOFError:
        _cache = {}
    except Exception as e:
        print(f"Warning: Could not load cache: {e}")

def save_cache():
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        temp_file = CACHE_FILE + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(_cache, f)
        os.replace(temp_file, CACHE_FILE)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

atexit.register(save_cache)

def embed_text_batch(texts: list[str]) -> np.ndarray | None:
    keys = [text.strip().lower() for text in texts]
    embs = [_cache[key] if key in _cache else None for key in keys]

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": keys
            }
        )
    
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]

        result = np.array(embeddings)
        for key, emb in zip(keys, result):
            _cache[key] = emb
        
        return result
        
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"Network error: {e}")

def embed_texts(texts: list[str], chunksize=256) -> np.ndarray:
    batches = [texts[i : i + chunksize] for i in range(0, len(texts), chunksize)]
    results = Parallel(n_jobs=4, prefer="threads")(
        delayed(embed_text_batch)(batch) for batch in batches
    )
    
    # Filter out Nones in case of failures and concierge results
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        return np.array([])
        
    return np.vstack(valid_results)