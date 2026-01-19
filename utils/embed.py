import os
import pickle
import atexit
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/embeddings_cache.pkl")

_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            _cache = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(_cache, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

atexit.register(save_cache)

def embed_text(text: str) -> np.ndarray:
    key = text.strip().lower()
    if key in _cache:
        return _cache[key]

    print(f"Embedding: {key[:50]}...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": key
                }
            )
            
            if response.status_code != 200:
                print(f"Error embedding text (attempt {attempt+1}): {response.text}")
                if attempt == max_retries - 1:
                    response.raise_for_status()
                time.sleep(1)
                continue

            data = response.json()
            embedding = data["data"][0]["embedding"]

            result = np.array(embedding).reshape(1, len(embedding))
            _cache[key] = result
            return result
            
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print(f"Network error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))

def embed_texts(texts: list[str], n_threads=16) -> np.ndarray:
    if not texts:
        return np.array([])

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(embed_text, texts))

    if not results:
         return np.array([])
         
    return np.vstack(results)