import requests
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "qwen/qwen3-embedding-8b",
                    "input": ["hello"]*512
                }
            )

print(len(response.json()["data"]))