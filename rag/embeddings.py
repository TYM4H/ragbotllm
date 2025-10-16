import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

def embed_text(text: str):
    """Получает embedding для текста через Ollama."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": text},
            timeout=120
        )
        if not r.ok:
            print(f"Ошибка Ollama: {r.status_code}")
            return []
        return r.json().get("embedding", [])
    except Exception as e:
        print(f"Ошибка при обращении к Ollama: {e}")
        return []
