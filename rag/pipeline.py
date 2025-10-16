import requests
import tempfile
import chromadb
from pypdf import PdfReader
from rag.embeddings import embed_text
from rag.utils import clean_text

client = chromadb.PersistentClient(path="./chroma_db")


def get_user_collection(user_id):
    return client.get_or_create_collection(f"user_{user_id}")


def ingest_pdf(user_id, file_bytes: bytes):
    """Добавляет PDF в Chroma одним embedding."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = clean_text(text)

    emb = embed_text(text)
    collection = get_user_collection(user_id)
    collection.add(
        ids=[f"{user_id}_full"],
        embeddings=[emb],
        documents=[text],
        metadatas=[{"source": "pdf"}]
    )


def query_rag(user_id, question):
    """Находит релевантный документ и формирует ответ."""
    collection = get_user_collection(user_id)
    q_emb = embed_text(question)
    if not q_emb:
        return "Ошибка embedding при запросе."

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=1,
        include=["documents", "distances"]
    )

    if not results or not results.get("documents") or len(results["documents"][0]) == 0:
        return "В документе нет информации об этом."

    doc = results["documents"][0][0]

    prompt = f"""
    Ты — ассистент. Отвечай строго на основе контекста.
    Если ответ есть в документе — сформулируй кратко по-русски.
    Если нет — скажи: "В документе нет информации об этом."

    === КОНТЕКСТ ===
    {doc}

    === ВОПРОС ===
    {question}
    """

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=180
    )

    if not r.ok:
        raise RuntimeError(f"Ollama generation error: {r.text}")

    return r.json().get("response", "").strip()
