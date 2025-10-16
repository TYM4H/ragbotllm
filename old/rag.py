import requests
import chromadb
from pypdf import PdfReader
import tempfile
import re

# --- Persistent клиент Chroma ---
client = chromadb.PersistentClient(path="./chroma_db")


def get_user_collection(user_id):
    """Возвращает (или создаёт) персональную коллекцию пользователя"""
    return client.get_or_create_collection(f"user_{user_id}")


# --- Векторизация текста через Ollama API ---
def ollama_embed(text: str):
    """Получает embedding из локальной модели Ollama"""
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    if not r.ok:
        raise RuntimeError(f"Ollama embeddings error: {r.text}")
    return r.json()["embedding"]


# --- Очистка текста от мусора ---
def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^\w\s.,;:!?()\-\[\]]', '', t)
    return t.strip()


# --- Обработка и добавление PDF в векторное хранилище ---
async def ingest_pdf(user_id, file):
    """Читает PDF, делит на чанки, векторизует и сохраняет в Chroma"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = clean_text(text)

    # делим по предложениям, чтобы не рвать смысл
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = ""
    for s in sentences:
        if len(chunk) + len(s) < 1000:
            chunk += " " + s
        else:
            chunks.append(chunk.strip())
            chunk = s
    if chunk:
        chunks.append(chunk.strip())

    collection = get_user_collection(user_id)
    print(f"📄 Индексируем {len(chunks)} чанков для пользователя {user_id}")

    for i, chunk in enumerate(chunks):
        emb = ollama_embed(chunk)
        collection.add(
            ids=[f"{user_id}_{i}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"source": "pdf"}]
        )

    print("✅ Документ успешно добавлен.")


# --- Основной RAG-запрос ---
async def query_rag(user_id, question):
    """Находит релевантные фрагменты и отвечает через LLM"""
    collection = get_user_collection(user_id)
    q_emb = ollama_embed(question)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=10,
        include=["documents", "distances"]
    )

    pairs = list(zip(results["documents"][0], results["distances"][0]))
    pairs.sort(key=lambda x: x[1])  # чем меньше distance, тем ближе

    # берём только топ-4 ближайших чанка
    top_docs = [d for d, _ in pairs[:4]]

    # чистим и убираем дубли
    seen = set()
    clean_docs = []
    for d in top_docs:
        d = d.replace('\n', ' ').replace('\uf02d', '-').strip()
        if d not in seen:
            seen.add(d)
            clean_docs.append(d)

    context = "\n\n".join(clean_docs)

    print("\n🔍 Найденные документы (distance ↑):")
    for doc, dist in pairs[:6]:
        print(f"• dist={dist:.4f} → {doc[:120]}...")
    print("\n>>> CONTEXT SENT TO LLM:\n", context[:1000], "\n")

    prompt = f"""
Ты — интеллектуальный ассистент. Ответь строго по приведённому контексту.
Если ответ явно указан, перескажи его своими словами по-русски.
Если ответа нет — скажи: "В документе нет информации об этом."

=== КОНТЕКСТ ===
{context}

=== ВОПРОС ===
{question}
"""

    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    if not r.ok:
        raise RuntimeError(f"Ollama generation error: {r.text}")

    answer = r.json()["response"].strip()
    print("🤖 Ответ модели:", answer[:300])
    return answer
