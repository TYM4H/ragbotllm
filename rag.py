import os, tempfile, subprocess, json
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# Chroma в локальном каталоге
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs")

# эмбеддер через Ollama
def ollama_embed(text):
    cmd = ["ollama", "embeddings", "--model", "nomic-embed-text", text]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(out.stdout)["embedding"]

async def ingest_pdf(user_id, file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        path = tmp.name
    reader = PdfReader(path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    for i, chunk in enumerate(chunks):
        emb = ollama_embed(chunk)
        collection.add(
            ids=[f"{user_id}_{i}"],
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"user_id": user_id}]
        )

async def query_rag(user_id, question):
    q_emb = ollama_embed(question)
    results = collection.query(query_embeddings=[q_emb], n_results=4, where={"user_id": user_id})
    context = "\n\n".join(results["documents"][0])
    prompt = f"Ты помощник. Используй только контекст ниже:\n\n{context}\n\nВопрос: {question}"
    out = subprocess.run(["ollama", "run", "llama3", prompt], capture_output=True, text=True)
    return out.stdout.strip()
