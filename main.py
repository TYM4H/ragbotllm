from fastapi import FastAPI, UploadFile, Form
from rag.pipeline import ingest_pdf, query_rag

app = FastAPI()


@app.post("/ingest")
async def ingest(user_id: str = Form(...), file: UploadFile = None):
    file_bytes = await file.read() 
    ingest_pdf(user_id, file_bytes)  # передаём байты
    return {"ok": True}


@app.post("/query")
async def query(user_id: str = Form(...), question: str = Form(...)):
    answer = query_rag(user_id, question)
    return {"answer": answer}
