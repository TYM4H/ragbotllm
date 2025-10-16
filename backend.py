from fastapi import FastAPI, UploadFile, Form
from rag import ingest_pdf, query_rag

app = FastAPI()

@app.post("/ingest")
async def ingest(user_id: str = Form(...), file: UploadFile = None):
    await ingest_pdf(user_id, file)
    return {"ok": True}

@app.post("/query")
async def query(user_id: str = Form(...), question: str = Form(...)):
    answer = await query_rag(user_id, question)
    return {"answer": answer}
