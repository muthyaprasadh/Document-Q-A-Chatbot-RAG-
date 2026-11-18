from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.llm_generate import generate_answer
from rag.retrieve import ingest_documents

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Endee RAG Document Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_documents() -> None:
    ingest_documents()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PROJECT_ROOT / "ui" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask_get(question: str):
    return ask(QuestionRequest(question=question))


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        answer = generate_answer(question)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"question": question, "answer": answer}
