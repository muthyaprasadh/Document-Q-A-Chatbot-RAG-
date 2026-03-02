from fastapi import FastAPI
from rag.llm_generate import generate_answer

app = FastAPI(title="Endee RAG Document Q&A")

@app.get("/ask")
def ask(question: str):
    answer = generate_answer(question)
    return {
        "question": question,
        "answer": answer
    }
