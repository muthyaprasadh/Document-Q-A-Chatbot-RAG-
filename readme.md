# Endee RAG Document Q&A

A FastAPI-based document question-answering app. It embeds text from `data/` with
Sentence Transformers and Endee when available, and includes an in-memory/keyword
fallback so the app remains usable without optional services.

## Run

```powershell
py -m pip install -r requirements.txt
py -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Optional OpenAI

Set `OPENAI_API_KEY` to generate answers with an LLM:

```powershell
$env:OPENAI_API_KEY="your-key"
py -m uvicorn app:app --reload
```

Without an API key, the app answers from the most relevant document text.

## API

- `GET /health` - service health
- `GET /ask?question=What+is+a+DFA` - question via query parameter
- `POST /ask` with `{"question":"What is a DFA?"}` - JSON question