# Tito RAG Backend (FastAPI)

A minimal FastAPI backend for ADRC Policies RAG (Tito).  
Serves `/api/rag` to answer questions using vector search + summarization.

## Requirements
- Python 3.11+
- See `requirements.txt`

## Environment Variables
Copy `.env.example` to `.env` locally and fill in real values (do NOT commit `.env`):
- `OPENAI_API_KEY`
- `ALLOWED_ORIGINS` (e.g., `https://adrc-p.com,https://www.adrc-p.com`)
- `RAG_INDEX_DIR` (default `./data/index`)
- `LOG_LEVEL` (default `info`)

## Run locally
```bash
pip install -r requirements.txt
# choose your entry file name:
uvicorn app:app --host 0.0.0.0 --port 8000   # if your file is app.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000  # if your file is main.py
