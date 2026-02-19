
Workspace: Collecting workspace information# ORREC — Orange Recruitment

Simple RAG-powered recruitment assistant (FastAPI backend + static frontend).

Status: minimal demo — ingests PDFs, builds embeddings with SentenceTransformers, serves a chat UI.

Files
- backend.py — main backend and API server; exposes `app` and `QueryRequest`.  
- frontend.html — static chat UI served at `/`.  
- usage_tracker.py — usage telemetry helper: `track_usage`.  
- requirements.txt — Python dependencies.  
- document.ipynb — ingestion / notebook examples.  
- pdf_loader.ipynb — PDF loader notebook.  
- memory_store.json — persisted user corrections.  
- data/ — local data (PDFs under data/pdf and vector DB under data/vector_store).

Key components
- `RecruitmentRAG` — main class. Important methods:
  - `RecruitmentRAG.ingest_data` — load PDFs, chunk, embed, and add to Chroma.  
  - `RecruitmentRAG.hybrid_search` — semantic + keyword hybrid retrieval.  
  - `RecruitmentRAG.query` — end-to-end query handling (memory check, correction detection, retrieval, LLM call).  
  - `RecruitmentRAG.load_memory`, `RecruitmentRAG.save_memory`, `RecruitmentRAG.store_correction`, `RecruitmentRAG.check_memory`.

Quick start
1. Create a .env with required keys (e.g., `GROQ_API_KEY1`, `AGENT_ID`, `USAGE_TRACKER_KEY`). See backend.py and usage_tracker.py.  
2. Install deps:
```sh
pip install -r requirements.txt
```
3. Run the API (example with uvicorn):
```sh
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```
4. Open http://localhost:8000/ to use the UI (frontend.html).

API
- GET / → serves frontend.html.  
- POST /query → accepts JSON matching `QueryRequest` { "question": "..." } and returns the assistant response (handled by `RecruitmentRAG.query`).

Notes
- Ingested vectors are stored under data/vector_store (Chroma).  
- Corrections persist in memory_store.json and are checked before retrieval.  
- Usage tracking calls `track_usage`; disable or adjust if needed.

Contributing / Troubleshooting
- Update PDFs under data/pdf and restart to re-ingest (ingestion runs on startup in `RecruitmentRAG.__init__`).  
- Check logs for ingestion and tracking messages (logger in backend.py).  

License
- Project files are local demo code. See repository owner for licensing.
