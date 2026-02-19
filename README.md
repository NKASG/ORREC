# ORREC — Orange Recruitment

A lightweight RAG-powered recruitment assistant combining FastAPI backend with a modern static frontend. Leverages semantic search and LLM capabilities for intelligent candidate and job matching.

## Overview

ORREC ingests recruitment documents (PDFs), builds semantic embeddings using SentenceTransformers, and serves intelligent queries through a clean chat interface. The system maintains conversation memory and learns from user corrections.

**Status:** Production-ready minimal implementation

## Features

- 🔍 **Hybrid Search** — Combines semantic and keyword-based retrieval for accurate results
- 💾 **Persistent Memory** — Stores user corrections and interaction history
- 📄 **PDF Ingestion** — Automatic document processing and vectorization
- ⚡ **Fast API** — RESTful backend with async support
- 🎨 **Chat UI** — Responsive web interface for seamless interaction
- 📊 **Usage Tracking** — Built-in telemetry for monitoring and analytics

## Project Structure

```
ORREC/
├── backend.py              # FastAPI application & RecruitmentRAG class
├── frontend.html           # Static chat UI
├── usage_tracker.py        # Telemetry helper
├── requirements.txt        # Python dependencies
├── memory_store.json       # Persisted corrections & history
├── document.ipynb          # Data ingestion examples
├── pdf_loader.ipynb        # PDF processing notebook
└── data/
    ├── pdf/                # Source PDF documents
    └── vector_store/       # Chroma vector database
```

## Core Components

### RecruitmentRAG
Main orchestration class with the following key methods:

| Method | Purpose |
|--------|---------|
| `ingest_data()` | Load PDFs, chunk content, generate embeddings, store in Chroma |
| `hybrid_search()` | Perform semantic + keyword retrieval |
| `query()` | End-to-end query pipeline (memory → correction → retrieval → LLM) |
| `load_memory()` / `save_memory()` | Manage persistent state |
| `check_memory()` | Retrieve interaction history |
| `store_correction()` | Learn from user feedback |

## Getting Started

### Prerequisites
- Python 3.9+
- API keys for: `GROQ_API_KEY1`, `AGENT_ID`, `USAGE_TRACKER_KEY`

### Installation

1. **Set up environment variables**
   ```sh
   # Create .env file in project root
   GROQ_API_KEY1=your_key_here
   AGENT_ID=your_agent_id
   USAGE_TRACKER_KEY=your_tracker_key
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Start the API server**
   ```sh
   uvicorn backend:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the application**
   - Open http://localhost:8000/ in your browser

## API Reference

### GET `/`
Returns the frontend chat interface.

**Response:** `text/html`

### POST `/query`
Submit a recruitment query and receive an intelligent response.

**Request:**
```json
{
  "question": "What are the top candidates for software engineer roles?"
}
```

**Response:**
```json
{
  "response": "Based on available candidates...",
  "sources": ["doc1.pdf", "doc2.pdf"]
}
```

## Configuration

### Document Management
- Place PDF documents in `data/pdf/`
- Vectors are automatically generated and stored in `data/vector_store/`
- Restart the server to re-ingest new documents

### Memory & Corrections
- User corrections are persisted in `memory_store.json`
- Memory is checked before each retrieval operation
- Clear memory by deleting the JSON file (will be recreated on next interaction)

### Usage Tracking
Telemetry is collected via `track_usage()`. To disable:
- Comment out tracking calls in `backend.py`
- Or set `USAGE_TRACKER_KEY` to empty string

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PDFs not loading | Ensure files are in `data/pdf/` and restart server |
| Vector DB errors | Delete `data/vector_store/` and restart to rebuild |
| Missing environment keys | Verify `.env` file exists with all required keys |
| API timeout | Check GROQ API rate limits and status |

## Development

### Running Tests
```sh
pytest tests/
```

### Notebooks
- `document.ipynb` — Data ingestion workflow
- `pdf_loader.ipynb` — PDF processing examples

### Logging
Check console output for ingestion and API logs (configured in `backend.py`).

## Performance Notes

- First query may take longer due to initial vectorization
- Subsequent queries benefit from vector caching
- Hybrid search balances semantic accuracy with keyword matching
- Memory checks are O(1) for cached interactions

## Security Considerations

- Protect `.env` file with API keys
- Validate `QueryRequest` inputs on the backend
- Consider rate limiting for production deployments
- Sanitize user corrections before storing in memory

## Contributing

1. Update PDFs in `data/pdf/`
2. Test changes locally before committing
3. Update this README for significant feature changes
4. Follow PEP 8 style guidelines

## License

Project files are proprietary demo code. Contact repository owner for licensing details.

## Support

For issues or questions:
- Check logs in console output
- Review `backend.py` for configuration options
- Refer to notebooks for usage examples

---

**Last Updated:** February 19, 2026