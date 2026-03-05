import os
import uuid
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from sentence_transformers import SentenceTransformer

from usage_tracker import track_usage


# =================================
# ENV
# =================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY1")

DATA_DIR = "./data"
MEMORY_FILE = "memory_store.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")


# =================================
# DATABASE
# =================================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class DocumentVector(Base):
    __tablename__ = "document_vectors"

    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
    source = Column(String)


Base.metadata.create_all(engine)


# =================================
# MEMORY STORE
# =================================

class MemoryStore:

    def __init__(self):

        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w") as f:
                json.dump({"corrections": [], "current_roles": []}, f)

        self.memory = self.load()

    def load(self):

        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    def save(self):

        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=4)

    def check(self, question):

        for item in self.memory.get("corrections", []):
            if item["question"].lower() == question.lower():
                return item["answer"]

        return None

    def store(self, question, answer):

        self.memory["corrections"].append(
            {"question": question, "answer": answer}
        )

        self.save()


# =================================
# RAG SYSTEM
# =================================

class RecruitmentRAG:

    def __init__(self, data_dir="./data"):

        self.data_dir = data_dir
        self.db = SessionLocal()
        self.memory = MemoryStore()

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        self.auto_ingest()

    # ===============================
    # INGEST DOCUMENTS
    # ===============================

    def auto_ingest(self):

        count = self.db.query(DocumentVector).count()

        if count == 0:
            logger.info("Database empty — ingesting PDFs")
            self.ingest()

    def ingest(self):

        pdf_files = list(Path(self.data_dir).glob("*.pdf"))

        documents = []

        for pdf in pdf_files:

            loader = PyPDFLoader(str(pdf))
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = pdf.name

            documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        vectors = self.embed_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        self.db.query(DocumentVector).delete()

        for i, text in enumerate(texts):

            row = DocumentVector(
                id=str(uuid.uuid4()),
                content=text,
                embedding=json.dumps(vectors[i]),
                source=metadatas[i].get("source")
            )

            self.db.add(row)

        self.db.commit()

        logger.info("Document ingestion complete")

    # ===============================
    # VECTOR SEARCH
    # ===============================

    def search(self, question, top_k=5):

        query_vector = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        )[0]

        docs = self.db.query(DocumentVector).all()

        scored = []

        for doc in docs:

            vector = json.loads(doc.embedding)

            score = sum(q * d for q, d in zip(query_vector, vector))

            scored.append((score, doc))

        scored.sort(reverse=True, key=lambda x: x[0])

        top_docs = [doc for _, doc in scored[:top_k]]

        context = "\n\n".join(doc.content for doc in top_docs)
        sources = list(set(doc.source for doc in top_docs if doc.source))

        return context, sources

    # ===============================
    # MAIN QUERY
    # ===============================

    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        correction = self.memory.check(question)

        if correction:
            return {
                "answer": correction,
                "sources": ["Memory Correction"]
            }

        context, sources = self.search(question)

        if not context.strip():
            return {
                "answer": "No relevant recruitment information found.",
                "sources": []
            }

        prompt = f"""
You are Orange Group's official Recruitment Assistant.

Rules:
- Use bullet points with "•"
- Never use asterisks
- Only answer using the context
- Respond professionally

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        response = self.llm.invoke(prompt).content.strip()

        try:

            tokens = len(prompt.split()) + len(response.split())

            cost = (tokens / 1_000_000) * 0.05

            track_usage(
                prompt=question,
                engine="llama-3.1-8b-instant",
                tokens_used=tokens,
                cost=cost
            )

        except Exception as e:
            logger.error(e)

        return {"answer": response, "sources": sources}


# =================================
# FASTAPI
# =================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = RecruitmentRAG()


class QueryRequest(BaseModel):
    question: str


class CorrectionRequest(BaseModel):
    question: str
    answer: str


@app.get("/")
def ui():
    return FileResponse("frontend.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_api(request: QueryRequest):
    return bot.query(request.question)


@app.post("/correct")
def correct_answer(request: CorrectionRequest):

    bot.memory.store(request.question, request.answer)

    return {"status": "saved"}