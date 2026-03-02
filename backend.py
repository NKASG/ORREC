import os
import uuid
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from usage_tracker import track_usage

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# DATABASE CONFIG (EDIT THIS)
# ======================================================

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class DocumentVector(Base):
    __tablename__ = "document_vectors"

    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # stored as JSON
    source = Column(String)


Base.metadata.create_all(engine)


# ======================================================
# MAIN RAG SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.groq_api_key = os.getenv("GROQ_API_KEY1")

        self.embed_model = None  # Lazy loading
        self.db = SessionLocal()

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )

    # ======================================================
    # LAZY MODEL LOADER
    # ======================================================
    def get_embed_model(self):
        if self.embed_model is None:
            logger.info("Loading embedding model...")
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embed_model

    # ======================================================
    # INGEST PDFs (MANUAL)
    # ======================================================
    def ingest_data(self):

        logger.info("Starting ingestion...")

        self.db.query(DocumentVector).delete()
        self.db.commit()

        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        documents = []

        for pdf in pdf_files:
            loader = PyPDFLoader(str(pdf))
            loaded_docs = loader.load()

            for doc in loaded_docs:
                doc.metadata["source"] = pdf.name

            documents.extend(loaded_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        vectors = self.get_embed_model().encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        for i, text in enumerate(texts):
            doc = DocumentVector(
                id=str(uuid.uuid4()),
                content=text,
                embedding=json.dumps(vectors[i]),
                source=metadatas[i].get("source")
            )
            self.db.add(doc)

        self.db.commit()

        logger.info("Ingestion completed.")

    # ======================================================
    # HYBRID SEARCH
    # ======================================================
    def hybrid_search(self, question, top_k=5):

        query_vector = self.get_embed_model().encode(
            [question],
            normalize_embeddings=True
        )[0]

        documents = self.db.query(DocumentVector).all()

        scored_results = []

        for doc in documents:
            stored_vector = json.loads(doc.embedding)

            semantic_score = float(
                sum(q * d for q, d in zip(query_vector, stored_vector))
            )

            question_words = set(re.findall(r"\w+", question.lower()))
            doc_words = set(re.findall(r"\w+", doc.content.lower()))
            keyword_score = len(question_words.intersection(doc_words)) * 0.05

            final_score = semantic_score + keyword_score

            scored_results.append((final_score, doc))

        scored_results.sort(reverse=True, key=lambda x: x[0])

        top_docs = [doc for _, doc in scored_results[:top_k]]

        context = "\n\n".join(doc.content for doc in top_docs)
        sources = list(set(doc.source for doc in top_docs if doc.source))

        return context, sources

    # ======================================================
    # MAIN QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        context, sources = self.hybrid_search(question)

        if not context.strip():
            return {
                "answer": "No relevant recruitment information found.",
                "sources": []
            }

        prompt = f"""
You are Orange Group’s Official Recruitment Assistant.

Provide a clear, professional, conversational response.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        response_obj = self.llm.invoke(prompt)
        response = response_obj.content.strip()

        try:
            tokens_used = len(prompt.split()) + len(response.split())
            cost = (tokens_used / 1_000_000) * 0.05

            track_usage(
                prompt=question,
                engine="llama-3.1-8b-instant",
                tokens_used=tokens_used,
                cost=round(cost, 6)
            )
        except:
            pass

        return {
            "answer": response,
            "sources": sources
        }


# ======================================================
# FASTAPI
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = RecruitmentRAG()


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def serve_ui():
    return FileResponse("frontend.html")


@app.post("/query")
def query_api(request: QueryRequest):
    return bot.query(request.question)


# 🔥 MANUAL INGEST ENDPOINT
@app.post("/ingest")
def ingest():
    bot.ingest_data()
    return {"status": "Ingestion completed"}