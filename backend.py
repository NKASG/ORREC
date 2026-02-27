import os
import uuid
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import ProgrammingError

from usage_tracker import track_usage

# Try pgvector (enterprise mode)
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except:
    PGVECTOR_AVAILABLE = False


# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# DATABASE CONFIG
# ======================================================

DATABASE_URL = "postgresql://postgres:newpassword123@127.0.0.1:5433/recruitment_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ======================================================
# VECTOR MODEL
# ======================================================

EMBED_DIM = 384  # all-MiniLM-L6-v2


# ======================================================
# TABLE DEFINITION
# ======================================================

if PGVECTOR_AVAILABLE:
    class DocumentVector(Base):
        __tablename__ = "document_vectors"

        id = Column(String, primary_key=True)
        content = Column(Text, nullable=False)
        embedding = Column(Vector(EMBED_DIM))
        source = Column(String)
else:
    class DocumentVector(Base):
        __tablename__ = "document_vectors"

        id = Column(String, primary_key=True)
        content = Column(Text, nullable=False)
        embedding = Column(Text)  # JSON fallback
        source = Column(String)


Base.metadata.create_all(engine)


# ======================================================
# MAIN RAG SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.groq_api_key = os.getenv("GROQ_API_KEY1")

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.db = SessionLocal()

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )

        self.ingest_data()

    # ======================================================
    # INGEST DATA
    # ======================================================
    def ingest_data(self):

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

        vectors = self.embed_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        for i, text in enumerate(texts):

            if PGVECTOR_AVAILABLE:
                embedding_value = vectors[i]
            else:
                embedding_value = json.dumps(vectors[i])

            doc = DocumentVector(
                id=str(uuid.uuid4()),
                content=text,
                embedding=embedding_value,
                source=metadatas[i].get("source")
            )

            self.db.add(doc)

        self.db.commit()

    # ======================================================
    # HYBRID SEARCH (Enterprise Mode)
    # ======================================================
    def hybrid_search(self, question, top_k=5):

        query_vector = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        )[0]

        if PGVECTOR_AVAILABLE:
            results = (
                self.db.query(DocumentVector)
                .order_by(DocumentVector.embedding.cosine_distance(query_vector))
                .limit(top_k)
                .all()
            )
        else:
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
            results = [doc for _, doc in scored_results[:top_k]]

        context = "\n\n".join(doc.content for doc in results)
        sources = list(set(doc.source for doc in results if doc.source))

        return context, sources

    # ======================================================
    # MAIN QUERY (UNCHANGED LOGIC)
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        context, sources = self.hybrid_search(question)

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