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
from sqlalchemy.exc import ProgrammingError

from dotenv import load_dotenv
from usage_tracker import track_usage


# ==============================
# ENV
# ==============================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY1")

DATA_DIR = "./data"
MEMORY_FILE = "memory_store.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# DATABASE CONFIG
# ======================================================

# ==============================
# DATABASE
# ==============================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


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


# ==============================
# MEMORY SYSTEM
# ==============================

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
            {
                "question": question,
                "answer": answer
            }
        )

        self.save()


# ==============================
# RAG SYSTEM
# ==============================

class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.groq_api_key = os.getenv("GROQ_API_KEY1")

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.db = SessionLocal()

        self.memory = MemoryStore()

        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        self.ingest_data()

    # ======================================================
    # INGEST DATA
    # ======================================================
    def ingest_data(self):

        count = self.db.query(DocumentVector).count()

        if count == 0:
            logger.info("Database empty — ingesting documents")
            self.ingest()

    # ==============================
    # DOCUMENT INGESTION
    # ==============================

    def ingest(self):

        logger.info("Starting ingestion")

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
        logger.info(f"Created {len(chunks)} chunks")

        texts = [c.page_content for c in chunks]

        vectors = self.get_embed_model().encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        self.db.query(DocumentVector).delete()

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

        query_vector = self.get_embed_model().encode(
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

        # INTENT DETECTION
        intent = self.detect_intent(question)

        if intent:
            return intent

        # MEMORY CORRECTION
        correction = self.memory.check(question)

        if correction:
            return {
                "answer": correction,
                "sources": ["Memory Correction"]
            }

        # DOCUMENT SEARCH
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
- Never use asterisks:
- Only answer using the provided context
- Never invent information
- Respond clearly and professionally
- Use bullet points when helpful
- If information is not found say:
  "I could not find this information in For further assistance, please contact recruitment@orangegroups.com."

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


# ==============================
# FASTAPI
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RecruitmentRAG()


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