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


# ==============================
# DATABASE
# ==============================

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class DocumentVector(Base):
    __tablename__ = "document_vectors"

    id = Column(String, primary_key=True)
    content = Column(Text)
    embedding = Column(Text)
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

    def __init__(self):

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

        self.auto_ingest_if_empty()

    # ==============================
    # AUTO INGEST
    # ==============================

    def auto_ingest_if_empty(self):

        count = self.db.query(DocumentVector).count()

        if count == 0:
            logger.info("Database empty — ingesting documents")
            self.ingest()

    # ==============================
    # DOCUMENT INGESTION
    # ==============================

    def ingest(self):

        logger.info("Starting ingestion")

        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

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
                source=chunks[i].metadata["source"]
            )

            self.db.add(row)

        self.db.commit()

        logger.info("Ingestion complete")

    # ==============================
    # SEARCH
    # ==============================

    def search(self, question, top_k=5):

        query_vector = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        )[0]

        docs = self.db.query(DocumentVector).all()

        scored = []

        for doc in docs:

            vector = json.loads(doc.embedding)

            semantic = sum(q * d for q, d in zip(query_vector, vector))

            q_words = set(re.findall(r"\w+", question.lower()))
            d_words = set(re.findall(r"\w+", doc.content.lower()))

            keyword = len(q_words.intersection(d_words)) * 0.05

            score = semantic + keyword

            scored.append((score, doc))

        scored.sort(reverse=True, key=lambda x: x[0])

        top_docs = [doc for _, doc in scored[:top_k]]

        context = "\n\n".join(doc.content for doc in top_docs)

        sources = list(set(doc.source for doc in top_docs if doc.source))

        return context, sources

    # ==============================
    # SMART INTENT DETECTION
    # ==============================

    def detect_intent(self, question):

        q = question.lower()

        # GREETINGS
        greetings = [
            "hi", "hello", "hey",
            "good morning", "good afternoon",
            "good evening", "how are you"
        ]

        if any(g in q for g in greetings):

            return {
                "answer": (
                    "Hello 👋\n\n"
                    "Welcome to Orange Group's Recruitment Assistant.\n\n"
                    "I can help you with:\n"
                    "• Job openings\n"
                    "• How to apply\n"
                    "• Interview process\n"
                    "• Eligibility requirements\n\n"
                    "How may I assist you today?"
                ),
                "sources": []
            }

        # JOB OPENINGS
        job_words = ["job", "jobs", "vacancy", "vacancies", "openings"]

        if any(word in q for word in job_words):

            roles = self.memory.memory.get("current_roles", [])

            if not roles:
                return {
                    "answer": "There are currently no job openings available.",
                    "sources": []
                }

            response = "Current available roles:\n\n"

            for role in roles:
                response += (
                    f"• {role['title']} — {role['department']} ({role['location']})\n"
                    f"{role['description']}\n\n"
                )

            return {
                "answer": response,
                "sources": ["Internal Job Database"]
            }

        # HOW TO APPLY
        if "apply" in q or "application" in q:

            return {
                "answer": (
                    "To apply for a role:\n\n"
                    "• Visit the official recruitment portal\n"
                    "• Complete the online application\n"
                    "• Upload required documents:\n"
                    "   - Updated CV\n"
                    "   - Degree certificate\n"
                    "   - NYSC certificate\n"
                    "   - Relevant credentials\n\n"
                    "Shortlisted candidates may be invited for testing or interviews."
                ),
                "sources": ["Recruitment Guidelines"]
            }

        # INTERVIEW
        if "interview" in q or "test" in q:

            return {
                "answer": (
                    "The recruitment process typically includes:\n\n"
                    "1. Online application submission\n"
                    "2. Candidate shortlisting\n"
                    "3. Test or interview stage\n"
                    "4. Final evaluation\n\n"
                    "Candidates who pass the primary test may proceed to further interviews."
                ),
                "sources": ["Recruitment Process"]
            }

        # ELIGIBILITY
        if "eligibility" in q or "requirements" in q or "qualification" in q:

            return {
                "answer": (
                    "Eligibility requirements may include:\n\n"
                    "• A recognized university degree\n"
                    "• Completion of NYSC (for Nigerian applicants)\n"
                    "• Relevant professional credentials\n"
                    "• Meeting the requirements of the specific role"
                ),
                "sources": ["Recruitment Policy"]
            }

        return None

    # ==============================
    # MAIN QUERY
    # ==============================

    def query(self, question):

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

    logger.info(f"Question: {request.question}")

    return rag.query(request.question)


@app.post("/correct")
def correct_answer(request: CorrectionRequest):

    rag.memory.store(request.question, request.answer)

    return {"status": "saved"}