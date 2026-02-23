import os
import uuid
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from usage_tracker import track_usage

# ======================================================
# ENV
# ======================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# RAG SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir

        # Use Chroma default lightweight embedding
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.chroma_client = chromadb.PersistentClient(
            path="/var/data/vector_store"  # Render persistent path
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="recruitment_docs",
            embedding_function=self.embedding_function
        )

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY1"),
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )

        # Only ingest if empty
        if self.collection.count() == 0:
            self.ingest_data()

    # ======================================================
    # MEMORY
    # ======================================================
    def load_memory(self):
        try:
            with open("memory_store.json", "r") as f:
                return json.load(f)
        except:
            return {"corrections": [], "current_roles": []}

    def save_memory(self, data):
        with open("memory_store.json", "w") as f:
            json.dump(data, f, indent=4)

    def check_memory(self, question):
        memory = self.load_memory()
        q = question.lower()

        for item in memory.get("corrections", []):
            if item["question"] in q:
                return item["answer"]
        return None

    # ======================================================
    # INGEST
    # ======================================================
    def ingest_data(self):
        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        documents = []

        for pdf in pdf_files:
            loader = PyPDFLoader(str(pdf))
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        texts = [c.page_content for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.collection.add(
            documents=texts,
            ids=ids
        )

        logger.info("PDF ingestion complete.")

    # ======================================================
    # QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        q_lower = question.lower()

        # Greetings
        if q_lower in ["hi", "hello", "hey", "good morning", "good afternoon"]:
            return {
                "answer": "Hello 👋 I am Orange Recruitment. How may I assist you today?",
                "sources": []
            }

        # Memory check
        memory_answer = self.check_memory(question)
        if memory_answer:
            return {"answer": memory_answer, "sources": ["Learned Correction"]}

        # Retrieval (Domain Gate)
        results = self.collection.query(
            query_texts=[question],
            n_results=5
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            return {
                "answer": "I only answer questions that are Orange Group recruitment related.",
                "sources": []
            }

        context = "\n\n".join(documents)

        prompt = f"""
You are Orange Recruitment.

STRICT RULES:
- Use ONLY the provided CONTEXT.
- Do NOT invent numbers, dates, names, or locations.
- If unsupported, respond EXACTLY with:
"For further assistance, please contact recruitment@orangegroups.com"

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        response = self.llm.invoke(prompt).content.strip()

        # Numeric hallucination guard
        numbers_in_response = re.findall(r"\d+", response)
        numbers_in_context = re.findall(r"\d+", context)

        for num in numbers_in_response:
            if num not in numbers_in_context:
                response = "For further assistance, please contact recruitment@orangegroups.com"
                break

        # Usage tracking
        try:
            tokens = len(prompt.split()) + len(response.split())
            cost = (tokens / 1_000_000) * 0.05

            track_usage(
                prompt=question,
                engine="llama-3.1-8b-instant",
                tokens_used=tokens,
                cost=round(cost, 6)
            )
        except:
            pass

        return {"answer": response, "sources": []}


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