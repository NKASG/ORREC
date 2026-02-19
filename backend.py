import os
import uuid
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from usage_tracker import track_usage

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# RECRUITMENT RAG SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.groq_api_key = os.getenv("GROQ_API_KEY1")

        logger.info("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("Initializing vector store...")
        self.chroma_client = chromadb.PersistentClient(path="./data/vector_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="recruitment_docs"
        )

        logger.info("Loading LLM...")
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.5
        )

        # Always re-ingest on startup
        self.ingest_data()

    # ======================================================
    # MEMORY SYSTEM
    # ======================================================
    def load_memory(self):
        try:
            with open("memory_store.json", "r") as f:
                return json.load(f)
        except:
            return {"corrections": []}

    def save_memory(self, data):
        with open("memory_store.json", "w") as f:
            json.dump(data, f, indent=4)

    def store_correction(self, question, correct_answer):
        memory = self.load_memory()

        memory["corrections"].append({
            "question": question.lower(),
            "answer": correct_answer
        })

        self.save_memory(memory)
        logger.info("Correction stored successfully.")

    def check_memory(self, question):
        memory = self.load_memory()
        q_lower = question.lower()

        for item in memory["corrections"]:
            if item["question"] in q_lower:
                return item["answer"]

        return None

    # ======================================================
    # INGEST PDFs
    # ======================================================
    def ingest_data(self):

        # Clear existing vectors
        try:
            self.collection.delete(where={})
            logger.info("Cleared existing vector store.")
        except:
            pass

        pdf_files = list(Path(self.data_dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files.")

        documents = []

        for pdf in pdf_files:
            logger.info(f"Ingesting {pdf.name}")
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
        ids = [str(uuid.uuid4()) for _ in chunks]

        vectors = self.embed_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        self.collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )

        logger.info("All PDFs ingested successfully.")

    # ======================================================
    # HYBRID SEARCH
    # ======================================================
    def hybrid_search(self, question, top_k=5):

        query_vector = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        semantic_results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = semantic_results["documents"][0]
        metadatas = semantic_results["metadatas"][0]
        distances = semantic_results["distances"][0]

        keyword_scores = []
        question_words = set(re.findall(r"\w+", question.lower()))

        for doc in documents:
            doc_words = set(re.findall(r"\w+", doc.lower()))
            overlap = len(question_words.intersection(doc_words))
            keyword_scores.append(overlap)

        hybrid_rank = []
        for i in range(len(documents)):
            semantic_score = 1 - distances[i]
            keyword_score = keyword_scores[i] * 0.05
            final_score = semantic_score + keyword_score
            hybrid_rank.append((final_score, i))

        hybrid_rank.sort(reverse=True)

        best_indices = [i for _, i in hybrid_rank[:5]]

        context = "\n\n".join(documents[i] for i in best_indices)
        sources = list(set(metadatas[i]["source"] for i in best_indices))

        return context, sources

    # ======================================================
    # QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        # 1️⃣ Check memory first
        memory_answer = self.check_memory(question)
        if memory_answer:
            return {
                "answer": memory_answer,
                "sources": ["Learned Correction"]
            }

        # 2️⃣ Detect correction phrases
        correction_phrases = ["that is wrong", "correction:", "the correct answer is"]

        lower_q = question.lower()

        for phrase in correction_phrases:
            if phrase in lower_q:
                correct_answer = question.split(phrase)[-1].strip()
                self.store_correction(question, correct_answer)

                return {
                    "answer": "✅ Thank you. I’ve learned this correction and will use it next time.",
                    "sources": []
                }

        # 3️⃣ Hybrid Search
        context, sources = self.hybrid_search(question)

        if not context.strip():
            return {
                "answer": "For further assistance, please contact recruitment@orangegroups.com",
                "sources": []
            }

        prompt = f"""
You are Orange Group’s Official Recruitment Assistant.

STRICT RULES:
- Answer ONLY from the provided context.
- Do NOT invent information.
- If not found, respond EXACTLY with:
"For further assistance, please contact recruitment@orangegroups.com"

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        response_obj = self.llm.invoke(prompt)
        response = response_obj.content.strip()

        # ==============================
        # Usage Tracking
        # ==============================
        try:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(response.split())
            tokens_used = prompt_tokens + completion_tokens

            cost_per_million = 0.05
            cost = (tokens_used / 1_000_000) * cost_per_million

            track_usage(
                prompt=question,
                engine="llama-3.1-8b-instant",
                tokens_used=tokens_used,
                cost=round(cost, 6)
            )
        except Exception as e:
            logger.warning(f"Usage tracking failed: {e}")

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
