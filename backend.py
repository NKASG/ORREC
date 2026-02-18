import os
import uuid
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from usage_tracker import track_usage

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

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

# ======================================================
# SIMPLE ANALYTICS STORAGE
# ======================================================
analytics_data = {
    "total_queries": 0,
    "questions": []
}


# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# BASIC SMALL TALK HANDLER
# ======================================================
def handle_small_talk(question: str):
    q = question.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
    if any(greet in q for greet in greetings):
        return {
            "answer": "Hello 👋 I am Orange Group’s Recruitment Assistant. How can I assist you today?",
            "sources": []
        }

    if "how are you" in q:
        return {
            "answer": "I'm functioning optimally and ready to assist with recruitment inquiries 😊",
            "sources": []
        }

    if "thank you" in q:
        return {
            "answer": "You're welcome! If you need further assistance, I’m here to help.",
            "sources": []
        }

    return None


# ======================================================
# RECRUITMENT RAG CLASS
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

        if self.collection.count() == 0:
            self.ingest_data()

    # ======================================================
    # INGEST
    # ======================================================
    def ingest_data(self):
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

        logger.info("Hybrid-ready ingestion complete.")

    # ======================================================
    # HYBRID SEARCH
    # ======================================================
    def hybrid_search(self, question: str, top_k=5):

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

        best_indices = [i for _, i in hybrid_rank[:3]]

        context = "\n\n".join(documents[i] for i in best_indices)
        sources = list(set(metadatas[i]["source"] for i in best_indices))

        return context, sources

    # ======================================================
    # QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        # Small talk first
        small_talk_response = handle_small_talk(question)
        if small_talk_response:
            return small_talk_response

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
- Do NOT reveal confidential policies.
- If not found, respond EXACTLY with:
"For further assistance, please contact recruitment@orangegroups.com"

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""
        # ---- Call LLM ----
        response_obj = self.llm.invoke(prompt)
        response = response_obj.content.strip()

        # ---- Estimate tokens (rough estimate using word count) ----
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response.split())
        tokens_used = prompt_tokens + completion_tokens

        # ---- Rough cost calculation ----
        # Adjust if your Groq pricing changes
        cost_per_million = 0.05
        cost = (tokens_used / 1_000_000) * cost_per_million

        # ---- Send usage tracking ----
        track_usage(
            prompt=question,
            engine="llama-3.1-8b-instant",
            tokens_used=tokens_used,
            cost=round(cost, 6)
        )

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
    answer_data = bot.query(request.question)

    # Store analytics
    analytics_data["total_queries"] += 1
    analytics_data["questions"].append(request.question)

    return answer_data


@app.get("/admin/analytics")
def get_analytics():
    return analytics_data

