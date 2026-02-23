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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from usage_tracker import track_usage

# ======================================================
# ENV & LOGGING
# ======================================================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# MAIN SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chroma_client = chromadb.PersistentClient(path="/var/data/vector_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="recruitment_docs"
        )

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY1"),
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )

        self.ingest_data()

    # ======================================================
    # MEMORY SYSTEM
    # ======================================================
    def load_memory(self):
        try:
            with open("memory_store.json", "r") as f:
                data = json.load(f)
                data.setdefault("corrections", [])
                data.setdefault("current_roles", [])
                return data
        except:
            return {"corrections": [], "current_roles": []}

    def save_memory(self, data):
        with open("memory_store.json", "w") as f:
            json.dump(data, f, indent=4)

    def store_correction(self, question, correct_answer):
        memory = self.load_memory()
        memory["corrections"].append({
            "question": question.lower().strip(),
            "answer": correct_answer.strip()
        })
        self.save_memory(memory)

    def check_memory(self, question):
        memory = self.load_memory()
        q = question.lower()

        for item in memory["corrections"]:
            if item["question"] in q:
                return item["answer"]
        return None

    def get_current_roles_response(self):
        memory = self.load_memory()
        roles = memory.get("current_roles", [])

        if not roles:
            return {
                "answer": "There are currently no active recruitment roles available. Please check back later.",
                "sources": []
            }

        response = "We are currently recruiting for the following roles:\n\n"

        for role in roles:
            response += f"- {role['title']} ({role['department']}, {role['location']})\n"
            response += f"  {role['description']}\n\n"

        return {
            "answer": response.strip(),
            "sources": ["Live Recruitment Roles"]
        }

    # ======================================================
    # INGEST PDFs
    # ======================================================
    def ingest_data(self):
        try:
            self.collection.delete(where={})
        except:
            pass

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

    # ======================================================
    # HYBRID SEARCH
    # ======================================================
    def hybrid_search(self, question, top_k=5):

        query_vector = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # If similarity too weak → treat as no context
        if min(distances) > 0.85:
            return "", []

        context = "\n\n".join(documents)
        sources = list(set(m["source"] for m in metadatas))

        return context, sources

    # ======================================================
    # MAIN QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        q_lower = question.lower().strip()

        # Greetings
        if q_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]:
            return {"answer": "Hello 👋 I am Orange Recruitment. How may I assist you today?", "sources": []}

        if "how are you" in q_lower:
            return {"answer": "I'm functioning optimally and ready to assist you with recruitment inquiries.", "sources": []}

        # Current roles
        if any(x in q_lower for x in ["available roles", "current roles", "vacancies", "job openings"]):
            return self.get_current_roles_response()

        # Memory correction
        memory_answer = self.check_memory(question)
        if memory_answer:
            return {"answer": memory_answer, "sources": ["Learned Correction"]}

        # Detect correction
        if any(x in q_lower for x in ["that is wrong", "correction:", "the correct answer is"]):
            correct_answer = question.split("is")[-1].strip()
            self.store_correction(question, correct_answer)
            return {"answer": "✅ Thank you. I’ve learned this correction.", "sources": []}

        # Hybrid search (Domain Gate)
        context, sources = self.hybrid_search(question)

        if not context:
            return {
                "answer": "I only answer questions that are Orange Group recruitment related.",
                "sources": []
            }

        # Strict Anti-Hallucination Prompt
        prompt = f"""
You are "Orange Recruitment", a professional and compliance-restricted recruitment specialist for Orange Group.

PRIMARY FUNCTION:
Your sole responsibility is to provide accurate recruitment information strictly based on the provided CONTEXT extracted from official PDF documents in the knowledge base.

You are NOT a general AI assistant.

------------------------------------------------------------
PURPOSE AND OBJECTIVES
------------------------------------------------------------

• Provide precise and factual answers regarding recruitment processes, eligibility criteria, job roles, staffing policies, candidate requirements, and related matters.
• Ensure every statement made is directly supported by the provided CONTEXT.
• Help users locate and understand recruitment-related information clearly and efficiently.

------------------------------------------------------------
MANDATORY RULES (STRICT COMPLIANCE)
------------------------------------------------------------

1. INFORMATION SOURCE RESTRICTION
   - You MUST answer ONLY using the information explicitly stated in the CONTEXT.
   - You MUST NOT use external knowledge.
   - You MUST NOT infer missing details.
   - You MUST NOT fabricate dates, times, locations, names, numbers, or policies.

2. IF INFORMATION IS NOT AVAILABLE
   - If the answer is not clearly supported by the CONTEXT, respond EXACTLY with:
     "For further assistance, please contact recruitment@orangegroups.com"
   - Do NOT attempt to guess or partially answer.

3. CONTEXT BOUNDARY
   - Do NOT go outside the provided CONTEXT.
   - Do NOT provide general recruitment advice unless explicitly stated in the CONTEXT.
   - Do NOT summarize beyond what is directly supported.

4. OUTPUT STRUCTURE
   - Do NOT mention section numbers, clause numbers, or document headings.
   - Do NOT reference "the document states" or similar phrasing.
   - Rewrite the answer in clear, professional language.
   - Maintain a formal and business-like tone.

5. UNCERTAINTY HANDLING
   - If the CONTEXT is partially relevant but does not fully answer the question, you MUST refuse.

------------------------------------------------------------
PROFESSIONAL TONE
------------------------------------------------------------

• Formal
• Objective
• Concise
• Business-oriented
• Authoritative but factual

------------------------------------------------------------
CONTEXT:
{context}

------------------------------------------------------------
USER QUESTION:
{question}

------------------------------------------------------------
FINAL ANSWER:
"""

        response = self.llm.invoke(prompt).content.strip()

        # Numeric Hallucination Guard
        numbers_in_response = re.findall(r"\d+", response)
        numbers_in_context = re.findall(r"\d+", context)

        for num in numbers_in_response:
            if num not in numbers_in_context:
                response = "For further assistance, please contact recruitment@orangegroups.com"
                break

        # Usage Tracking
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

        return {"answer": response, "sources": sources}


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