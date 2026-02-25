import os
import uuid
import logging
import re
import json
from pathlib import Path
from typing import Dict, Any

# import chromadb
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

import pydantic.v1.fields



if not hasattr(pydantic.v1.fields.ModelField, '_type_hints_patched'):
    original_infer = pydantic.v1.fields.ModelField.infer
    
    def patched_infer(*args, **kwargs):
        # Only try to cast to int if the value is not None 
        # and the name matches known problematic fields
        problematic_fields = {'chroma_server_nofile', 'chroma_server_grpc_port'}
        
        name = kwargs.get('name')
        value = kwargs.get('value')
        
        if name in problematic_fields and value is not None:
            try:
                kwargs['value'] = int(value)
            except (ValueError, TypeError):
                pass 
                
        return original_infer(*args, **kwargs)
    
    pydantic.v1.fields.ModelField.infer = patched_infer
    pydantic.v1.fields.ModelField._type_hints_patched = True

import chromadb  # Now import chromadb

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrangeRecruitment")

# ======================================================
# MAIN RAG SYSTEM
# ======================================================
class RecruitmentRAG:

    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.groq_api_key = os.getenv("GROQ_API_KEY1")

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chroma_client = chromadb.PersistentClient(path="./data/vector_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="recruitment_docs"
        )

        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
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

                if "corrections" not in data:
                    data["corrections"] = []

                if "current_roles" not in data:
                    data["current_roles"] = []

                return data
        except:
            return {"corrections": [], "current_roles": []}

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

    def check_memory(self, question):
        memory = self.load_memory()
        q_lower = question.lower()

        for item in memory["corrections"]:
            if item["question"] in q_lower:
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

        response_text = "We are currently recruiting for the following roles:\n\n"

        for role in roles:
            response_text += f"- {role['title']} ({role['department']}, {role['location']})\n"
            response_text += f"  {role['description']}\n\n"

        return {
            "answer": response_text.strip(),
            "sources": ["Live Recruitment Roles"]
        }

    # ======================================================
    # INTELLIGENT DOMAIN FILTER
    # ======================================================
    def is_recruitment_related(self, question: str) -> bool:

        recruitment_profile = """
        Orange Group recruitment, employment, job application,
        interview process, candidate eligibility,
        contract staff recruitment, staffing process,
        vacancies, job openings, hiring process,
        management trainee programme,
        sales and marketing bootcamp,
        employment requirements and policies.
        """

        question_vec = self.embed_model.encode(
            [question],
            normalize_embeddings=True
        )[0]

        profile_vec = self.embed_model.encode(
            [recruitment_profile],
            normalize_embeddings=True
        )[0]

        similarity = float(question_vec @ profile_vec)
        logger.info(f"Domain similarity score: {similarity}")

        return similarity >= 0.30

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
    # MAIN QUERY
    # ======================================================
    def query(self, question: str) -> Dict[str, Any]:

        if not question.strip():
            return {"answer": "Please enter a valid question.", "sources": []}

        q_lower = question.lower().strip()

        # 1️⃣ Greetings
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

        if q_lower in greetings:
            return {
                "answer": "Hello 👋 I am Orange Group’s Recruitment Assistant. How may I assist you today?",
                "sources": []
            }

        if "how are you" in q_lower:
            return {
                "answer": "I'm functioning optimally and ready to assist you with Orange Group recruitment inquiries 😊",
                "sources": []
            }

        # 2️⃣ Current Roles
        role_keywords = [
            "available roles",
            "current roles",
            "open positions",
            "vacancies",
            "job openings",
            "are you recruiting",
            "what roles are available",
            "what are the roles"
        ]

        if any(keyword in q_lower for keyword in role_keywords):
            return self.get_current_roles_response()

        # 3️⃣ Intelligent Domain + Hybrid Fallback
        if not self.is_recruitment_related(question):

            context_check, _ = self.hybrid_search(question)

            if not context_check.strip():
                return {
                    "answer": "I only answer questions that are Orange Group recruitment related.",
                    "sources": []
                }

        # 4️⃣ Memory Correction
        memory_answer = self.check_memory(question)
        if memory_answer:
            return {
                "answer": memory_answer,
                "sources": ["Learned Correction"]
            }

        # 5️⃣ Detect Correction
        correction_phrases = ["that is wrong", "correction:", "the correct answer is"]

        for phrase in correction_phrases:
            if phrase in q_lower:
                correct_answer = question.split(phrase)[-1].strip()
                self.store_correction(question, correct_answer)
                return {
                    "answer": "✅ Thank you. I’ve learned this correction and will use it next time.",
                    "sources": []
                }

        # 6️⃣ Hybrid Search
        context, sources = self.hybrid_search(question)

        if not context.strip():
            return {
                "answer": "For further assistance, please contact recruitment@orangegroups.com",
                "sources": []
            }

        # 7️⃣ Conversational Prompt
        prompt = f"""
You are Orange Group’s Official Recruitment Assistant.

Provide a clear, professional, conversational response.
Do not mention section numbers or document headings.
Rewrite naturally as if explaining to a staff member.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER:
"""

        response_obj = self.llm.invoke(prompt)
        response = response_obj.content.strip()

        # 8️⃣ Usage Tracking
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