import os
import requests
from dotenv import load_dotenv
from typing import List
from pydantic import PrivateAttr

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


class GeminiEmbeddings(Embeddings):
    """Gemini text embeddings via embedContent endpoint."""
    _api_key: str = PrivateAttr()
    _endpoint: str = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._api_key = os.getenv("GEMINI_API_KEY")
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self._api_key}"
        )

    def embed_query(self, text: str) -> List[float]:
        payload = {"content": {"parts": [{"text": text}]}}
        resp = requests.post(self._endpoint, json=payload)
        resp.raise_for_status()
        embedding = resp.json().get("embedding", {})
        return embedding.get("values", [])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class GeminiLLM(LLM):
    """Gemini text generation via generateContent endpoint."""
    _model: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _endpoint: str = PrivateAttr()

    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__()
        self._model = model
        self._api_key = os.getenv("GEMINI_API_KEY")
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self._api_key}"
        )

    def _call(self, prompt: str, **kwargs) -> str:
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 256
            }
        }
        resp = requests.post(self._endpoint, json=payload)
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if not candidates:
            return ""
        return candidates[0]["content"]["parts"][0]["text"].strip()

    @property
    def _identifying_params(self):
        return {"model": self._model}

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

    @property
    def _output_type(self) -> str:
        return "text"


# Build the Retrieval QA chain

def get_qa_chain(
    persist_directory: str = "./chroma_db",
    k: int = 3
) -> RetrievalQA:
    """
    Build a RetrievalQA chain using GeminiEmbeddings and GeminiLLM.
    """
    # 1) Load or create the Chroma vector store
    embeddings = GeminiEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 2) Configure the Gemini LLM
    llm = GeminiLLM()

    # 3) Define a prompt template
    prompt = PromptTemplate(
        template=(
            "You are a financial planning expert."
            "\nBased ONLY on the provided context, answer the multiple-choice question by returning EXACTLY one letter: A, B, C, or D.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    # 4) Assemble the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
