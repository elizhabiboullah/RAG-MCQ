import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

load_dotenv()

def get_gemini_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {os.getenv('GEMINI_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    url = "https://gemini.googleapis.com/v1beta2/models/embedding:generate"
    payload = {
        "texts": texts
    }

    response = requests.post(url, json=payload, headers=headers)
    embeddings = response.json().get("embeddings", [])
    
    return embeddings

def embed_and_store(chunks, persist_directory="./chroma_db"):
    embeddings = get_gemini_embeddings([chunk['text'] for chunk in chunks])
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb
