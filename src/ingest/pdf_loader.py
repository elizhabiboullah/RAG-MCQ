from langchain.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdfs_from_directory(directory_path: str):
    pdf_files = Path(directory_path).glob("*.pdf")
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents
