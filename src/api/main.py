from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from src.retriever.rag_chain import get_qa_chain
import time

app = FastAPI()
router = APIRouter()

rag_chain = get_qa_chain()

class QuestionRequest(BaseModel):
    question: str
    options: list[str]

@app.post("/predict")
async def answer_mcq(request: QuestionRequest):
    start_time = time.time()  # Track the start time
    formatted_question = request.question + "\n" + "\n".join(request.options)
    print(f"Processing question: {request.question}")
    
    # Call the RAG chain
    response = rag_chain.run(formatted_question)
    
    end_time = time.time()  # Track the end time
    elapsed_time = end_time - start_time
    print(f"Processed in {elapsed_time:.2f} seconds")

    # Return in the expected format
    return {"predicted_answer": response, "confidence": 1.0}

