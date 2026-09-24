from fastapi import FastAPI
from pydantic import BaseModel
from backend.rag import ask_question

app = FastAPI(title="AgriGPT API")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(data: Question):
    response, sources = ask_question(data.question)
    return {
        "answer": response,
        "sources": sources
    }
