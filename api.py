# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from main2 import answer_question

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    response = answer_question(query.question)
    return {"response": response}
