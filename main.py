"""
FastAPI entry point for the Natural Language → SQL pipeline.

Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.backend import ask_database

app = FastAPI(
    title="LLM based Analytics Assistant",
    description="Natural Language → SQL pipeline using LangChain + Llama 3.1 + MySQL",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    sql: str | None
    answer: str


@app.get("/")
def home():
    return {"message": "Welcome to the LLM based Analytics Assistant API!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accept a natural-language question and return the generated SQL
    plus a natural-language answer.
    """
    result = ask_database(request.question)
    return QueryResponse(
        question=result["question"],
        sql=result["sql"],
        answer=result["answer"],
    )