"""
FastAPI entry point for the Natural Language → SQL pipeline.

Run with: uvicorn main:app --reload
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.backend import ask_database, create_db_engine

app = FastAPI(
    title="LLM based Analytics Assistant",
    description="Natural Language → SQL pipeline using LangChain + Llama 3.1 + MySQL",
    version="1.0.0",
)

# Build the engine from environment variables (or defaults) at startup.
DB_USER = os.getenv("DB_USER", "sql_agent")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Delhi@369")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "sakila")

engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)


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
    result = ask_database(request.question, db_engine=engine)
    return QueryResponse(
        question=result["question"],
        sql=result["sql"],
        answer=result["answer"],
    )