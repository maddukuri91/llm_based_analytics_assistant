"""Natural-language to SQL backend with a multi-turn clarification engine.

Flow:
User question -> clarification engine -> SQL generation -> validation ->
MySQL execution -> natural-language answer.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import sqlparse
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import Engine, create_engine, inspect, text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "3"))
MAX_RESULT_ROWS_FOR_LLM = int(os.getenv("MAX_RESULT_ROWS_FOR_LLM", "200"))

ALLOWED_TABLES = {
    "film",
    "film_category",
    "category",
    "film_actor",
    "actor",
    "inventory",
    "rental",
    "payment",
    "customer",
    "store",
    "staff",
    "address",
    "city",
    "country",
}


def create_db_engine(
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: str,
    db_name: str,
) -> Engine:
    """Create a SQLAlchemy engine from explicit database credentials."""
    db_password_escaped = quote_plus(db_password)
    database_url = (
        f"mysql+pymysql://{db_user}:{db_password_escaped}@"
        f"{db_host}:{db_port}/{db_name}"
    )
    return create_engine(database_url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def create_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> Any:
    """Create an LLM instance based on the provider and optional API key.

    Supported providers: groq, openai, ollama.
    - groq / openai require an API key (from argument or environment).
    - ollama runs locally and does not need an API key.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()

    if provider == "groq":
        from langchain_groq import ChatGroq

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("A Groq API key is required.")
        return ChatGroq(
            model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=key,
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("An OpenAI API key is required.")
        return ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o"),
            api_key=key,
            temperature=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            temperature=0,
        )

    raise ValueError("LLM provider must be 'groq', 'openai', or 'ollama'.")


def build_chains(llm: Any) -> dict[str, Any]:
    """Build the clarification, SQL, and answer chains for a given LLM."""
    return {
        "clarification": clarification_prompt | llm,
        "sql": sql_prompt | llm,
        "answer": answer_prompt | llm,
    }


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def get_schema_info(db_engine: Engine, allowed_tables: set[str]) -> str:
    """Return columns and foreign keys for tables the model may query."""
    inspector = inspect(db_engine)
    available_tables = set(inspector.get_table_names())
    schema_lines: list[str] = []

    for table_name in sorted(available_tables & allowed_tables):
        columns = inspector.get_columns(table_name)
        column_description = ", ".join(
            f"{column['name']} ({column['type']})" for column in columns
        )
        schema_lines.append(f"Table `{table_name}`: {column_description}")

        for foreign_key in inspector.get_foreign_keys(table_name):
            local_columns = ", ".join(foreign_key["constrained_columns"])
            remote_columns = ", ".join(foreign_key["referred_columns"])
            schema_lines.append(
                f"FK: {table_name}.({local_columns}) -> "
                f"{foreign_key['referred_table']}.({remote_columns})"
            )

    return "\n".join(schema_lines)


# ---------------------------------------------------------------------------
# Clarification engine
# ---------------------------------------------------------------------------

clarification_prompt = ChatPromptTemplate.from_template(
    """You are the clarification stage of a natural-language database assistant.

Available schema:
{schema_info}

Original user question:
{original_question}

Clarification conversation so far:
{clarification_history}

Decide whether the request is sufficiently precise to generate one SQL query.

Use CLARIFY only when missing information would materially change the query,
such as an ambiguous metric, entity, time range, grouping, ranking, or words
like "best", "top", "recent", and "active". Do not ask unnecessary questions.

Rules:
- Ask exactly one short, specific question at a time.
- Offer 2-4 choices when that makes the question easier to answer.
- Use READY when the request is sufficiently precise.
- When READY, produce a complete standalone resolved_query incorporating all
  clarification answers.
- Use CANNOT_ANSWER only when the schema cannot answer the request.
- Never generate SQL here.
- Return valid JSON only. Do not use Markdown.

JSON format:
{{
  "status": "READY" | "CLARIFY" | "CANNOT_ANSWER",
  "question": null | "one user-facing clarification question",
  "resolved_query": null | "complete standalone question",
  "reason": "brief internal reason"
}}"""
)


def _format_clarification_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No clarification questions have been asked."

    lines: list[str] = []
    for index, turn in enumerate(history, start=1):
        lines.append(f"Round {index} assistant question: {turn.get('question', '')}")
        lines.append(f"Round {index} user answer: {turn.get('answer', '')}")
    return "\n".join(lines)


def _parse_json_object(raw_content: str) -> dict[str, Any]:
    """Parse a JSON object even if the model accidentally adds code fences."""
    cleaned = raw_content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("The clarification model did not return a JSON object.")
    return json.loads(match.group(0))


def assess_clarity(
    original_question: str,
    schema_info: str,
    clarification_chain: Any,
    clarification_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    history = clarification_history or []
    response = clarification_chain.invoke(
        {
            "schema_info": schema_info,
            "original_question": original_question,
            "clarification_history": _format_clarification_history(history),
        }
    )
    decision = _parse_json_object(response.content)

    status = str(decision.get("status", "")).upper()
    if status not in {"READY", "CLARIFY", "CANNOT_ANSWER"}:
        raise ValueError(f"Invalid clarification status: {status or 'missing'}")

    decision["status"] = status
    if status == "CLARIFY" and not decision.get("question"):
        raise ValueError("CLARIFY response did not contain a question.")
    if status == "READY" and not decision.get("resolved_query"):
        decision["resolved_query"] = original_question
    return decision


# ---------------------------------------------------------------------------
# SQL generation
# ---------------------------------------------------------------------------

SQL_RULES = """Rules:
- Use only tables and columns in the supplied schema.
- Generate exactly one SELECT statement.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
  GRANT, REVOKE, EXEC, or EXECUTE.
- Use explicit JOINs based on the supplied foreign-key relationships.
- Prefer qualified column names when multiple tables are used.
- Do not use common table expressions (WITH clauses).
- Return only raw SQL, without explanations or Markdown fences.
- If the request cannot be answered from the schema, return CANNOT_ANSWER.
"""

sql_prompt = ChatPromptTemplate.from_template(
    """You are a MySQL expert. Convert the resolved user request into SQL.

Schema:
{schema_info}

{sql_rules}

Resolved user request:
{user_query}

SQL:"""
)


def generate_sql(user_query: str, schema_info: str, sql_chain: Any) -> str:
    response = sql_chain.invoke(
        {
            "schema_info": schema_info,
            "sql_rules": SQL_RULES,
            "user_query": user_query,
        }
    )
    sql = response.content.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
}


class SQLValidationError(Exception):
    """Raised when generated SQL violates a safety rule."""


def validate_sql(sql: str, allowed_tables: set[str]) -> str:
    normalized = sql.strip()
    if not normalized or normalized.upper() == "CANNOT_ANSWER":
        raise SQLValidationError("The model could not translate the request into SQL.")

    split_statements = [item for item in sqlparse.split(normalized) if item.strip()]
    if len(split_statements) != 1:
        raise SQLValidationError("Exactly one SQL statement is allowed.")

    parsed = sqlparse.parse(normalized)
    if not parsed or not parsed[0].tokens:
        raise SQLValidationError("The generated SQL could not be parsed.")

    statement_type = parsed[0].get_type()
    if statement_type != "SELECT":
        raise SQLValidationError(
            f"Only SELECT statements are allowed; received {statement_type}."
        )

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized, re.IGNORECASE):
            raise SQLValidationError(f"Forbidden SQL keyword detected: {keyword}")

    referenced_tables = {
        table_name.lower()
        for table_name in re.findall(
            r"(?:FROM|JOIN)\s+(?:`?\w+`?\.)?`?([A-Za-z_][A-Za-z0-9_]*)`?",
            normalized,
            flags=re.IGNORECASE,
        )
    }
    disallowed_tables = referenced_tables - {name.lower() for name in allowed_tables}
    if disallowed_tables:
        raise SQLValidationError(
            f"Query references disallowed tables: {sorted(disallowed_tables)}"
        )

    return normalized


# ---------------------------------------------------------------------------
# Query execution and answer generation
# ---------------------------------------------------------------------------

def run_query(db_engine: Engine, sql: str) -> pd.DataFrame:
    with db_engine.connect() as connection:
        result = connection.execute(text(sql))
        return pd.DataFrame(result.fetchall(), columns=result.keys())


answer_prompt = ChatPromptTemplate.from_template(
    """The user asked:
{user_query}

The retrieved data is:
{query_result}

Answer the user's question concisely using only the retrieved data.
Do not mention SQL, tables, schemas, or databases. If there are no rows, say
that no matching information was found. Do not invent missing information."""
)


def explain_results(
    user_query: str,
    dataframe: pd.DataFrame,
    answer_chain: Any,
) -> str:
    limited = dataframe.head(MAX_RESULT_ROWS_FOR_LLM)
    result_text = limited.to_string(index=False) if not limited.empty else "No rows returned."
    if len(dataframe) > len(limited):
        result_text += f"\nShowing {len(limited)} of {len(dataframe)} returned rows."

    response = answer_chain.invoke(
        {"user_query": user_query, "query_result": result_text}
    )
    return response.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _response(
    *,
    status: str,
    question: str,
    answer: str,
    resolved_query: str | None = None,
    clarification_question: str | None = None,
    sql: str | None = None,
    result: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "question": question,
        "resolved_query": resolved_query,
        "clarification_question": clarification_question,
        "sql": sql,
        "result": result,
        "answer": answer,
    }


def ask_database(
    user_query: str,
    db_engine: Engine,
    llm: Any | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    retries: int = 1,
    allowed_tables: set[str] | None = None,
    clarification_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run clarification first and execute SQL only when the request is ready.

    Each clarification history item must look like:
    {"question": "Question previously shown to the user", "answer": "User reply"}

    Pass an ``llm`` instance explicitly, or provide ``provider``/``api_key``/
    ``model`` to create one. If none are given, environment defaults are used.
    """
    user_query = user_query.strip()
    history = clarification_history or []
    tables = set(allowed_tables or ALLOWED_TABLES)

    if not user_query:
        return _response(
            status="error",
            question=user_query,
            answer="Please enter a question.",
        )

    try:
        active_llm = llm or create_llm(
            provider=provider, api_key=api_key, model=model
        )
        chains = build_chains(active_llm)
    except Exception as exc:
        return _response(
            status="error",
            question=user_query,
            answer=f"Could not initialise the LLM: {exc}",
        )

    try:
        current_schema_info = get_schema_info(db_engine, tables)
    except Exception as exc:
        return _response(
            status="error",
            question=user_query,
            answer=f"Could not retrieve schema information: {exc}",
        )

    if not current_schema_info:
        return _response(
            status="error",
            question=user_query,
            answer="No permitted database schema is available.",
        )

    try:
        clarity = assess_clarity(
            user_query,
            current_schema_info,
            chains["clarification"],
            history,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return _response(
            status="error",
            question=user_query,
            answer=f"Could not assess the request clearly: {exc}",
        )

    if clarity["status"] == "CLARIFY":
        if len(history) >= MAX_CLARIFICATION_ROUNDS:
            return _response(
                status="cannot_answer",
                question=user_query,
                answer=(
                    "I still do not have enough detail to answer safely. "
                    "Please restate the request with the metric, filters, and time period."
                ),
            )
        clarification_question = str(clarity["question"])
        return _response(
            status="needs_clarification",
            question=user_query,
            clarification_question=clarification_question,
            answer=clarification_question,
        )

    if clarity["status"] == "CANNOT_ANSWER":
        return _response(
            status="cannot_answer",
            question=user_query,
            answer="I cannot answer that from the available information.",
        )

    resolved_query = str(clarity.get("resolved_query") or user_query).strip()
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            sql = generate_sql(resolved_query, current_schema_info, chains["sql"])
            validated_sql = validate_sql(sql, tables)
            dataframe = run_query(db_engine, validated_sql)
            answer = explain_results(resolved_query, dataframe, chains["answer"])
            return _response(
                status="complete",
                question=user_query,
                resolved_query=resolved_query,
                sql=validated_sql,
                result=dataframe,
                answer=answer,
            )
        except SQLValidationError as exc:
            last_error = exc
            break
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break

    return _response(
        status="error",
        question=user_query,
        resolved_query=resolved_query,
        answer=f"I could not answer the request: {last_error or 'unknown error'}",
    )


def continue_after_clarification(
    original_question: str,
    previous_history: list[dict[str, str]],
    clarification_question: str,
    user_answer: str,
    db_engine: Engine,
    llm: Any | None = None,
    **ask_options: Any,
) -> dict[str, Any]:
    """Convenience helper for submitting the user's next clarification reply."""
    updated_history = [
        *previous_history,
        {"question": clarification_question, "answer": user_answer.strip()},
    ]
    return ask_database(
        original_question,
        db_engine=db_engine,
        llm=llm,
        clarification_history=updated_history,
        **ask_options,
    )


def check_database_connection(db_engine: Engine) -> None:
    """Optional startup health check; call this from the application startup hook."""
    with db_engine.connect() as connection:
        connection.execute(text("SELECT 1"))