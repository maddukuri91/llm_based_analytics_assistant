"""
Backend for the Natural Language → SQL pipeline.

Implements the flow:
User Query → Prompt Template (schema/rules) → Llama 3.1 → SQL Validator → SQLAlchemy → MySQL → Results → LLM (English answer) → User Response
"""

import os
import re
from urllib.parse import quote_plus

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, inspect, text

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# --- MySQL connection (override via environment variables) ---
DB_USER = os.getenv("DB_USER", "sql_agent")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", "Delhi@369"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "sakila")

# --- LLM provider: "groq" or "ollama" ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Only needed if LLM_PROVIDER == "groq"
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "your_groq_api_key"))

# Tables the LLM is allowed to query (used by both the prompt and the validator)
ALLOWED_TABLES = {
    "film", "film_category", "category", "film_actor", "actor",
    "inventory", "rental", "payment", "customer", "store", "staff",
    "address", "city", "country",
}

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def get_schema_info(engine, allowed_tables):
    inspector = inspect(engine)
    schema_lines = []

    for table in inspector.get_table_names():
        if table not in allowed_tables:
            continue

        cols = inspector.get_columns(table)
        col_desc = ", ".join(
            f"{c['name']} ({c['type']})" for c in cols
        )
        schema_lines.append(f"Table `{table}`: {col_desc}")

        for fk in inspector.get_foreign_keys(table):
            schema_lines.append(
                f"FK: {table}.{fk['constrained_columns']} -> "
                f"{fk['referred_table']}.{fk['referred_columns']}"
            )

    return "\n".join(schema_lines)


# ---------------------------------------------------------------------------
# Prompt template (LangChain)
# ---------------------------------------------------------------------------

SQL_RULES = """
Rules:
- Only use tables from the schema provided below.
- Only generate SELECT statements. Never write INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.
- Always use explicit JOINs based on the foreign key relationships given.
- Return ONLY the raw SQL query. No explanation, no markdown code fences.
- If the question cannot be answered with the given schema, return exactly: CANNOT_ANSWER
"""

sql_prompt = ChatPromptTemplate.from_template(
    """You are a MySQL expert. Convert the user's question into a single SQL query.

Schema:
{schema_info}

{sql_rules}

User question: {user_query}

SQL query:"""
)


# ---------------------------------------------------------------------------
# LLM (Llama 3.1)
# ---------------------------------------------------------------------------

if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
elif LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen3:8b", temperature=0)
else:
    raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'")

sql_chain = sql_prompt | llm


def generate_sql(user_query: str, schema_info: str) -> str:
    response = sql_chain.invoke({
        "schema_info": schema_info,
        "sql_rules": SQL_RULES,
        "user_query": user_query,
    })
    sql = response.content.strip()
    # Strip accidental markdown fences
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


# ---------------------------------------------------------------------------
# SQL Validator Layer
# ---------------------------------------------------------------------------

FORBIDDEN_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
}


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str, allowed_tables: set) -> str:
    if not sql or sql.strip().upper() == "CANNOT_ANSWER":
        raise SQLValidationError("Model could not translate the question into SQL.")

    # 1. Syntax check
    parsed = sqlparse.parse(sql)
    if not parsed or not parsed[0].tokens:
        raise SQLValidationError("Could not parse SQL — invalid syntax.")

    statement = parsed[0]
    stmt_type = statement.get_type()

    # 2. SELECT-only check
    if stmt_type != "SELECT":
        raise SQLValidationError(f"Only SELECT statements are allowed, got: {stmt_type}")

    # 3. Forbidden keyword check (defense in depth beyond stmt_type)
    tokens_upper = sql.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", tokens_upper):
            raise SQLValidationError(f"Forbidden keyword detected: {kw}")

    # 4. Allowed-tables check
    referenced_tables = set(re.findall(r"(?:FROM|JOIN)\s+`?(\w+)`?", sql, re.IGNORECASE))
    disallowed = referenced_tables - allowed_tables
    if disallowed:
        raise SQLValidationError(f"Query references disallowed table(s): {disallowed}")

    # 5. Single statement only (no stacked queries)
    if len(sqlparse.split(sql)) > 1:
        raise SQLValidationError("Multiple statements are not allowed.")

    return sql


# ---------------------------------------------------------------------------
# Execute query (SQLAlchemy → MySQL)
# ---------------------------------------------------------------------------

def run_query(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# LLM converts results into an English answer
# ---------------------------------------------------------------------------

answer_prompt = ChatPromptTemplate.from_template(
    """The user asked: "{user_query}"

The SQL query below was run and returned this data (as a table):
{query_result}

Write a concise, natural-language answer to the user's question based only on this data.
Do not mention SQL or databases in your answer."""
)

answer_chain = answer_prompt | llm


def explain_results(user_query: str, df: pd.DataFrame) -> str:
    result_str = df.to_string(index=False) if not df.empty else "No rows returned."
    response = answer_chain.invoke({
        "user_query": user_query,
        "query_result": result_str,
    })
    return response.content.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def ask_database(user_query: str, retries: int = 1, allowed_tables: set | None = None):
    """
    Full pipeline: NL question -> SQL -> validate -> execute -> NL answer.
    Retries once with the LLM if validation or execution fails.
    
    Args:
        user_query: Natural language question
        retries: Number of retries on failure
        allowed_tables: Set of table names the LLM can query. If None, uses global ALLOWED_TABLES
    """
    # Use provided allowed_tables or fall back to global
    tables = allowed_tables if allowed_tables is not None else ALLOWED_TABLES
    
    # Get schema info for the selected tables
    try:
        current_schema_info = get_schema_info(engine, tables)
    except Exception as e:
        return {
            "question": user_query,
            "sql": None,
            "result": None,
            "answer": f"Sorry, could not retrieve schema information: {str(e)}",
        }
    
    if current_schema_info is None or current_schema_info == "":
        return {
            "question": user_query,
            "sql": None,
            "result": None,
            "answer": "Sorry, schema information is not available. Please check your database connection.",
        }

    last_error = None
    for attempt in range(retries + 1):
        try:
            sql = generate_sql(user_query, current_schema_info)
            sql = validate_sql(sql, tables)
            df = run_query(engine, sql)
            answer = explain_results(user_query, df)
            return {
                "question": user_query,
                "sql": sql,
                "result": df,
                "answer": answer,
            }
        except SQLValidationError as e:
            last_error = e
            # Don't retry validation errors - they won't improve
            break
        except Exception as e:
            last_error = e
            if attempt < retries:
                continue
            break
    
    error_msg = str(last_error) if last_error else "Unknown error occurred"
    return {
        "question": user_query,
        "sql": None,
        "result": None,
        "answer": f"Sorry, I couldn't answer that. ({error_msg})",
    }


# ---------------------------------------------------------------------------
# Initialize schema info at import time
# ---------------------------------------------------------------------------

try:
    with engine.connect() as conn:
        print("✅ Connected to MySQL")
        print(conn.execute(text("SELECT DATABASE();")).fetchone())

    schema_info = get_schema_info(engine, ALLOWED_TABLES)
    print(schema_info)

except Exception as e:
    print(f"⚠️  Could not connect to database: {e}")
    schema_info = None
    raise RuntimeError(f"Failed to initialize database connection and schema: {e}") from e
