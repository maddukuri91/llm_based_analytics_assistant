"""
Streamlit UI for the Natural Language → SQL pipeline.

Run with: streamlit run app/ui.py
"""

import os
import sys

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect

# Ensure the project root is on the path so `app.backend` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend import ask_database, get_schema_info, ALLOWED_TABLES as DEFAULT_ALLOWED_TABLES  # noqa: E402

st.set_page_config(
    page_title="LLM based Analytics Assistant",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: connection setup and table selection
# ---------------------------------------------------------------------------

st.sidebar.header("Database Connection")

# MySQL connection form
with st.sidebar.form("connection_form"):
    st.subheader("MySQL Connection Details")
    db_user = st.text_input("Username", value="")
    db_password = st.text_input("Password", type="password", value="")
    db_host = st.text_input("Host", value="")
    db_port = st.text_input("Port", value="")
    db_name = st.text_input("Database", value="")
    connect_button = st.form_submit_button("Connect")

# Initialize session state for database connection
if "engine" not in st.session_state:
    st.session_state.engine = None
if "schema_info" not in st.session_state:
    st.session_state.schema_info = None
if "available_tables" not in st.session_state:
    st.session_state.available_tables = []
if "selected_tables" not in st.session_state:
    st.session_state.selected_tables = []

# Handle connection
if connect_button:
    try:
        from urllib.parse import quote_plus
        db_password_escaped = quote_plus(db_password)
        database_url = f"mysql+pymysql://{db_user}:{db_password_escaped}@{db_host}:{db_port}/{db_name}"
        
        with st.spinner("Connecting to database..."):
            engine = create_engine(database_url)
            # Test connection
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT DATABASE();"))
                db = result.fetchone()[0]
                st.sidebar.success(f"Connected to: {db}")
            
            # Get all available tables
            inspector = inspect(engine)
            all_tables = inspector.get_table_names()
            
            st.session_state.engine = engine
            st.session_state.available_tables = sorted(all_tables)
            st.session_state.selected_tables = []  # Reset selection
            
    except Exception as e:
        st.sidebar.error(f"Connection failed: {str(e)}")
        st.session_state.engine = None
        st.session_state.available_tables = []
        st.session_state.selected_tables = []

# Table selection
if st.session_state.available_tables:
    st.sidebar.subheader("Select Tables to Query")
    
    # Filter to show only tables from the default allowed set
    default_tables = sorted(DEFAULT_ALLOWED_TABLES)
    available_default = [t for t in st.session_state.available_tables if t in default_tables]
    
    if available_default:
        selected = st.sidebar.multiselect(
            "Available tables:",
            options=available_default,
            default=st.session_state.selected_tables,
            help="Select the tables you want the LLM to be able to query"
        )
        st.session_state.selected_tables = selected
    else:
        st.sidebar.warning("No matching tables found in database")
else:
    st.sidebar.info("Connect to database to load tables")

# Query input form
st.sidebar.subheader("Ask a Question")
with st.sidebar.form("query_form"):
    user_query = st.text_area(
        "Enter your question:",
        height=150,
        placeholder="e.g. What were the top 5 most rented films?",
        disabled=not st.session_state.selected_tables
    )
    submitted = st.form_submit_button("Submit", disabled=not st.session_state.selected_tables)

if not st.session_state.selected_tables:
    st.sidebar.warning("Please connect and select tables first")

# ---------------------------------------------------------------------------
# Main area: chat interface
# ---------------------------------------------------------------------------

st.title("LLM based Analytics Assistant")

st.markdown(
    """
    This is a Streamlit app that allows you to interact with a Large Language Model (LLM) to perform analytics tasks. 
    Enter your database credentials in the sidebar, select tables, and ask questions about your data.
    """
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # Assistant message - show SQL, answer, and results
            if message.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(message["sql"], language="sql")
            st.write(message.get("content", message.get("answer", "")))
            if message.get("result") is not None and not message["result"].empty:
                with st.expander("Query Results"):
                    st.dataframe(message["result"])

# Handle new submission
if submitted and user_query.strip() and st.session_state.engine and st.session_state.selected_tables:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    try:
        # Get schema info for selected tables only and store in session state
        schema_info = get_schema_info(st.session_state.engine, set(st.session_state.selected_tables))
        st.session_state.schema_info = schema_info
        
        # Get response from backend with selected tables
        with st.spinner("Generating SQL and querying the database..."):
            result = ask_database(user_query, allowed_tables=set(st.session_state.selected_tables))
        
        # Add assistant response to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sql": result.get("sql"),
            "result": result.get("result"),
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)}",
            "sql": None,
            "result": None,
        })
    
    # Rerun to display the new messages
    st.rerun()
elif submitted and not user_query.strip():
    st.sidebar.warning("Please enter a question first.")

# ---------------------------------------------------------------------------
# Show schema info in an expander (useful for debugging)
# ---------------------------------------------------------------------------

if st.session_state.schema_info:
    with st.expander("Database Schema"):
        st.text(st.session_state.schema_info)
elif st.session_state.engine:
    with st.expander("Database Schema"):
        st.info("Select tables to view schema")