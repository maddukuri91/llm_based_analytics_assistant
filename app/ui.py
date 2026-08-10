"""
Streamlit UI for the Natural Language → SQL pipeline.

Run with: streamlit run app/ui.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import inspect

try:
    import graphviz
except ImportError:
    graphviz = None


def create_er_diagram(engine, selected_tables):
    """Create an ER diagram using graphviz"""
    if not graphviz:
        return None
    
    inspector = inspect(engine)
    dot = graphviz.Digraph(comment='Database Schema')
    dot.attr(rankdir='LR', size='12,8')
    dot.attr('node', shape='record', style='filled', fillcolor='lightblue')
    
    # Add tables as nodes
    for table in selected_tables:
        columns = inspector.get_columns(table)
        col_str = "{" + table + "|"
        for i, col in enumerate(columns):
            col_type = str(col['type'])
            col_str += f"{col['name']} : {col_type}\\l"
        col_str += "}"
        dot.node(table, label=col_str)
    
    # Add foreign key relationships as edges
    for table in selected_tables:
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            referred_table = fk['referred_table']
            if referred_table in selected_tables:
                constrained = fk['constrained_columns']
                referred = fk['referred_columns']
                dot.edge(
                    f"{table}:{constrained[0]}",
                    f"{referred_table}:{referred[0]}",
                    style='dashed'
                )
    
    return dot

# Ensure the project root is on the path so `app.backend` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend import ask_database, get_schema_info, ALLOWED_TABLES as DEFAULT_ALLOWED_TABLES  # noqa: E402
from app.backend import create_db_engine  # noqa: E402

st.set_page_config(
    page_title="LLM based Analytics Assistant",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: LLM provider selection
# ---------------------------------------------------------------------------

st.sidebar.header("LLM Configuration")

llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    options=["Ollama", "Groq", "OpenAI"],
    index=0,
    help="Choose which LLM backend to use. Ollama runs locally; Groq and OpenAI require an API key.",
)

llm_model = ""

if llm_provider == "Groq":
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a key at https://console.groq.com/keys",
    )
    llm_model = st.sidebar.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="llama-3.3-70b-versatile is higher quality; llama-3.1-8b-instant is faster and lighter.",
    )
    if not groq_api_key:
        st.sidebar.warning("Please enter your Groq API key.")
elif llm_provider == "OpenAI":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get a key at https://platform.openai.com/api-keys",
    )
    llm_model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o is the latest flagship; gpt-3.5-turbo is lighter and faster.",
    )
    if not openai_api_key:
        st.sidebar.warning("Please enter your OpenAI API key.")
else:
    llm_model = st.sidebar.text_input("Model", value="qwen3:8b")
    st.sidebar.info("Ollama runs locally — no API key required.")

# Store LLM config in session state
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = llm_provider
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = ""
if "llm_model" not in st.session_state:
    st.session_state.llm_model = llm_model

st.session_state.llm_provider = llm_provider
st.session_state.llm_model = llm_model
if llm_provider == "Groq":
    st.session_state.llm_api_key = groq_api_key
elif llm_provider == "OpenAI":
    st.session_state.llm_api_key = openai_api_key
else:
    st.session_state.llm_api_key = ""

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
        with st.spinner("Connecting to database..."):
            engine = create_db_engine(db_user, db_password, db_host, db_port, db_name)
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
        # Add "All" option at the beginning
        table_options = ["All"] + available_default
        
        # Determine default selection
        if st.session_state.selected_tables == available_default or not st.session_state.selected_tables:
            default_selection = ["All"] if not st.session_state.selected_tables else ["All"]
        else:
            default_selection = st.session_state.selected_tables
        
        selected = st.sidebar.multiselect(
            "Available tables:",
            options=table_options,
            default=default_selection,
            help="Select 'All' to query all tables, or select specific tables"
        )
        
        # Handle "All" selection
        if "All" in selected:
            if len(selected) == 1:
                # Only "All" is selected, use all available tables
                st.session_state.selected_tables = available_default
            else:
                # "All" plus other tables selected - remove "All" and keep others
                selected.remove("All")
                st.session_state.selected_tables = selected
        else:
            st.session_state.selected_tables = selected
        
        # Load schema info when tables are selected
        if st.session_state.selected_tables and st.session_state.engine:
            try:
                st.session_state.schema_info = get_schema_info(st.session_state.engine, set(st.session_state.selected_tables))
            except Exception as e:
                st.session_state.schema_info = f"Error loading schema: {str(e)}"
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
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# Display chat history
for i, message in enumerate(st.session_state.messages):
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
                result_df = message["result"]
                with st.expander("Query Results"):
                    st.dataframe(result_df, use_container_width=True)
                
                # Add visualization section
                with st.expander("📊 Visualize Data"):
                    col1, col2 = st.columns(2)
                    with col1:
                        chart_type = st.selectbox(
                            "Chart Type",
                            ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Area Chart"],
                            key=f"chart_type_{i}"
                        )
                    with col2:
                        # Auto-suggest columns for axes
                        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
                        all_cols = result_df.columns.tolist()
                        
                        x_axis = st.selectbox("X Axis", all_cols, key=f"x_axis_{i}")
                        y_axis = st.selectbox("Y Axis", numeric_cols if numeric_cols else all_cols, key=f"y_axis_{i}")
                    
                    # Generate chart based on selection
                    try:
                        if chart_type == "Bar Chart":
                            fig = px.bar(result_df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                        elif chart_type == "Line Chart":
                            fig = px.line(result_df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                        elif chart_type == "Pie Chart":
                            fig = px.pie(result_df, names=x_axis, values=y_axis, title=f"{y_axis} by {x_axis}")
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(result_df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                        elif chart_type == "Area Chart":
                            fig = px.area(result_df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                        
                        fig.update_layout(autosize=True, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate chart: {str(e)}")

# Handle new submission
if submitted and user_query.strip() and st.session_state.engine and st.session_state.selected_tables:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    try:
        # Get schema info for selected tables only and store in session state
        schema_info = get_schema_info(st.session_state.engine, set(st.session_state.selected_tables))
        st.session_state.schema_info = schema_info
        
        # Validate API key requirement before calling backend
        if st.session_state.llm_provider in ("Groq", "OpenAI") and not st.session_state.llm_api_key:
            raise ValueError(f"Please enter your {st.session_state.llm_provider} API key.")
        
        # Check if there's a pending clarification to continue
        pending = st.session_state.get("pending_clarification")
        clarification_history = None
        api_user_query = user_query

        if pending:
            # This submission is an answer to a pending clarification question
            last_question = pending["last_question"]
            clarification_history = [
                *pending["history"],
                {"question": last_question, "answer": user_query.strip()},
            ]
            api_user_query = pending["original_question"]

        # Get response from backend with selected tables and LLM config
        with st.spinner("Generating SQL and querying the database..."):
            result = ask_database(
                api_user_query,
                db_engine=st.session_state.engine,
                provider=st.session_state.llm_provider.lower(),
                api_key=st.session_state.llm_api_key or None,
                model=st.session_state.llm_model or None,
                allowed_tables=set(st.session_state.selected_tables),
                clarification_history=clarification_history,
            )

        # Handle clarification flow — store pending state so the next user
        # submission is correctly appended to the conversation history.
        if result["status"] == "needs_clarification":
            st.session_state.pending_clarification = {
                "original_question": api_user_query,
                "history": clarification_history or [],
                "last_question": result.get("clarification_question", result["answer"]),
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sql": None,
                "result": None,
                "is_clarification": True,
            })
        else:
            # Clear any pending clarification
            st.session_state.pending_clarification = None
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
        st.session_state.pending_clarification = None

    # Rerun to display the new messages
    st.rerun()
elif submitted and not user_query.strip():
    st.sidebar.warning("Please enter a question first.")

# ---------------------------------------------------------------------------
# Show schema info in an expander (useful for debugging)
# ---------------------------------------------------------------------------

if st.session_state.schema_info:
    with st.expander("Database Schema"):
        if graphviz and st.session_state.engine and st.session_state.selected_tables:
            try:
                er_diagram = create_er_diagram(st.session_state.engine, st.session_state.selected_tables)
                if er_diagram:
                    st.graphviz_chart(er_diagram)
                else:
                    st.info("Could not generate ER diagram")
            except Exception as e:
                st.error(f"Error generating ER diagram: {str(e)}")
        else:
            st.info("Install graphviz to view ER diagrams: pip install graphviz")
elif st.session_state.engine:
    with st.expander("Database Schema"):
        st.info("Select tables to view schema")