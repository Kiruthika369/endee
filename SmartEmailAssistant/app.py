"""
app.py
------
Streamlit UI for the Voice-Based Smart Email Assistant.

Features:
  - Text query input
  - Display of semantically retrieved emails
  - AI-generated answer using RAG
  - Sidebar statistics and settings
  - Clean, professional UI
"""

import os
import json
import sys

import streamlit as st

# Ensure project utils are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.search import initialize_store, get_vector_store
from utils.rag_pipeline import run_rag_pipeline

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Email Assistant",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS for polished UI
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Email card styling */
    .email-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-left: 4px solid #4f8ef7;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .email-card:hover { border-left-color: #7eb8f7; }

    /* Score badge */
    .score-badge {
        background: #4f8ef7;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1a2340, #1e2130);
        border: 1px solid #4f8ef7;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 10px;
    }

    /* Category tag */
    .cat-tag {
        background: #252a3a;
        color: #a0aec0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        margin-right: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 10px 0 20px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load and initialize data (cached for performance)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading email index...")
def load_and_index_emails():
    """Load emails and build vector store — cached after first run."""
    email_path = os.path.join(os.path.dirname(__file__), "data", "sample_emails.json")
    with open(email_path, "r") as f:
        emails = json.load(f)
    store = initialize_store(emails)
    return emails, store


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider(
        "Number of emails to retrieve",
        min_value=1, max_value=6, value=3,
        help="How many emails the RAG pipeline will retrieve per query"
    )

    use_openai = st.toggle(
        "Use OpenAI API",
        value=False,
        help="Enable if you have an OPENAI_API_KEY environment variable set"
    )

    if use_openai:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.markdown("### 📊 Stats")

    try:
        emails, store = load_and_index_emails()
        st.metric("Emails Indexed", len(emails))
        st.metric("Embedding Model", "MiniLM-L6-v2")
        st.metric("Embedding Dims", "384")
        st.metric("Search Type", "Cosine Similarity")

        # Category breakdown
        st.markdown("### 📂 Categories")
        categories = {}
        for e in emails:
            cat = e.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            st.write(f"`{cat}` — {count} email(s)")
    except Exception:
        st.info("Loading data...")

    st.divider()
    st.markdown("""
    **How it works:**
    1. 📧 Emails → Embeddings (SentenceTransformers)
    2. 🔍 Query → Semantic Search (Cosine Similarity)
    3. 🤖 Context + Query → LLM Answer (RAG)
    """)


# ------------------------------------------------------------------
# Main App
# ------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>📬 Smart Email Assistant</h1>
    <p style="color: #718096; font-size: 16px;">
        Search your inbox using natural language · Powered by RAG + Semantic Search
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize data
try:
    emails, store = load_and_index_emails()
except FileNotFoundError:
    st.error("❌ `data/sample_emails.json` not found. Run `python main.py` first.")
    st.stop()

# ------------------------------------------------------------------
# Query Input
# ------------------------------------------------------------------

st.markdown("### 🔍 Ask a Question About Your Emails")

# Example queries as quick-select buttons
st.markdown("**Quick queries:**")
cols = st.columns(4)
quick_queries = [
    "Show me emails about internship",
    "What did my manager say about the project?",
    "Any security alerts in my inbox?",
    "What is my AWS billing amount?",
]
selected_quick = None
for i, col in enumerate(cols):
    with col:
        if st.button(quick_queries[i], use_container_width=True):
            selected_quick = quick_queries[i]

# Main text input
query = st.text_input(
    label="Or type your own question",
    value=selected_quick or "",
    placeholder="e.g. What did my manager say about the project deadline?",
    label_visibility="collapsed"
)

search_clicked = st.button("🚀 Search & Answer", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

if (search_clicked or selected_quick) and query.strip():
    with st.spinner("🔍 Searching emails and generating answer..."):
        result = run_rag_pipeline(query, top_k=top_k, use_openai=use_openai)

    # Two-column layout: emails on left, answer on right
    col_emails, col_answer = st.columns([1.2, 1], gap="large")

    with col_emails:
        st.markdown(f"### 📧 Retrieved Emails ({len(result['retrieved_emails'])})")
        st.caption(f"Semantic search results for: *\"{query}\"*")

        for email, score in result["retrieved_emails"]:
            relevance_pct = int(score * 100)
            category = email.get("category", "general")

            # Color the relevance bar
            bar_color = "#4f8ef7" if relevance_pct > 60 else "#f7924f" if relevance_pct > 40 else "#718096"

            st.markdown(f"""
            <div class="email-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <strong style="color:#e2e8f0; font-size:14px;">{email.get('subject','No Subject')}</strong>
                    <span class="score-badge" style="background:{bar_color};">{relevance_pct}% match</span>
                </div>
                <div style="color:#a0aec0; font-size:12px; margin-bottom:8px;">
                    📤 {email.get('sender','')} &nbsp;·&nbsp; 📅 {email.get('date','')}
                    &nbsp;<span class="cat-tag">{category}</span>
                </div>
                <p style="color:#cbd5e0; font-size:13px; line-height:1.5; margin:0;">
                    {email.get('body','')[:200]}{'...' if len(email.get('body','')) > 200 else ''}
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_answer:
        st.markdown("### 🤖 AI-Generated Answer")
        st.caption("Based on retrieved email context via RAG")

        st.markdown(f"""
        <div class="answer-box">
            <p style="color:#4f8ef7; font-size:12px; font-weight:bold; margin-bottom:8px;">
                💬 QUERY: {query}
            </p>
            <p style="color:#e2e8f0; font-size:14px; line-height:1.7; margin:0;">
                {result['answer'].replace(chr(10), '<br>')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Expandable context
        with st.expander("🔎 View raw context sent to LLM"):
            st.code(result["context_used"], language="text")

elif not query.strip() and search_clicked:
    st.warning("⚠️ Please enter a question before searching.")

else:
    # Landing state — show email inbox preview
    st.divider()
    st.markdown("### 📥 Your Inbox Preview")
    st.caption(f"{len(emails)} emails indexed and ready for semantic search")

    # Show all emails in a compact grid
    cols = st.columns(2)
    for i, email in enumerate(emails):
        with cols[i % 2]:
            category = email.get("category", "general")
            cat_colors = {
                "internship": "#48bb78",
                "project": "#4f8ef7",
                "academic": "#f6ad55",
                "security": "#fc8181",
                "business": "#b794f4",
                "billing": "#f6ad55",
                "career": "#68d391",
                "newsletter": "#76e4f7",
                "company": "#fbb6ce",
            }
            color = cat_colors.get(category, "#718096")

            st.markdown(f"""
            <div class="email-card" style="border-left-color:{color};">
                <div style="font-size:12px; color:{color}; font-weight:bold; margin-bottom:4px;">
                    {category.upper()}
                </div>
                <div style="font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:4px;">
                    {email.get('subject', '')[:55]}{'...' if len(email.get('subject',''))>55 else ''}
                </div>
                <div style="font-size:11px; color:#718096;">
                    {email.get('sender','')} · {email.get('date','')}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border:1px solid #2d3250; margin-top:40px;">
<p style="text-align:center; color:#4a5568; font-size:12px;">
    Smart Email Assistant · Built with SentenceTransformers + RAG · 
    <a href="https://github.com/endee-io/endee" style="color:#4f8ef7;">Endee Vector DB</a>
</p>
""", unsafe_allow_html=True)
