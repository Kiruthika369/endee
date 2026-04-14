"""
main.py
-------
Initialization script for the Smart Email Assistant.

Run this ONCE before launching the Streamlit app:
    python main.py

What it does:
  1. Loads sample email data from data/sample_emails.json
  2. Generates embeddings for all emails
  3. Stores embeddings in the vector store (vector_store/email_store.pkl)
  4. Runs a quick sanity-check search to confirm everything works
"""

import json
import os
import sys

# Add project root to path so utils imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.search import initialize_store
from utils.rag_pipeline import run_rag_pipeline


def load_emails(path: str = "data/sample_emails.json") -> list:
    """Load emails from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        emails = json.load(f)
    print(f"[Main] Loaded {len(emails)} emails from {path}")
    return emails


def run_demo_queries(store) -> None:
    """Run a few demo queries to verify the pipeline works."""
    demo_queries = [
        "Show me emails about internship",
        "What did my manager say about the project?",
        "Any security alerts?",
        "What is my AWS bill?",
    ]

    print("\n" + "="*60)
    print("DEMO: Running sample queries")
    print("="*60)

    for query in demo_queries:
        print(f"\n🔍 Query: {query}")
        result = run_rag_pipeline(query, top_k=2)

        print("📧 Retrieved Emails:")
        for email, score in result["retrieved_emails"]:
            print(f"   [{score:.2f}] {email['subject']} (from {email['sender']})")

        print(f"\n🤖 AI Answer:\n{result['answer'][:200]}...")
        print("-"*50)


def main():
    print("="*60)
    print("  Smart Email Assistant — Initialization")
    print("="*60)

    # 1. Load email data
    emails = load_emails()

    # 2. Build/load vector store
    print("\n[Main] Building vector store...")
    store = initialize_store(emails, force_rebuild=True)
    print(f"[Main] Vector store ready with {len(store.emails)} emails.")

    # 3. Run demo queries
    run_demo_queries(store)

    print("\n✅ Initialization complete!")
    print("🚀 Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
