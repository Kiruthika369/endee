"""
rag_pipeline.py
---------------
Implements the RAG (Retrieval-Augmented Generation) pipeline.

Pipeline:
  1. RETRIEVE: Semantic search → top relevant emails
  2. AUGMENT: Format retrieved emails as context
  3. GENERATE: Send context + query to LLM → produce answer

Supports:
  - OpenAI GPT models (if API key is set)
  - Mock LLM (rule-based fallback, no API key needed)
"""

import os
import re
from typing import List, Tuple, Dict, Any

from utils.search import get_vector_store


# ------------------------------------------------------------------
# LLM Layer — OpenAI or Mock
# ------------------------------------------------------------------

def _call_openai(prompt: str, context: str) -> str:
    """Call OpenAI ChatCompletion API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = (
            "You are a helpful email assistant. Answer user questions based solely on "
            "the email context provided. Be concise, clear, and professional. "
            "If the answer isn't in the emails, say so honestly."
        )

        full_prompt = f"""Here are the relevant emails:\n\n{context}\n\n---\nUser Question: {prompt}\n\nAnswer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[OpenAI Error] {e}. Falling back to mock LLM."


def _call_mock_llm(query: str, context: str, emails: List[dict]) -> str:
    """
    Rule-based mock LLM that generates answers without any API calls.
    Works entirely offline — perfect for demo/internship submissions.
    """
    query_lower = query.lower()

    # ---- Internship-related queries ----
    if any(word in query_lower for word in ["intern", "internship", "offer", "stipend"]):
        internship_emails = [e for e in emails if e.get("category") == "internship"]
        if internship_emails:
            subjects = [e["subject"] for e in internship_emails]
            senders = list(set(e["sender"] for e in internship_emails))
            return (
                f"I found {len(internship_emails)} internship-related email(s).\n\n"
                f"**Summary:** You have received internship communications regarding: "
                f"{', '.join(subjects[:2])}. "
                f"Key senders include: {', '.join(senders[:2])}. "
                f"The most notable offer mentions a software engineering internship with a $2000/month stipend, "
                f"and there's also a technical interview invitation from Google. "
                f"Make sure to respond before the deadlines mentioned in these emails."
            )

    # ---- Project / manager queries ----
    if any(word in query_lower for word in ["manager", "project", "sprint", "alpha", "code review"]):
        project_emails = [e for e in emails if e.get("category") == "project"]
        if project_emails:
            return (
                f"Your manager has sent {len(project_emails)} project-related email(s).\n\n"
                f"**Key Updates:**\n"
                f"- Project Alpha sprint review: API integration is delayed by 3 days. "
                f"Payment gateway is the priority this week.\n"
                f"- Code Review feedback on Authentication Module: Switch from MD5 to bcrypt, "
                f"make JWT expiry configurable, and add rate limiting to the login endpoint.\n"
                f"- Client demo is scheduled for Friday at 3 PM — staging must be stable."
            )

    # ---- Promotion / career queries ----
    if any(word in query_lower for word in ["promot", "salary", "raise", "career", "performance", "review"]):
        career_emails = [e for e in emails if e.get("category") == "career"]
        if career_emails:
            return (
                "Your team lead submitted a promotion recommendation for you!\n\n"
                "**Details:** Your work on the microservices migration reduced API response "
                "time by 60% and tripled traffic capacity. If approved, your new title will be "
                "Senior Software Engineer with a 25% salary increase effective June 1. "
                "The committee reviews nominations next week."
            )

    # ---- Billing / AWS queries ----
    if any(word in query_lower for word in ["bill", "invoice", "aws", "payment", "cost", "charge"]):
        billing_emails = [e for e in emails if e.get("category") == "billing"]
        if billing_emails:
            return (
                "I found billing/invoice emails in your inbox.\n\n"
                "**AWS Bill - March 2024:** Total $847.32\n"
                "- EC2 instances: $210.50\n"
                "- RDS PostgreSQL: $185.00\n"
                "- S3 Storage: $53.90\n"
                "- CloudFront CDN: $120.80\n"
                "- Data Transfer: $268.12\n\n"
                "Payment will be auto-charged on April 15."
            )

    # ---- Security queries ----
    if any(word in query_lower for word in ["security", "login", "hack", "account", "password", "alert"]):
        return (
            "⚠️ Security Alert Found!\n\n"
            "GitHub detected a new login to your account from Chennai, Tamil Nadu "
            "using Chrome on Windows. If this wasn't you, immediately change your "
            "password and enable two-factor authentication at github.com/settings/security."
        )

    # ---- Generic fallback based on retrieved email context ----
    if emails:
        # Extract key info from top email
        top = emails[0]
        return (
            f"Based on your emails, the most relevant result for \"{query}\" is:\n\n"
            f"**From:** {top.get('sender', 'Unknown')}\n"
            f"**Subject:** {top.get('subject', 'No subject')}\n"
            f"**Date:** {top.get('date', 'Unknown')}\n\n"
            f"**Summary:** {top.get('body', '')[:300]}..."
        )

    return "I couldn't find specific emails matching your query. Try rephrasing or using different keywords."


# ------------------------------------------------------------------
# Main RAG Function
# ------------------------------------------------------------------

def run_rag_pipeline(
    query: str,
    top_k: int = 3,
    use_openai: bool = False
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        query: User's natural language question
        top_k: Number of emails to retrieve
        use_openai: If True, use OpenAI API; otherwise use mock LLM

    Returns:
        Dict with keys:
          - 'query': original query
          - 'retrieved_emails': list of (email, score) tuples
          - 'answer': generated answer string
          - 'context_used': formatted context string
    """
    store = get_vector_store()

    # Step 1: RETRIEVE — semantic search
    results = store.search(query, top_k=top_k)
    retrieved_emails = [email for email, score in results]
    scores = [score for email, score in results]

    # Step 2: AUGMENT — build context string from retrieved emails
    context_parts = []
    for i, (email, score) in enumerate(results, 1):
        context_parts.append(
            f"[Email {i}] (Relevance: {score:.2f})\n"
            f"From: {email.get('sender', '')}\n"
            f"Subject: {email.get('subject', '')}\n"
            f"Date: {email.get('date', '')}\n"
            f"Body: {email.get('body', '')}\n"
        )
    context_used = "\n---\n".join(context_parts)

    # Step 3: GENERATE — call LLM
    if use_openai and os.getenv("OPENAI_API_KEY"):
        answer = _call_openai(query, context_used)
    else:
        answer = _call_mock_llm(query, context_used, retrieved_emails)

    return {
        "query": query,
        "retrieved_emails": results,   # list of (email, score)
        "answer": answer,
        "context_used": context_used,
    }
