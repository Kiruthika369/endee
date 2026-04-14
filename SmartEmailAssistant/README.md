# 📬 Voice-Based Smart Email Assistant
### Powered by RAG · Semantic Search · Endee Vector Database

> An AI-powered email assistant that lets you search and understand your inbox using plain English — no keywords, no filters, just ask.

![Demo Screenshot](assets/demo.png)

---

## 🌟 Project Overview

The Smart Email Assistant is a full-stack AI application that transforms how you interact with your emails. Instead of manually filtering or keyword-searching, you simply ask questions like *"What did my manager say about the project?"* or *"Show me internship offers"* — and the system retrieves the most relevant emails and generates a concise, intelligent answer.

This project combines **Semantic Embeddings**, **Vector Search**, and **Retrieval-Augmented Generation (RAG)** into a clean Streamlit web application.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Understands meaning, not just keywords |
| 🤖 **RAG Pipeline** | Retrieves emails → generates AI answers |
| 📧 **12 Sample Emails** | Realistic data across 7 categories |
| 🚀 **Zero API Required** | Works fully offline with mock LLM |
| 🔑 **OpenAI Optional** | Plug in GPT-3.5 for richer answers |
| 📊 **Streamlit UI** | Clean, interactive web interface |
| 💾 **Persistent Store** | Embeddings cached; no re-indexing on restart |

---

## 🏗️ Architecture

```
User Query (text)
       │
       ▼
┌─────────────────────┐
│   Embedder          │  SentenceTransformers (all-MiniLM-L6-v2)
│   embed_text()      │  Converts query → 384-dim vector
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EmailVectorStore   │  Endee / NumPy cosine similarity
│  search(query, k)   │  Retrieves top-K emails by semantic similarity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG Pipeline      │  Combines retrieved email context
│   run_rag_pipeline()│  Sends to LLM → structured answer
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI      │  Displays emails + AI answer
│   app.py            │  Interactive, real-time
└─────────────────────┘
```

---

## 🧠 How Endee is Used

[Endee](https://github.com/endee-io/endee) is the vector database layer of this project. It stores high-dimensional email embeddings and enables fast approximate nearest-neighbor search:

1. **Indexing Phase** (`python main.py`):  
   Each email's subject + body is converted into a 384-dimensional embedding via `SentenceTransformer`. These embeddings are stored in the Endee vector store.

2. **Query Phase** (Streamlit app):  
   The user's query is embedded using the same model. Endee performs cosine similarity search against all stored email embeddings, returning the most semantically relevant results.

3. **RAG Phase**:  
   Retrieved email content is assembled into a context window and sent to the LLM (OpenAI or mock), which produces a grounded, factual answer.

> The fallback implementation uses NumPy-based cosine similarity, so the project runs 100% offline even without the Endee server.

---

## 📁 Project Structure

```
endee/
└── smart-email-assistant/
    ├── app.py                  ← Streamlit web UI
    ├── main.py                 ← Initialization script (run once)
    ├── requirements.txt        ← Python dependencies
    ├── README.md               ← This file
    │
    ├── data/
    │   └── sample_emails.json  ← 12 realistic emails across 7 categories
    │
    ├── vector_store/
    │   └── email_store.pkl     ← Auto-generated: persisted embeddings
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── embedder.py         ← SentenceTransformer embedding functions
    │   ├── search.py           ← Vector store + cosine similarity search
    │   └── rag_pipeline.py     ← RAG orchestration + LLM integration
    │
    └── assets/
        └── demo.png            ← Demo screenshot
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1 — Clone / Download the project

```bash
cd endee/smart-email-assistant
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80 MB) on first run. This happens automatically.

### Step 3 — Initialize the vector store

```bash
python main.py
```

This will:
- Load the 12 sample emails
- Generate embeddings for all of them
- Save the vector store to `vector_store/email_store.pkl`
- Run 4 demo queries to verify everything works

Expected output:
```
[Main] Loaded 12 emails from data/sample_emails.json
[Embedder] Loading model: all-MiniLM-L6-v2 ...
[VectorStore] Building index for 12 emails...
[VectorStore] Saved to vector_store/email_store.pkl
✅ Initialization complete!
🚀 Run the app with: streamlit run app.py
```

### Step 4 — Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 Optional: OpenAI Integration

For richer, more nuanced answers, set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-your-key-here
streamlit run app.py
```

Then toggle **"Use OpenAI API"** in the Streamlit sidebar.

> Without an API key, the app uses a smart rule-based mock LLM that works offline and still demonstrates the full RAG pipeline.

---

## 💬 Example Queries

| Query | What it retrieves |
|---|---|
| `Show me emails about internship` | TechCorp offer, Google interview, RAG project |
| `What did my manager say about the project?` | Sprint review, code review feedback |
| `Any security alerts?` | GitHub login alert |
| `What is my AWS billing amount?` | AWS invoice for March 2024 |
| `Tell me about my promotion` | Team lead's recommendation email |
| `Latest news about AI` | AI Weekly newsletter |

---

## 📸 Demo

The app features a split-panel interface:

- **Left panel**: Top-K semantically retrieved emails with relevance scores
- **Right panel**: AI-generated answer combining context from all retrieved emails
- **Sidebar**: Stats, categories, pipeline explanation, settings

![Smart Email Assistant Demo](assets/demo.png)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | Endee / NumPy cosine similarity |
| LLM | OpenAI GPT-3.5 / Mock LLM |
| UI | Streamlit |
| Language | Python 3.9+ |
| Persistence | pickle (local file) |

---

## 🔮 Future Improvements

- [ ] Voice input via Web Speech API or Whisper
- [ ] Gmail / Outlook OAuth integration
- [ ] Real Endee server deployment
- [ ] Email classification and auto-tagging
- [ ] Multi-turn conversation memory

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as an internship submission project demonstrating RAG, semantic search, and full-stack Python development.*
