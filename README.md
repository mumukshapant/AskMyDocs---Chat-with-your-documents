# AskMyDocs — RAG Sandbox

A Retrieval-Augmented Generation (RAG) pipeline for asking questions over a folder of PDFs. It loads PDFs, chunks them, embeds the chunks, indexes them in a vector store, retrieves relevant chunks for a question, and uses an LLM (via Groq) to generate a cited answer.

## Architecture (high level)

Load PDFs → chunk → embed → store (one-time ingestion) → retrieve → generate → format (per-query).

## UI/UX

A [Streamlit](https://streamlit.io/) app ([`app.py`](app.py)) provides a simple interface: a text input for the question, a Submit button, and the answer displayed below it, with collapsible sections for the summary, sources, and question history.

## Prompt

The LLM ([`src/llm.py`](src/llm.py)) is instructed to answer **only** from the retrieved context — not from its own general knowledge — and to say explicitly when the context doesn't contain enough information, rather than guessing. Exact template used:

```
You are a helpful AI assistant. Answer the question using ONLY the context below — do not use outside knowledge. If the context does not contain enough information to answer, say so explicitly instead of guessing. Keep the answer concise (a few sentences, unless the question needs more detail).
Context:
{context}

Question: {question}

Answer:
```

## Setup

1. Create/activate the virtual environment and install dependencies:
   ```bash
   python3 -m venv mp_personal_env
   source mp_personal_env/bin/activate
   pip install -r requirements.txt
   ```

2. Add your Groq API key to `.env` in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get a key at [console.groq.com](https://console.groq.com/keys).

## Input

- **Documents:** drop PDF files into `data/pdf/`. All PDFs in that folder (recursively) are loaded and indexed.
- **Question:** entered directly in the app's text input at runtime.

## Output

- **Vector index:** persisted to `data/vector_store/` (ChromaDB).
- **Answer:** displayed in the app — includes the generated answer with citations, an optional 2-sentence summary, the source chunks (file, page, similarity score, preview) used to answer, and the running question history for the session.

## How to run

```bash
source mp_personal_env/bin/activate
streamlit run app.py
```

This starts the app locally, deployed at **http://localhost:8501**. Open that URL in a browser, type a question, and hit Submit.
