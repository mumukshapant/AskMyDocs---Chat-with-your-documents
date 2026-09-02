# AskMyDocs — RAG Sandbox

A Retrieval-Augmented Generation (RAG) pipeline for asking questions over a folder of PDFs. It loads PDFs, chunks them, embeds the chunks, indexes them in a vector store, retrieves relevant chunks for a question, and uses an LLM (via Groq) to generate a cited answer.

## Architecture

**Mental model:** load → chunk → embed → store (one-time ingestion) → retrieve → generate → format (per-query).

**Entry point:** [`src/main.py`](src/main.py) — run with `python -m src.main`. It calls every step below in sequence.

### Step 1 — Load PDFs
`process_all_pdfs("data")` → [`src/loader.py:14`](src/loader.py#L14)
Finds every PDF under `data/`, extracts text page-by-page with `PyPDFLoader`, tags each page with metadata (`source_file`, `file_type`).
*Why first:* nothing downstream has any input until raw text exists — LLMs/embedders can't read PDF binary directly.

### Step 2 — Chunk
`split_documents(...)` → [`src/loader.py:46`](src/loader.py#L46)
Breaks each page into ~1000-character pieces with 200-character overlap using `RecursiveCharacterTextSplitter`.
*Why:* a whole paper is too big to embed meaningfully or fit in an LLM prompt; small chunks let you retrieve just the relevant paragraph. Overlap stops a fact from being cut in half at a boundary.

### Step 3 — Embedding manager
`EmbeddingManager()` → [`src/embeddings.py:9`](src/embeddings.py#L9)
Loads `all-MiniLM-L6-v2` (sentence-transformers), which turns text into 384-dim vectors.
*Why:* this is what enables semantic (meaning-based) search instead of keyword matching.

### Step 4 — Vector store
`VectorStore(persist_directory="data/vector_store")` → [`src/vector_store.py:11`](src/vector_store.py#L11)
Initializes a persistent ChromaDB collection using cosine distance.
*Why:* need somewhere to store and efficiently search vectors, and persistence avoids re-embedding everything on every run.

### Step 4b — Embed + index chunks
`embedding_manager.generate_embeddings(texts)` then `vectorstore.add_documents(chunks, embeddings)` → [`src/vector_store.py:50`](src/vector_store.py#L50)
Converts every chunk to a vector and writes text + metadata + vector into Chroma.
*Why:* this is the actual "build the knowledge base" step — without it the store is empty.

### Step 5 — Retriever
`RAGRetriever(vectorstore, embedding_manager)` → [`src/retriever.py:9`](src/retriever.py#L9)
At query time: embeds the question the same way, does similarity search in Chroma, converts distance → similarity score, filters by threshold.
*Why:* this is the "R" (Retrieval) in RAG — finds the chunks most relevant to what was asked.

### Step 6 — LLM
`GroqLLM(model_name=...)` → [`src/llm.py:10`](src/llm.py#L10)
Wraps Groq's chat model; builds a prompt from context + question and generates an answer.
*Why:* this is the "G" (Generation) — turns raw retrieved text into a coherent answer.

### Step 7 — Pipeline orchestration
`AdvancedRAGPipeline(retriever, llm)` → [`src/pipeline.py:6`](src/pipeline.py#L6)
`.query()` ties it together: retrieve → build context → generate answer → attach citations → optionally summarize → save to history → return result dict.

### Step 8 — Run + print
`adv_rag.query(question=query, ...)` then `print_rag_result(result)` in [`src/formatting.py`](src/formatting.py) — displays question, answer, summary, sources.

## Setup

1. Create/activate the virtual environment and install dependencies:
   ```bash
   python3 -m venv mp_personal_env # to create 
   source mp_personal_env/bin/activate
   pip install -r requirements.txt
   ```

2. Add your Groq API key to `.env` in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Get a key at [console.groq.com](https://console.groq.com/keys).

## Input

- **Documents:** drop PDF files into `data/pdf/`. All PDFs in that folder (recursively) are loaded and indexed on each run.
- **Question:** currently hardcoded — the pipeline does not accept input at runtime. Edit the `query` variable in [`src/main.py`](src/main.py#L48) (or the corresponding cell in the notebook) to the question you want to ask, then rerun.

## Output

- **Vector index:** persisted to `data/vector_store/` (ChromaDB). This is rebuilt/appended to on each run of the ingestion steps.
- **Answer:** printed to the terminal/notebook output — includes the question, generated answer with citations, an optional 2-sentence summary, and the source chunks (file, page, similarity score, preview) used to answer.

## How to run

### Option A — Script
```bash
source mp_personal_env/bin/activate
python -m src.main
```
This re-runs the full ingestion pipeline (load → chunk → embed → index) and then answers the hardcoded `query` in `main.py`.

### Option B — Jupyter Notebook
```bash
source mp_personal_env/bin/activate
jupyter notebook notebook/demo.ipynb
```
Run the cells top to bottom. The notebook walks through the same steps interactively and lets you experiment with different questions/parameters (`top_k`, `min_score`) without re-running the whole script each time.
