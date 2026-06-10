"""FastAPI backend for AskMyDocs — ask questions, list sources, upload PDFs."""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make the project root importable so `from src import ...` works.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src import (
    process_all_pdfs,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    GroqLLM,
    AdvancedRAGPipeline,
)

# ---- Configuration ----
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

load_dotenv(PROJECT_ROOT / ".env")

# ---- Build the pipeline once, when the server starts ----
embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL)
vectorstore = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
llm = GroqLLM(model_name=LLM_MODEL)
pipeline = AdvancedRAGPipeline(retriever=rag_retriever, llm=llm)

app = FastAPI(title="AskMyDocs")


# ---- The request body for /query ----
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    min_score: float = 0.2
    summarize: bool = False


# ---- Ask a question ----
@app.post("/query")
def query(request: QueryRequest):
    result = pipeline.query(
        question=request.question,
        top_k=request.top_k,
        min_score=request.min_score,
        summarize=request.summarize,
    )
    return {
        "question": result["question"],
        "answer": result["answer"],
        "summary": result["summary"],
        "sources": result["sources"],
    }


# ---- List the PDFs that are indexed ----
@app.get("/papers")
def list_papers():
    records = vectorstore.collection.get(include=["metadatas"])

    names = set()
    for metadata in records["metadatas"]:
        names.add(metadata.get("source_file", "unknown"))

    return {"papers": sorted(names)}


# ---- Upload a new PDF and add it to the index ----
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the upload into a temp folder so we can reuse process_all_pdfs on it.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / file.filename
        with open(temp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        documents = process_all_pdfs(temp_dir)
        chunks = split_documents(documents)

        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add_documents(chunks, embeddings)

    return {"filename": file.filename, "chunks_added": len(chunks)}