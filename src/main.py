"""Run the full RAG pipeline end to end from the project root."""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from src import (
    process_all_pdfs,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    GroqLLM,
    AdvancedRAGPipeline,
    print_rag_result,
)

load_dotenv()

# Step 1 — Load PDFs
all_pdf_documents = process_all_pdfs("data")

# Step 2 — Chunk
chunks = split_documents(all_pdf_documents)

# Step 3 — Embedding manager
embedding_manager = EmbeddingManager()

# Step 4 — Vector store
vectorstore = VectorStore(persist_directory="data/vector_store")

# Step 4b — Embed chunks and index into the vector store
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)
vectorstore.add_documents(chunks, embeddings)

# Step 5 — Retriever
rag_retriever = RAGRetriever(vectorstore, embedding_manager)

# Step 6 — LLM
llm = GroqLLM(model_name="llama-3.3-70b-versatile")

# Step 7 — Advanced pipeline
adv_rag = AdvancedRAGPipeline(retriever=rag_retriever, llm=llm)

# Run a query
query = "If you were building a question answering system today, which components from the Transformer, BERT, and RAG papers would you combine and why?"

result = adv_rag.query(
    question=query,
    top_k=3,
    min_score=0.1,
    summarize=True
)

print_rag_result(result)
