"""Simple Streamlit UI for the RAG pipeline: type a question, get an answer."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from src import EmbeddingManager, VectorStore, RAGRetriever, GroqLLM, AdvancedRAGPipeline, split_documents

load_dotenv()


@st.cache_resource
def load_components():
    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore(persist_directory="data/vector_store")
    retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = GroqLLM(model_name="openai/gpt-oss-120b")
    pipeline = AdvancedRAGPipeline(retriever=retriever, llm=llm)
    return embedding_manager, vectorstore, pipeline


st.title("AskMyDocs")

embedding_manager, vectorstore, pipeline = load_components()

# Widget to let the user upload one or more PDFs from their browser
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

def already_indexed(filename):
    """Check the vector store itself for existing chunks from this file,
    so re-uploading the same PDF (even in a new session) doesn't create duplicates."""
    existing = vectorstore.collection.get(where={"source_file": filename})
    return len(existing["ids"]) > 0


for uploaded_file in uploaded_files or []:
    if already_indexed(uploaded_file.name):
        continue

    # Save the uploaded file to data/pdf/ on local disk, same place process_all_pdfs() reads from
    dest = Path("data/pdf") / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    st.success(f"Saved to {dest}")

    # A new PDF isn't searchable yet — it has to be loaded, chunked, embedded,
    # and added to the vector store before questions can use it.
    with st.spinner(f"Updating vector store with '{uploaded_file.name}'..."):
        pages = PyPDFLoader(str(dest)).load()
        for page in pages:
            page.metadata["source_file"] = dest.name
            page.metadata["file_type"] = "pdf"

        chunks = split_documents(pages)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add_documents(chunks, embeddings)

    st.success(f"'{uploaded_file.name}' is now searchable.")

question = st.text_input("Ask a question about your documents:")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        result = pipeline.query(question=question, top_k=6, min_score=0.1, summarize=True, per_file=True)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Summary"):
        st.write(result["summary"])

    with st.expander("Sources"):
        for src in result["sources"]:
            st.markdown(f"**{src['source']}** (page {src['page']}, score {src['score']})")
            st.caption(src["preview"])

