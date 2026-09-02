"""Simple Streamlit UI for the RAG pipeline: type a question, get an answer."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src import EmbeddingManager, VectorStore, RAGRetriever, GroqLLM, AdvancedRAGPipeline

load_dotenv()


@st.cache_resource
def load_pipeline():
    embedding_manager = EmbeddingManager()
    vectorstore = VectorStore(persist_directory="data/vector_store")
    retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = GroqLLM(model_name="openai/gpt-oss-120b")
    return AdvancedRAGPipeline(retriever=retriever, llm=llm)


st.title("AskMyDocs")

pipeline = load_pipeline()

# Widget to let the user upload a PDF from their browser
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the uploaded file to data/pdf/ on local disk, same place process_all_pdfs() reads from
    dest = Path("data/pdf") / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    st.success(f"Saved to {dest}")

question = st.text_input("Ask a question about your documents:")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        result = pipeline.query(question=question, top_k=3, min_score=0.1, summarize=True)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Summary"):
        st.write(result["summary"])

    with st.expander("Sources"):
        for src in result["sources"]:
            st.markdown(f"**{src['source']}** (page {src['page']}, score {src['score']})")
            st.caption(src["preview"])

