"""Simple Streamlit UI for the RAG pipeline: type a question, get an answer."""

import warnings
warnings.filterwarnings("ignore")

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

    with st.expander("History"):
        for turn in reversed(result["history"]):
            st.markdown(f"**Q:** {turn['question']}")
            st.write(turn["answer"])
