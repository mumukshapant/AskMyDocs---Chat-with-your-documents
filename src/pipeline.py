"""End-to-end RAG pipeline with citations, history, and summarization."""

from typing import Any, Dict


class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        summarize: bool = False,
        per_file: bool = False
    ) -> Dict[str, Any]:

        if per_file:
            results = self.retriever.retrieve_per_file(
                question,
                top_k_per_file=top_k,
                score_threshold=min_score
            )
        else:
            results = self.retriever.retrieve(
                question,
                top_k=top_k,
                score_threshold=min_score
            )

        if not results:
            answer = "No relevant context found."
            sources = []
        else:
            context = "\n\n".join([doc["content"] for doc in results])

            sources = [
                {
                    "source": doc["metadata"].get(
                        "source_file",
                        doc["metadata"].get("source", "unknown")
                    ),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": round(doc["similarity_score"], 3),
                    "preview": doc["content"][:120] + "..."
                }
                for doc in results
            ]

            answer = self.llm.generate_response(
                query=question,
                context=context
            )

        # Add citations
        citations = [
            f"[{i+1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations)
            if citations
            else answer
        )

        # Optional summarization — no document context needed here
        summary = None
        if summarize and answer:
            summary = self.llm.generate_response_simple(
                query=f"Summarize the following answer in 2 concise sentences: {answer}",
                context=""
            )

        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary
        })

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history
        }
