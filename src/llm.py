"""Groq LLM wrapper for answer generation."""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


class GroqLLM:
    def __init__(self, model_name: str = "gemma2-9b-it", api_key: str = None):
        """
        Initialize Groq LLM

        Args:
            model_name: Groq model name (qwen2-72b-instruct, llama3-70b-8192, etc.)
            api_key: Groq API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Groq API key is required.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )

        print(f"Initialized Groq LLM with model: {self.model_name}")

    # "context" is the relevant text chunks you pulled from your documents and are handing to the LLM alongside the question.
    # context generated using :: `rag_retriever.retrieve("..")`
    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        Returns:
            Generated response string
        """

        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context as your primary source, and supplement with your own knowledge where the context is incomplete.
            Context:
            {context}

            Question: {question}

            Answer:"""
        )

        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting
        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}
        Question: {query}
        Answer:"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
