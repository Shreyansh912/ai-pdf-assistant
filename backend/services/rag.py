from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore


class RAGService:
    """
    Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline:
    Question -> Query Embedding -> Vector Retrieval -> Context Assembly -> LLM Generation.
    """

    SYSTEM_INSTRUCTION = (
        "You are an AI assistant that answers questions about uploaded PDF documents.\n"
        "Use only the information provided in the retrieved context.\n"
        "Do not invent facts or extrapolate beyond what is stated.\n"
        "If the answer cannot be found in the provided context, reply exactly:\n"
        "\"I couldn't find this information in the uploaded document.\"\n"
        "Answer clearly and concisely."
    )

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        api_key: str,
        model_name: str = "gemini-2.5-flash"
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.model_name = model_name
        self.api_key = api_key

        # Initialize the Google GenAI client if an API key is provided
        self.client: Optional[genai.Client] = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def _build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunk dictionaries into an indexed context block.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            filename = chunk.get("filename", "unknown")
            page = chunk.get("page_number", 0)
            text = chunk.get("text", "").strip()
            context_parts.append(f"[Excerpt {i} | File: {filename} | Page: {page}]\n{text}")

        return "\n\n".join(context_parts)

    def _extract_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts deduplicated filename and page combinations from retrieved chunks.
        """
        seen = set()
        sources = []
        for chunk in retrieved_chunks:
            filename = chunk.get("filename")
            page = chunk.get("page_number")
            if filename and page:
                key = (filename, page)
                if key not in seen:
                    seen.add(key)
                    sources.append({"filename": filename, "page": page})
        return sources

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end RAG question answering.

        Args:
            question: The user's query string.
            top_k: Number of chunks to retrieve from FAISS.

        Returns:
            Dict containing:
            - answer: str
            - sources: List[Dict[str, Any]]
        """
        question = question.strip()
        if not question:
            return {
                "answer": "Please ask a valid question.",
                "sources": []
            }

        # 1. Embed query
        query_vec = self.embedding_service.generate_embeddings(question)

        # 2. Retrieve top chunks from FAISS
        retrieved_chunks = self.vector_store.search(query_vec, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": []
            }

        # 3. Assemble formatted context
        context_text = self._build_context(retrieved_chunks)

        # 4. Verify API key
        if not self.client:
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        # 5. Call LLM with strict grounding prompt
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.0  # Zero temperature for deterministic, hallucination-free answers
            )
        )

        raw_answer = response.text.strip() if response.text else ""

        # 6. If the model indicates missing info, omit irrelevant sources
        if "I couldn't find this information" in raw_answer:
            sources = []
        else:
            sources = self._extract_sources(retrieved_chunks)

        return {
            "answer": raw_answer,
            "sources": sources
        }
