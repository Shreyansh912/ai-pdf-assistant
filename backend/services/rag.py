import json
from typing import AsyncGenerator, Dict, Any, List, Optional
from google import genai
from google.genai import types


class RAGService:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.6-flash",
        embedding_service: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        top_k: int = 5,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k

    def _build_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        formatted_chunks = "\n\n".join(
            f"[Doc: {c.get('filename', 'doc')} | Page: {c.get('page') or c.get('page_number', 1)}]\n{c['text']}"
            for c in context_chunks
        )
        return f"""You are a precise, document-grounded AI assistant.
Answer the user's question STRICTLY based on the context snippets below.
If the context does not contain sufficient facts to answer, respond with exactly:
"I couldn't find this information in the uploaded document."

--- CONTEXT ---
{formatted_chunks}

--- USER QUESTION ---
{query}
"""

    def answer_query(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if context_chunks is None:
            if self.embedding_service and self.vector_store:
                q_vec = self.embedding_service.encode([query])
                context_chunks = self.vector_store.search(q_vec, top_k=self.top_k)
            else:
                context_chunks = []

        if not context_chunks:
            return {
                "answer": "I couldn't find this information in the uploaded document.",
                "sources": [],
            }

        prompt = self._build_prompt(query, context_chunks)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        answer_text = response.text or "I couldn't find this information in the uploaded document."

        if "I couldn't find this information" in answer_text:
            return {"answer": answer_text, "sources": []}

        seen = set()
        sources = []
        for c in context_chunks:
            filename = c.get("filename", "document.pdf")
            page = c.get("page") or c.get("page_number", 1)
            doc_id = c.get("document_id", "")
            key = (filename, page, doc_id)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": filename,
                    "page": page,
                    "document_id": doc_id,
                })
        return {"answer": answer_text, "sources": sources}

    # Backward compatibility method alias for unit tests
    def answer_question(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self.answer_query(query=query, context_chunks=context_chunks)

    async def stream_answer(
        self, query: str, context_chunks: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        if not context_chunks:
            fallback = json.dumps({
                "type": "token",
                "token": "I couldn't find this information in the uploaded document.",
            })
            done_event = json.dumps({"type": "done", "sources": []})
            yield f"data: {fallback}\n\n"
            yield f"data: {done_event}\n\n"
            return

        prompt = self._build_prompt(query, context_chunks)
        seen = set()
        sources = []
        for c in context_chunks:
            filename = c.get("filename", "document.pdf")
            page = c.get("page") or c.get("page_number", 1)
            doc_id = c.get("document_id", "")
            key = (filename, page, doc_id)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": filename,
                    "page": page,
                    "document_id": doc_id,
                })

        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        full_text = ""
        for chunk in response_stream:
            if chunk.text:
                full_text += chunk.text
                payload = json.dumps({"type": "token", "token": chunk.text})
                yield f"data: {payload}\n\n"

        if "I couldn't find this information" in full_text:
            sources = []

        done_event = json.dumps({"type": "done", "sources": sources})
        yield f"data: {done_event}\n\n"