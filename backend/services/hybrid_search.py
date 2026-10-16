import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, vector_store, embedding_service, rrf_k: int = 60):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.bm25: Optional[BM25Okapi] = None
        self._build_bm25()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_bm25(self):
        if not self.vector_store.metadata:
            self.bm25 = None
            return
        corpus = [self._tokenize(chunk["text"]) for chunk in self.vector_store.metadata]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.vector_store.metadata:
            return []

        q_vec = self.embedding_service.encode([query])
        dense_hits = self.vector_store.search(q_vec, top_k=top_k * 2, document_id=document_id)

        if self.bm25 is None or len(self.bm25.doc_len) != len(self.vector_store.metadata):
            self._build_bm25()

        sparse_hits = []
        if self.bm25:
            tokenized_q = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_q)
            bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

            for idx in bm25_indices[:top_k * 2]:
                item = self.vector_store.metadata[idx].copy()
                if document_id and item.get("document_id") != document_id:
                    continue
                sparse_hits.append(item)
                if len(sparse_hits) >= top_k * 2:
                    break

        fused_scores: Dict[str, float] = {}
        item_map: Dict[str, Dict[str, Any]] = {}

        def get_key(doc: Dict[str, Any]) -> str:
            return f"{doc['document_id']}_{doc['page']}_{hash(doc['text'][:80])}"

        for rank, doc in enumerate(dense_hits):
            key = get_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            item_map[key] = doc

        for rank, doc in enumerate(sparse_hits):
            key = get_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            item_map[key] = doc

        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        return [item_map[k] for k in sorted_keys[:top_k]]
