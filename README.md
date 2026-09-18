# AI PDF Assistant: Production RAG-Based Document Q&A System

An end-to-end, high-performance Retrieval-Augmented Generation (RAG) system built with Python 3.11, FastAPI, FAISS, Sentence Transformers, and Google Gemini API (`gemini-3.6-flash`).

The system ingests digital PDF documents, extracts clean text page-by-page, produces overlapping semantic chunks, indexes them into a high-dimensional vector space using FAISS, and answers user queries strictly grounded in retrieved evidence with exact page citations and anti-hallucination guardrails.

---


## Key Features & Design Decisions

* **Page-by-Page Ingestion:** Rather than flattening the PDF into an unindexed text block, PyMuPDF (fitz) tracks human-readable, 1-indexed page numbers on every chunk for precise citations.
* **Sliding Window Chunking:** Configured with an 800-character window and 150-character overlap, utilizing backward boundary snapping to respect natural sentence boundaries.
* **Dense Vector Indexing with L2-Normalization:** Passages are embedded into 384-dimensional dense vectors using all-MiniLM-L6-v2. Each vector is unit-normalized, enabling exact inner-product search (IndexFlatIP) to compute cosine similarity.
* **Anti-Hallucination Guardrail:** Configured with temperature=0.0 and rigid prompt constraints. If context is missing, the engine returns:
  > *"I couldn't find this information in the uploaded document."*
* **Complete Test Coverage:** Includes 20 unit and integration tests across extraction, chunking, embeddings, vector indexing, RAG generation, and FastAPI endpoints.

---

## Tech Stack

* **Backend:** FastAPI, Uvicorn, Pydantic V2, Pydantic-Settings
* **PDF Extraction:** PyMuPDF (fitz)
* **Vector Store:** FAISS (CPU, IndexFlatIP)
* **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
* **LLM Engine:** Google GenAI SDK (gemini-3.6-flash)
* **Frontend:** Vanilla HTML5, CSS Variables, Native JavaScript
* **Testing:** Pytest

---

## Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/Shreyansh912/ai-pdf-assistant.git
cd ai-pdf-assistant

conda create --name myenv python=3.11 -y
conda activate myenv

pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
Visit http://127.0.0.1:8000 in your browser.

---

## Running Automated Tests

```bash
python -m pytest tests/ -v
```

