# AI PDF Assistant (RAG Document Q&A)

A lightweight, full-stack Retrieval-Augmented Generation (RAG) system built with **FastAPI** on the backend and a pure **HTML5/CSS3/JavaScript** frontend. This application allows users to upload PDF documents, extract and chunk text content, vectorize the chunks, and perform grounded Q&A strictly against the document contents.

---

## Features

- **FastAPI Backend**: High-performance REST endpoints for PDF ingestion and chat completion.
- **Vanilla Frontend**: Clean, responsive, zero-build client UI served directly via FastAPI's static file handler.
- **File Ingestion**: Validates and accepts PDF uploads up to 20MB, persisting files in a local data storage directory.
- **Document Metadata**: Inspects and presents document metadata (page counts, chunk counts, filename).
- **RAG Architecture**: Modular design prepared for vector stores (ChromaDB / FAISS) and embedding models.

---

## Project Structure

```text
ai-pdf-assistant/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py          # Handles /ask Q&A endpoint
│   │   └── upload.py        # Handles /upload document parsing
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models & request schemas
│   ├── services/            # Chunking, vector database, and LLM logic
│   │   ├── __init__.py
│   │   └── config.py
│   └── main.py              # Application entrypoint & static mount
├── data/
│   ├── uploads/             # Saved PDF documents
│   └── vector_store/        # Local vector index storage
├── frontend/
│   ├── index.html           # Main user interface
│   ├── style.css            # Dark theme styles & responsive layout
│   └── script.js            # DOM handling and API fetch workflows
├── tests/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Prerequisites

- **Python**: 3.10 to 3.12
- **Conda** or **venv** virtual environment
- Modern web browser (Chrome, Edge, Firefox, or Safari)

---

## Installation & Setup

### 1. Clone the Repository & Navigate to Directory
```powershell
git clone <repository-url>
cd ai-pdf-assistant
```

### 2. Set Up Virtual Environment
Using Anaconda / Conda:
```powershell
conda create -n myenv python=3.11 -y
conda activate myenv
```

Or using standard Python `venv`:
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

*(Ensure essential packages are present: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pypdf`)*

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```ini
OPENAI_API_KEY=your_openai_api_key_here
# Or other supported provider credentials
```

---

## Running the Application

Start the FastAPI application with Uvicorn:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Once running, access the web client in your browser:
```text
http://127.0.0.1:8000
```

Interactive API documentation (Swagger UI) is available at:
```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### 1. Upload Document
- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (binary PDF file)
- **Response**:
  ```json
  {
    "document_id": "handbook.pdf",
    "filename": "handbook.pdf",
    "page_count": 12,
    "chunk_count": 45
  }
  ```

### 2. Ask Question
- **URL**: `/ask`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Payload**:
  ```json
  {
    "question": "What is the primary topic of section 3?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "According to section 3, the primary topic covers..."
  }
  ```

---

## RAG Workflow Overview

1. **Ingestion**: Uploaded PDFs are parsed page-by-page.
2. **Chunking**: Text content is split into overlapping chunks (e.g., 500-character windows with 50-character overlap) to preserve contextual boundaries.
3. **Embedding**: Text chunks are transformed into dense vector representations.
4. **Vector Search**: Query vectors retrieve the top $K$ most semantically relevant chunks via cosine similarity.
5. **Synthesis**: The LLM synthesizes an accurate answer grounded strictly in the retrieved text.

---

## License

This project is licensed under the MIT License.