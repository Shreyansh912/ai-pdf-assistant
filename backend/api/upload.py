import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from backend.config import settings
from backend.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentMetadata
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    app_state = request.app.state
    upload_dir = getattr(getattr(app_state, "config", None), "UPLOAD_DIR", settings.UPLOAD_DIR)
    doc_id = str(uuid.uuid4())[:8]

    save_dir = Path(upload_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{doc_id}_{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processor = getattr(app_state, "pdf_processor", None)
    if not processor:
        from backend.services.pdf_processor import PDFProcessor
        processor = PDFProcessor()

    # Call PDFProcessor's actual method: extract_text_by_page
    if hasattr(processor, "extract_text_by_page"):
        raw_pages = processor.extract_text_by_page(str(save_path))
    elif hasattr(processor, "extract_text"):
        raw_pages = processor.extract_text(str(save_path))
    elif hasattr(processor, "extract_pages"):
        raw_pages = processor.extract_pages(str(save_path))
    else:
        raise HTTPException(status_code=500, detail="No PDF text extraction method available.")

    if not raw_pages:
        raise HTTPException(status_code=400, detail="PDF has no extractable text.")

    # Normalize extracted pages so each dict has filename, page_number, and text
    normalized_pages = []
    for idx, item in enumerate(raw_pages):
        if isinstance(item, str):
            if item.strip():
                normalized_pages.append({
                    "filename": file.filename,
                    "page_number": idx + 1,
                    "page": idx + 1,
                    "text": item,
                })
        elif isinstance(item, dict):
            page_copy = item.copy()
            page_copy.setdefault("filename", file.filename)
            page_copy.setdefault("page_number", idx + 1)
            page_copy.setdefault("page", page_copy["page_number"])
            if page_copy.get("text", "").strip():
                normalized_pages.append(page_copy)

    if not normalized_pages:
        raise HTTPException(status_code=400, detail="PDF has no extractable text.")

    chunker = getattr(app_state, "chunker", None)
    if not chunker:
        from backend.services.chunker import TextChunker
        chunker = TextChunker()

    chunks = chunker.chunk_pages(normalized_pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="PDF text yielded no chunks.")

    texts = [c["text"] for c in chunks]
    embeddings_service = getattr(app_state, "embeddings", None)
    if not embeddings_service:
        from backend.services.embeddings import EmbeddingService
        embeddings_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)

    embeddings = embeddings_service.encode(texts)

    vector_store = getattr(app_state, "vector_store", None)
    if not vector_store:
        from backend.services.vector_store import VectorStoreService
        vector_store = VectorStoreService(store_dir=settings.VECTOR_STORE_DIR)

    total_chunks = vector_store.add_document(
        document_id=doc_id,
        filename=file.filename,
        page_count=len(normalized_pages),
        embeddings=embeddings,
        chunks=chunks,
    )

    if hasattr(app_state, "hybrid_retriever") and app_state.hybrid_retriever:
        app_state.hybrid_retriever._build_bm25()

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        pages=len(normalized_pages),
        chunks=len(chunks),
        total_indexed_chunks=total_chunks,
        message="Document processed and indexed successfully."
    )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(request: Request):
    app_state = request.app.state
    vector_store = getattr(app_state, "vector_store", None)
    if not vector_store:
        return DocumentListResponse(documents=[], total_documents=0, total_chunks=0)

    docs = vector_store.list_documents()
    metadata_list = [DocumentMetadata(**d) for d in docs]
    total_chunks = sum(d["chunk_count"] for d in docs)
    return DocumentListResponse(
        documents=metadata_list,
        total_documents=len(metadata_list),
        total_chunks=total_chunks
    )