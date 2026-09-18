import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from backend.models.schemas import UploadResponse
from backend.services.pdf_processor import PDFProcessor, PDFProcessingError

router = APIRouter()

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    # 1. Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # 2. Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of 20 MB ({file_size / (1024*1024):.1f} MB uploaded)."
        )

    # 3. Save PDF to local disk
    upload_dir = request.app.state.settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file: {str(e)}")

    # 4. Extract text page-by-page
    processor = PDFProcessor()
    try:
        pages_data = processor.extract_text_by_page(file_path)
    except PDFProcessingError as e:
        # Clean up unreadable file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    # 5. Chunk text
    chunker = request.app.state.chunker
    chunks = chunker.chunk_pages(pages_data)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not create any text chunks from document.")

    # 6. Generate embeddings
    chunk_texts = [c["text"] for c in chunks]
    embedding_service = request.app.state.embedding_service
    embeddings = embedding_service.generate_embeddings(chunk_texts)

    # 7. Add to FAISS index & persist
    vector_store = request.app.state.vector_store
    vector_store.create_and_add(embeddings, chunks)
    vector_store_dir = request.app.state.settings.VECTOR_STORE_DIR
    vector_store.save(vector_store_dir)

    # Mark vector index as active
    request.app.state.active_pdf = file.filename

    return UploadResponse(
        message="PDF processed successfully",
        filename=file.filename,
        pages=len(pages_data),
        chunks=len(chunks)
    )
