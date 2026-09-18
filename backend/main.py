import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.models.schemas import HealthResponse
from backend.services.embeddings import EmbeddingService
from backend.services.chunker import TextChunker
from backend.services.vector_store import VectorStore
from backend.services.rag import RAGService
from backend.api.upload import router as upload_router
from backend.api.chat import router as chat_router

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize shared services once
    app.state.settings = settings
    app.state.embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
    app.state.chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    app.state.vector_store = VectorStore(dimension=app.state.embedding_service.dimension)

    # Load existing FAISS index if present
    if os.path.exists(os.path.join(settings.VECTOR_STORE_DIR, "index.faiss")):
        try:
            app.state.vector_store.load(settings.VECTOR_STORE_DIR)
            app.state.active_pdf = "Existing Index"
        except Exception:
            app.state.active_pdf = None
    else:
        app.state.active_pdf = None

    app.state.rag_service = RAGService(
        embedding_service=app.state.embedding_service,
        vector_store=app.state.vector_store,
        api_key=settings.GEMINI_API_KEY,
        model_name=settings.LLM_MODEL
    )
    yield


app = FastAPI(
    title="AI PDF Assistant",
    description="RAG-Based Document Question Answering System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


# Serve index.html and static assets
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(frontend_path, "style.css"), media_type="text/css")


@app.get("/script.js")
async def serve_js():
    return FileResponse(os.path.join(frontend_path, "script.js"), media_type="application/javascript")
