import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    UPLOAD_DIR: str = "data/uploads"
    VECTOR_STORE_DIR: str = "data/vector_store"

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 5

    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure data storage directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
