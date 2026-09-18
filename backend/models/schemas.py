from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the uploaded document")


class SourceItem(BaseModel):
    filename: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
