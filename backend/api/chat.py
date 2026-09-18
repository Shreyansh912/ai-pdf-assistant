from fastapi import APIRouter, HTTPException, Request
from backend.models.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, payload: AskRequest):
    vector_store = request.app.state.vector_store

    # Guard: Require that a document has been uploaded
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No document has been uploaded or processed yet. Please upload a PDF first."
        )

    rag_service = request.app.state.rag_service
    top_k = request.app.state.settings.TOP_K

    try:
        result = rag_service.answer_question(payload.question, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"]
    )
