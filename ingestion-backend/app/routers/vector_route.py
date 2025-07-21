from fastapi import APIRouter
from typing import List
from app.schemas.vector_schema import DocumentIn
from app.crud.vector_crud import upsert_document

router = APIRouter(prefix="/vectors", tags=["vectors"])

@router.post("/upsertdocument")
async def insert_document(doc: DocumentIn):
    return await upsert_document(doc)

@router.post("/upsertdocuments")
async def insert_documents(docs: List[DocumentIn]):
    results = []
    for doc in docs:
        result = await upsert_document(doc)
        results.append(result)
    return {
        "status": "success",
        "uploaded": len(results),
        "details": results
    }
