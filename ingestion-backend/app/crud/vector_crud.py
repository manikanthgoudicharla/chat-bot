from fastapi import HTTPException
from app.db.database import index
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.schemas.vector_schema import DocumentIn

# Load once
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

async def upsert_document(doc: DocumentIn):
    try:
        # 1. Split large text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(doc.text)

        # 2. Embed chunks
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # 3. Build vector objects
        vectors = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc.id}_chunk_{i}",
                "values": embedding,
                "metadata": {
                    "parent_id": doc.id,
                    "text": chunk_text,
                    **(doc.vector_metadata or {})
                }
            })

        # 4. Upsert to Pinecone
        index.upsert(vectors)

        return {"upserted": len(vectors)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone upsert error: {str(e)}")
