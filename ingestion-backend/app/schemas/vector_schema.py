from pydantic import BaseModel
from typing import Optional, Dict

class DocumentIn(BaseModel):
    id: str
    text: str
    vector_metadata: Optional[Dict[str, str]] = None
