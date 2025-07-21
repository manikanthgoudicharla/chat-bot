from pydantic import BaseModel


class UserPrompt(BaseModel):
    text: str
    
