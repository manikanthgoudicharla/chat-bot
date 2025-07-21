# routers/chatbot.py
from fastapi import APIRouter
from app.schemas.schema import UserPrompt
from app.controllers.chat import generate_answer

router = APIRouter(prefix="/chatbot")

@router.post('/ask')
async def ask_chatbot(data: UserPrompt):
    response = generate_answer(data.text)
    return {"response": response}
