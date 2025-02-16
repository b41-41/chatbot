from fastapi import FastAPI
from pydantic import BaseModel
from chat_service import ChatService

app = FastAPI()
chat_service = ChatService()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = chat_service.chat(request.message)
    return ChatResponse(response=response)

@app.post("/update")
async def update_documents():
    doc_count = chat_service.update_documents()
    return {"message": f"문서 {doc_count}개가 업데이트되었습니다."} 