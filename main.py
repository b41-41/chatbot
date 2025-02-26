from fastapi import FastAPI
from pydantic import BaseModel
from chat_service import ChatService
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    try:
        logger.info("문서 로딩 처리 시작...")
        result = chat_service.update_documents()
        
        if result.get("status") == "already_loaded":
            logger.info("이미 문서가 로드되어 있습니다.")
            return {"message": "이미 문서가 로드되어 있습니다. 새로 로드할 필요가 없습니다."}
        
        logger.info(f"문서 로딩 완료: {result.get('count')}개의 문서가 로드되었습니다.")
        return {"message": result.get("message")}
    except Exception as e:
        logger.error(f"문서 로딩 중 에러 발생: {e}")
        return {"error": str(e)}, 500 