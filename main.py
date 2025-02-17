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
        logger.info("문서 로딩 시작...")
        doc_count = chat_service.update_documents()
        logger.info(f"문서 로딩 완료: {doc_count}개의 문서가 로드되었습니다.")
        return {"message": f"문서 {doc_count}개가 업데이트되었습니다."}
    except Exception as e:
        logger.error(f"문서 로딩 중 에러 발생: {e}")
        return {"error": str(e)}, 500 