from fastapi import FastAPI, Query
from pydantic import BaseModel
from chat_service import ChatService
import logging
from model_downloader import ModelDownloader
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 앱 초기화 시 모델 다운로드 확인
logger.info("서버 시작 중... 모델 파일 확인 및 다운로드를 시작합니다.")
model_downloader = ModelDownloader()
model_downloader.check_and_download_models()

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
async def update_documents(force: bool = Query(False, description="강제 업데이트 여부. True인 경우 기존 인덱스가 있어도 강제로 업데이트합니다.")):
    try:
        logger.info(f"문서 로딩 처리 시작... (강제 업데이트: {force})")
        result = chat_service.update_documents(force=force)
        
        if result.get("status") == "already_loaded":
            logger.info("이미 문서가 로드되어 있습니다.")
            return {"message": "이미 문서가 로드되어 있습니다. 새로 로드할 필요가 없습니다."}
        elif result.get("status") == "updated":
            count = result.get("count", 0)
            logger.info(f"문서 로딩 완료: {count}개의 문서가 로드되었습니다.")
            return {"message": f"문서 {count}개가 업데이트되었습니다."}
        
        return {"message": "문서 로드 처리가 완료되었습니다.", "result": result}
    except Exception as e:
        logger.error(f"문서 로딩 중 에러 발생: {e}")
        return {"error": str(e)}, 500

# 직접 실행할 경우 (python main.py로 실행)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 