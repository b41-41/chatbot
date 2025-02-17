# Document Chat Bot

Confluence와 Notion의 문서들을 로드하여 채팅할 수 있는 API 서버입니다.

## 기능

- Confluence와 Notion 문서 데이터 로드 및 벡터 저장소 구축
- 문서 기반 질의응답 채팅
- 문서 데이터 실시간 업데이트

## 설치 방법

1. Python 3.11 이상 설치 (3.13은 일부 패키지와 호환성 문제가 있을 수 있음)

2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv myenv

# 가상환경 활성화
# Windows의 경우
myenv\Scripts\activate
# Mac/Linux의 경우
source myenv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 환경 변수 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 내용을 입력하세요:

```env
# Confluence 설정
CONFLUENCE_URL=your_confluence_url
CONFLUENCE_USERNAME=your_username
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_SPACE_KEY=your_space_key

# Notion 설정
NOTION_API_KEY=your_notion_api_key

# LLM 설정
LLAMA_MODEL_PATH=path_to_your_llama_model
```

## 실행 방법

서버 실행:
```bash
python -m uvicorn main:app --reload
```

서버가 실행되면 다음 주소에서 API를 사용할 수 있습니다:
- API 서버: http://localhost:8000
- API 문서: http://localhost:8000/docs

## API 엔드포인트

### 1. 채팅 API
- URL: `/chat`
- Method: POST
- Request Body:
```json
{
    "message": "질문 내용"
}
```

### 2. 문서 업데이트 API
- URL: `/update`
- Method: POST
- Response:
```json
{
    "message": "문서 n개가 업데이트되었습니다."
}
```

## 프로젝트 구조

```
.
├── main.py              # FastAPI 서버 및 엔드포인트
├── chat_service.py      # 채팅 서비스 로직
├── document_loaders.py  # 문서 로더 및 벡터 저장소 관리
├── requirements.txt     # 필요한 패키지 목록
├── .env                 # 환경 변수 파일
└── chroma_db/          # 벡터 데이터베이스 저장소
```

## 주의사항

1. LLama 모델 파일이 필요합니다. 적절한 모델을 다운로드하여 경로를 환경변수에 설정하세요.
2. Confluence와 Notion API 토큰이 필요합니다.
3. 첫 실행 시 문서 로딩에 시간이 걸릴 수 있습니다.

## 모델 파일 다운로드

LLama 모델 파일은 다음 주소에서 다운로드할 수 있습니다:
- llama-2-7b-chat.Q4_K_M.gguf: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

다운로드한 모델 파일을 `models` 디렉토리에 저장하고 `.env` 파일의 `LLAMA_MODEL_PATH`를 적절히 설정하세요.

