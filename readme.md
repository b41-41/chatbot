# Document Chat Bot

Confluence와 Notion의 문서들을 로드하여 채팅할 수 있는 API 서버입니다.

## 기능

- Confluence와 Notion 문서 데이터 로드 및 벡터 저장소 구축
- 문서 기반 질의응답 채팅
- 문서 데이터 실시간 업데이트
- Hugging Face 레포지토리에서 자동 모델 다운로드
- 분할된 GGUF 파일 자동 감지 및 병합 지원

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
# 옵션 1: 파일명만 입력 (자동으로 models/ 디렉토리에서 찾음)
LLAMA_MODEL=llama-2-7b-chat.Q4_K_M.gguf

# 옵션 2: Hugging Face 레포지토리 경로 입력 (자동으로 다운로드)
# LLAMA_MODEL=unsloth/DeepSeek-R1-GGUF

# 임베딩 모델도 마찬가지로 파일명 또는 Hugging Face 레포지토리 지정 가능
EMBEDDING_MODEL=all-MiniLM-L6-v2
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

모델은 두 가지 방식으로 지정할 수 있습니다:
1. **파일명만 입력**: 모델 파일은 자동으로 `models` 디렉토리에서 찾거나 다운로드합니다.
2. **Hugging Face 레포지토리 경로**: 예를 들어 `unsloth/DeepSeek-R1-GGUF`처럼 입력하면 해당 레포지토리에서 적절한 GGUF 파일을 자동으로 선택하여 다운로드합니다.
   * 분할된 GGUF 파일(예: `.gguf.part-1`, `model-00001-of-00003.gguf` 등)이 있는 경우 자동으로 감지하여 병합합니다.
   * 대용량 모델의 경우 다운로드 및 병합에 시간이 걸릴 수 있습니다.

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
├── model_downloader.py  # 모델 다운로드 및 관리
├── requirements.txt     # 필요한 패키지 목록
├── .env                 # 환경 변수 파일
├── models/             # 모델 파일 저장 디렉토리
└── faiss_index/         # 벡터 데이터베이스 저장소
```

## 주의사항

1. LLama 모델 파일이 필요합니다. 적절한 모델 이름 또는 Hugging Face 레포지토리를 환경변수에 설정하면 자동으로 다운로드됩니다.
2. Confluence와 Notion API 토큰이 필요합니다.
3. 첫 실행 시 문서 로딩에 시간이 걸릴 수 있습니다.
4. 분할된 GGUF 파일을 사용하는 모델은 처음 실행 시 다운로드 및 병합 과정이 필요하므로 시간이 더 걸릴 수 있습니다.

## 모델 파일 다운로드

모델 파일은 자동으로 다운로드되지만, 수동으로 다운로드하려면 다음 주소를 이용하세요:
- llama-2-7b-chat.Q4_K_M.gguf: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

다운로드한 모델 파일을 `models` 디렉토리에 저장하고 `.env` 파일의 `LLAMA_MODEL`에 모델 이름(파일명)만 입력하세요.

## 새로운 모델 사용 예시

DeepSeek R1 모델을 사용하려면:

```env
LLAMA_MODEL=unsloth/DeepSeek-R1-GGUF
```

이렇게 설정하면 서버 시작 시 자동으로 레포지토리에서 적절한 GGUF 파일을 선택하여 다운로드하고, 분할된 파일인 경우 자동으로 병합합니다.

