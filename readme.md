# Document Chat Bot

MCP(Message Context Processing) 서버를 통해 Notion과 Atlassian(Jira) 데이터를 자동으로 가져와 채팅할 수 있는 API 서버입니다.

## 기능

- MCP 서버를 통한 Notion과 Atlassian(Jira) 데이터 실시간 조회
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

## 환경 변수 및 MCP 설정

`.env` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 내용을 입력하세요:

```env
# LLM 설정
# 옵션 1: 파일명만 입력 (자동으로 models/ 디렉토리에서 찾음)
LLAMA_MODEL=llama-2-7b-chat.Q4_K_M.gguf

# 옵션 2: Hugging Face 레포지토리 경로 입력 (자동으로 다운로드)
# LLAMA_MODEL=unsloth/DeepSeek-R1-GGUF

# 임베딩 모델도 마찬가지로 파일명 또는 Hugging Face 레포지토리 지정 가능
EMBEDDING_MODEL=all-MiniLM-L6-v2
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

`mcp.json` 파일을 프로젝트 루트 디렉토리에 생성하고 다음 내용을 입력하세요:

```json
{
  "mcpServers": {
    "notion": {
      "command": "npx",
      "args": ["-y", "@suekou/mcp-notion-server"],
      "env": {
        "NOTION_API_TOKEN": "your-integration-token"
      }
    },
    "atlassian": {
      "command": "npx",
      "args": ["-y", "@sooperset/mcp-atlassian"],
      "env": {
        "JIRA_API_TOKEN": "your-api-token",
        "JIRA_USERNAME": "your-email",
        "JIRA_HOST": "your-instance.atlassian.net"
      }
    }
  },
  "prompt": {
    "system_message": "당신은 유용한 AI 어시스턴트입니다. 주어진 문서와 Notion, Jira의 데이터를 바탕으로 사용자의 질문에 정확하게 답변해주세요.",
    "format": {
      "korean": "한국어로 답변해주세요.",
      "english": "Please answer in English."
    }
  }
}
```

모델은 두 가지 방식으로 지정할 수 있습니다:
1. **파일명만 입력**: 모델 파일은 자동으로 `models` 디렉토리에서 찾거나 다운로드합니다.
2. **Hugging Face 레포지토리 경로**: 예를 들어 `unsloth/DeepSeek-R1-GGUF`처럼 입력하면 해당 레포지토리에서 적절한 GGUF 파일을 자동으로 선택하여 다운로드합니다.
   * 분할된 GGUF 파일(예: `.gguf.part-1`, `model-00001-of-00003.gguf` 등)이 있는 경우 자동으로 감지하여 병합합니다.
   * 대용량 모델의 경우 다운로드 및 병합에 시간이 걸릴 수 있습니다.

## MCP 서버 설정

### Notion MCP 서버
1. Notion 통합 생성:
   - [Notion Your Integrations 페이지](https://www.notion.so/my-integrations)에 접속
   - "New Integration" 클릭
   - 통합에 이름을 지정하고 적절한 권한 선택(예: "Read content", "Update content")
2. Secret Key 가져오기:
   - 생성된 통합의 "Internal Integration Token" 복사
3. 워크스페이스에 통합 추가:
   - Notion에서 통합에 액세스할 페이지나 데이터베이스 열기
   - 우측 상단의 "···" 버튼 클릭
   - "Connections" 버튼 클릭, 1단계에서 생성한 통합 선택

### Atlassian MCP 서버
1. Jira API 토큰 생성:
   - [Atlassian API 토큰 관리 페이지](https://id.atlassian.com/manage-profile/security/api-tokens)에 접속
   - "Create API token" 클릭
   - 토큰에 이름을 지정하고 생성
2. mcp.json 파일에 Jira 정보 입력:
   - "JIRA_API_TOKEN": 생성한 API 토큰
   - "JIRA_USERNAME": Atlassian 계정 이메일
   - "JIRA_HOST": Jira 인스턴스 URL (예: "your-instance.atlassian.net")

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
    "message": "문서 업데이트가 완료되었습니다."
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
├── mcp.json             # MCP 서버 설정 파일
├── models/              # 모델 파일 저장 디렉토리
└── faiss_index/         # 벡터 데이터베이스 저장소
```

## 주의사항

1. LLama 모델 파일이 필요합니다. 적절한 모델 이름 또는 Hugging Face 레포지토리를 환경변수에 설정하면 자동으로 다운로드됩니다.
2. Notion API 토큰과 Jira API 토큰이 필요합니다.
3. 첫 실행 시 MCP 서버 시작 및 모델 로딩에 시간이 걸릴 수 있습니다.
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

