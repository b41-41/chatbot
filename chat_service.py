from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_loaders import DocumentManager
import os
import subprocess
import json
import time
import requests
from dotenv import load_dotenv
from langdetect import detect
from model_downloader import ModelDownloader
from typing import List, Dict, Any

load_dotenv()

class ChatService:
    def __init__(self):
        self.document_manager = DocumentManager()
        
        # MCP 설정 로드
        self.load_mcp_config()
        
        # MCP 서버 시작
        self.start_mcp_servers()
        
        # 모델 다운로더 인스턴스 생성
        self.model_downloader = ModelDownloader()
        
        # 환경 변수에서 모델 이름/경로 가져오기
        model_path_or_repo = os.getenv("LLAMA_MODEL")
        
        # Hugging Face 레포지토리 경로인지 확인
        if model_path_or_repo and '/' in model_path_or_repo and '.' not in model_path_or_repo.split('/')[-1]:
            # 모델 다운로드 및 실제 파일 경로 얻기
            actual_model_path = self.model_downloader._download_from_hf_repo(model_path_or_repo)
            if actual_model_path:
                # 다운로드 성공 시 파일 경로 설정
                model_path = os.path.join("models", actual_model_path)
            else:
                # 다운로드 실패 시 원래 경로 사용
                model_path = os.path.join("models", model_path_or_repo)
        else:
            # 일반 파일 경로인 경우
            model_path = os.path.join("models", model_path_or_repo) if model_path_or_repo else None
        
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            n_ctx=4096,
            n_batch=512,
            verbose=True,
            f16_kv=True,
            streaming=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = None
        
        # MCP 서버 프로세스 및 엔드포인트 정보
        self.mcp_processes = {}
        self.mcp_endpoints = {}
        
        # 기본 MCP 포트
        self.default_port = 8000
        
        # 문서 관리자 초기화 후 체인도 자동으로 초기화
        self._initialize_chain()
        
    def load_mcp_config(self):
        """mcp.json 파일에서 MCP 설정을 로드합니다."""
        try:
            with open("mcp.json", "r", encoding="utf-8") as f:
                self.mcp_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"MCP 설정 로드 실패: {e}")
            # 기본 MCP 설정
            self.mcp_config = {
                "mcpServers": {},
                "prompt": {
                    "system_message": "문서와 대화 기록을 바탕으로 질문에 답변해주세요.",
                    "format": {
                        "korean": "한국어로 답변해주세요.",
                        "english": "Please answer in English."
                    }
                }
            }

    def start_mcp_servers(self):
        """MCP 서버들을 시작합니다."""
        if "mcpServers" not in self.mcp_config:
            print("MCP 서버 설정이 없습니다.")
            return
        
        for i, (server_name, server_config) in enumerate(self.mcp_config["mcpServers"].items()):
            try:
                # 서버 포트 설정 (각 서버마다 다른 포트 사용)
                port = self.default_port + i
                
                # 환경 변수 설정
                env = os.environ.copy()
                if "env" in server_config:
                    env.update(server_config["env"])
                
                # PORT 환경 변수 추가
                env["PORT"] = str(port)
                
                # 서버 시작 명령어
                cmd = [server_config["command"]] + server_config.get("args", [])
                
                # 서브프로세스로 서버 실행
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 프로세스 및 엔드포인트 정보 저장
                self.mcp_processes[server_name] = process
                self.mcp_endpoints[server_name] = f"http://localhost:{port}"
                
                print(f"MCP 서버 '{server_name}' 시작됨 (PID: {process.pid}, 엔드포인트: {self.mcp_endpoints[server_name]})")
                
                # 서버가 시작될 시간 제공
                time.sleep(2)
                
            except Exception as e:
                print(f"MCP 서버 '{server_name}' 시작 실패: {e}")

    def _initialize_chain(self):
        if self.document_manager.vector_store:
            # 벡터 스토어 검색기 설정
            self.retriever = self.document_manager.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # 원래 ConversationalRetrievalChain도 유지
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                max_tokens_limit=3000
            )

    def query_mcp_server(self, server_name: str, endpoint: str, query: str) -> Dict[str, Any]:
        """MCP 서버에 쿼리를 보내고 결과를 가져옵니다."""
        try:
            # 서버가 실행 중인지 확인
            process = self.mcp_processes.get(server_name)
            if not process or process.poll() is not None:
                print(f"MCP 서버 '{server_name}'가 실행 중이 아닙니다.")
                return {}
            
            # MCP 서버 엔드포인트로 요청 보내기
            response = requests.post(
                f"{endpoint}/query",
                json={"query": query}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"MCP 서버 '{server_name}' 쿼리 실패: 상태 코드 {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"MCP 서버 '{server_name}' 쿼리 중 오류 발생: {e}")
            return {}

    def chat(self, query: str) -> str:
        if not self.retriever:
            # 로드 시도
            self._initialize_chain()
            # 여전히 검색기가 없다면 오류 메시지 반환
            if not self.retriever:
                return "문서가 로드되지 않았습니다. /update API를 호출하여 문서를 로드해주세요."
        
        # 입력 언어 감지
        try:
            language = detect(query)
        except:
            language = "en"  # 언어 감지 실패 시 기본값은 영어
        
        # 문서 검색
        docs = self.retriever.get_relevant_documents(query)
        
        # 문서 데이터 준비
        document_context = ""
        for i, doc in enumerate(docs):
            document_context += f"문서 {i+1}:\n{doc.page_content}\n\n"
        
        # 모든 컨텍스트 소스를 담을 딕셔너리
        context_sources = {"documents": document_context}
        
        # MCP 서버들에서 데이터 수집
        for server_name, endpoint in self.mcp_endpoints.items():
            # 서버에 쿼리 보내기
            server_data = self.query_mcp_server(server_name, endpoint, query)
            
            # 결과가 있으면 컨텍스트에 추가
            if server_data and "data" in server_data:
                context_sources[server_name] = server_data["data"]
        
        # 이전 대화 기록 추가
        chat_history = ""
        messages = self.memory.chat_memory.messages
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content
                ai_msg = messages[i+1].content
                chat_history += f"사용자: {user_msg}\n시스템: {ai_msg}\n\n"
        
        # 언어에 따른 응답 형식 설정
        response_format = self.mcp_config.get("prompt", {}).get("format", {}).get(
            "korean" if language == "ko" else "english", ""
        )
        
        # 시스템 메시지
        system_message = self.mcp_config.get("prompt", {}).get("system_message", "")
        
        # MCP 컨텍스트와 함께 프롬프트 구성
        prompt = f"{system_message}\n\n"
        
        # 모든 컨텍스트 소스 추가
        prompt += "## 컨텍스트\n"
        for source_name, source_data in context_sources.items():
            if source_data and source_data.strip():  # 빈 데이터가 아닌 경우에만 추가
                prompt += f"### {source_name.upper()} 정보\n{source_data}\n\n"
        
        # 이전 대화 추가
        if chat_history:
            prompt += f"## 이전 대화\n{chat_history}\n"
        
        # 질문과 언어 형식 추가
        prompt += f"## 질문\n{query}\n\n{response_format}"
        
        # LLM으로 응답 생성
        response = self.llm(prompt)
        
        # 메모리에 대화 저장
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response)
        
        return response

    def update_documents(self, force=False):
        """
        문서를 업데이트합니다.
        
        Args:
            force (bool): 강제 업데이트 여부. True이면 이미 인덱스가 있어도 강제로 업데이트합니다.
            
        Returns:
            dict: 업데이트 결과 정보를 담은 사전
        """
        # 문서 매니저를 통해 인덱스 업데이트
        result = self.document_manager.load_and_update(force_update=force)
        
        # 업데이트 완료 후 체인 초기화
        if result.get("status") == "updated":
            self._initialize_chain()
            
        return result
    
    def __del__(self):
        """소멸자: MCP 서버 프로세스 종료"""
        for server_name, process in self.mcp_processes.items():
            if process and process.poll() is None:  # 프로세스가 실행 중인지 확인
                process.terminate()
                print(f"MCP 서버 '{server_name}' 종료됨") 