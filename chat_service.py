from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_loaders import DocumentManager
import os
from dotenv import load_dotenv
from langdetect import detect
from model_downloader import ModelDownloader

load_dotenv()

class ChatService:
    def __init__(self):
        self.document_manager = DocumentManager()
        
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
        
        # 문서 관리자 초기화 후 체인도 자동으로 초기화
        self._initialize_chain()

    def _initialize_chain(self):
        if self.document_manager.vector_store:
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.document_manager.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                max_tokens_limit=3000
            )

    def chat(self, query: str) -> str:
        if not self.chain:
            # 로드 시도
            self._initialize_chain()
            # 여전히 체인이 없다면 오류 메시지 반환
            if not self.chain:
                return "문서가 로드되지 않았습니다. /update API를 호출하여 문서를 로드해주세요."
        
        # 입력 언어 감지
        try:
            language = detect(query)
        except:
            language = "en"  # 언어 감지 실패 시 기본값은 영어
        
        # 한국어 입력인 경우, 한국어 응답을 요청하는 프롬프트 추가
        if language == "ko":
            prompt = f"{query}\n\nplease answer in korean"
        else:
            prompt = query
            
        response = self.chain({"question": prompt})
        return response["answer"]

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