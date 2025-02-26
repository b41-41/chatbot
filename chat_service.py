from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_loaders import DocumentManager
import os
from dotenv import load_dotenv

load_dotenv()

class ChatService:
    def __init__(self):
        self.document_manager = DocumentManager()
        self.llm = LlamaCpp(
            model_path=os.getenv("LLAMA_MODEL_PATH"),
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
        response = self.chain({"question": query})
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