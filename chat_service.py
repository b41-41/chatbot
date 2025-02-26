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

    def update_documents(self):
        # 이미 벡터 스토어가 로드되어 있는지 확인
        if self.document_manager.vector_store is not None:
            # 강제 업데이트가 필요한 경우에만 다시 로드하도록 수정 가능
            return {"message": "이미 문서가 로드되어 있습니다.", "count": 0, "status": "already_loaded"}
            
        count = self.document_manager.load_and_update()
        self._initialize_chain()
        return {"message": f"문서 {count}개가 업데이트되었습니다.", "count": count, "status": "updated"} 