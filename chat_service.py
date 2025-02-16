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
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self):
        if not self.document_manager.vector_store:
            self.document_manager.load_and_update()
            
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.document_manager.vector_store.as_retriever(),
            memory=self.memory
        )

    def chat(self, query: str) -> str:
        response = self.chain({"question": query})
        return response["answer"]

    def update_documents(self):
        count = self.document_manager.load_and_update()
        self._initialize_chain()
        return count 