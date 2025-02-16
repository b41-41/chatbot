from langchain_community.document_loaders import ConfluenceLoader, NotionDBLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentManager:
    def __init__(self):
        self.confluence_loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_key=os.getenv("CONFLUENCE_API_TOKEN")
        )
        
        self.notion_loader = NotionDBLoader(
            token=os.getenv("NOTION_API_KEY"),
            database_id=os.getenv("NOTION_DATABASE_ID")
        )
        
        self.embeddings = LlamaCppEmbeddings(
            model_path=os.getenv("LLAMA_MODEL_PATH")
        )
        
        self.vector_store = None

    def load_and_update(self):
        # Confluence 문서 로드
        confluence_docs = self.confluence_loader.load()
        
        # Notion 문서 로드
        notion_docs = self.notion_loader.load()
        
        # 모든 문서 합치기
        all_docs = confluence_docs + notion_docs
        
        # Chroma 벡터 스토어 생성
        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory="chroma_db"
        )
        
        return len(all_docs) 