from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import os
import logging

logger = logging.getLogger(__name__)

# .env 파일 로드 확인 (중복 load_dotenv() 제거)
env_path = find_dotenv(raise_error_if_not_found=True)
print(f"❤️Found .env file: {env_path}")
print("현재 작업 디렉토리:", os.getcwd())

# 환경변수 로드 (한 번만 호출)
load_dotenv(env_path, override=True, verbose=True)

class DocumentManager:
    def __init__(self):
        # 환경변수 값 확인을 위한 로깅
        for env_var in ["CONFLUENCE_URL", "CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN", "CONFLUENCE_SPACE_KEY", "LLAMA_MODEL_PATH"]:
            print(f"✅ {env_var}: {os.getenv(env_var)}")
            
        self.confluence_loader = ConfluenceLoader(
            url=os.getenv("CONFLUENCE_URL"),
            username=os.getenv("CONFLUENCE_USERNAME"),
            api_key=os.getenv("CONFLUENCE_API_TOKEN")
        )
        
        print(f"✅LLAMA_MODEL_PATH: {os.getenv('LLAMA_MODEL_PATH')}")

        self.embeddings = LlamaCppEmbeddings(
            model_path=os.getenv("LLAMA_MODEL_PATH")
        )
        
        self.vector_store = None

    def load_and_update(self):
        try:
            # Confluence 문서 로드
            logger.info("Confluence 문서 로딩 시작...")
            all_docs = self.confluence_loader.load(
                space_key=os.getenv("CONFLUENCE_SPACE_KEY")
            )
            logger.info(f"Confluence 문서 {len(all_docs)}개 로드 완료")
            
            # Chroma 벡터 스토어 생성
            logger.info("벡터 스토어 생성 시작...")
            self.vector_store = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                persist_directory="chroma_db"
            )
            logger.info("벡터 스토어 생성 완료")
            
            return len(all_docs)
            
        except Exception as e:
            logger.error(f"문서 로드 중 에러 발생: {e}")
            raise e 