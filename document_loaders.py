from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        
        # 기존 faiss 인덱스가 있는지 확인하고 있으면 로드
        self.try_load_existing_index()

    def try_load_existing_index(self):
        """로컬에 저장된 FAISS 인덱스가 있으면 로드"""
        try:
            faiss_path = "faiss_index"
            if os.path.exists(faiss_path):
                logger.info("기존 FAISS 인덱스 파일 발견, 로드 중...")
                self.vector_store = FAISS.load_local(faiss_path, self.embeddings)
                logger.info("기존 FAISS 인덱스 로드 완료")
                return True
            else:
                logger.info("기존 FAISS 인덱스 파일이 없습니다.")
                return False
        except Exception as e:
            logger.error(f"FAISS 인덱스 로드 중 오류 발생: {e}")
            return False

    def load_and_update(self):
        try:
            # Confluence 문서 로드
            logger.info("Confluence 문서 로딩 시작...")
            all_docs = self.confluence_loader.load(
                space_key=os.getenv("CONFLUENCE_SPACE_KEY")
            )
            logger.info(f"Confluence 문서 {len(all_docs)}개 로드 완료")
            
            # 문서 내용 확인을 위한 디버깅
            if all_docs:
                logger.info(f"첫 번째 문서 미리보기: {str(all_docs[0])[:200]}...")
            
            # FAISS 벡터 스토어 생성
            logger.info("벡터 스토어 생성 시작...")
            
            try:
                # 임베딩 테스트
                logger.info("임베딩 테스트 시작...")
                test_text = "테스트 문장입니다."
                test_embedding = self.embeddings.embed_query(test_text)
                logger.info(f"임베딩 테스트 성공: 벡터 크기 {len(test_embedding)}")
                
                # 벡터 스토어 생성
                logger.info("FAISS 벡터 스토어 생성 중...")
                self.vector_store = FAISS.from_documents(
                    documents=all_docs,
                    embedding=self.embeddings
                )
                
                # 벡터 스토어 저장
                self.vector_store.save_local("faiss_index")
                logger.info("벡터 스토어 생성 및 저장 완료")
                
            except Exception as embed_error:
                logger.error(f"임베딩/벡터 스토어 생성 중 에러: {embed_error}")
                logger.error(f"에러 타입: {type(embed_error)}")
                raise embed_error
            
            return len(all_docs)
            
        except Exception as e:
            logger.error(f"문서 로드 중 에러 발생: {e}")
            logger.error(f"에러 타입: {type(e)}")
            raise e 