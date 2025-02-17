from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings import LlamaCppEmbeddings
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

        self.embeddings = LlamaCppEmbeddings(
            model_path=os.getenv("LLAMA_MODEL_PATH"),
            n_ctx=2048,         # 컨텍스트 길이
            n_batch=512,        # 배치 크기
            n_gpu_layers=0,     # CPU만 사용
            verbose=True        # 디버깅을 위한 로그
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