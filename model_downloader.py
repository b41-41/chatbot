import os
import logging
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class ModelDownloader:
    def __init__(self):
        self.models_dir = os.path.join(os.getcwd(), "models")
        
        # models 디렉토리가 없으면 생성
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            logger.info(f"models 디렉토리 생성: {self.models_dir}")

    def check_and_download_models(self):
        """
        .env 파일에 지정된 모델이 models 디렉토리에 있는지 확인하고
        없으면 자동으로 다운로드합니다.
        """
        logger.info("모델 파일 확인 및 다운로드 시작...")
        
        # LLM 모델 확인 및 다운로드
        llama_model = os.getenv("LLAMA_MODEL")
        if llama_model:
            self._check_and_download_llm(llama_model)
        else:
            logger.warning("LLAMA_MODEL이 .env 파일에 설정되지 않았습니다.")
        
        # 임베딩 모델 확인 및 다운로드
        embedding_model = os.getenv("EMBEDDING_MODEL")
        if embedding_model:
            self._check_and_download_embedding(embedding_model)
        else:
            logger.warning("EMBEDDING_MODEL이 .env 파일에 설정되지 않았습니다.")
            
        logger.info("모델 확인 및 다운로드 완료")

    def _is_hf_repo_path(self, path):
        """
        주어진 경로가 Hugging Face 리포지토리 경로인지 확인합니다.
        예: unsloth/DeepSeek-R1-GGUF
        """
        # '/'를 포함하고 확장자가 없으면 HF 리포지토리 경로로 간주
        return '/' in path and '.' not in path.split('/')[-1]

    def _check_and_download_llm(self, model_path):
        """
        LLM 모델 파일을 확인하고 없으면 다운로드
        model_path는 파일명이거나 Hugging Face 레포지토리 경로일 수 있습니다.
        """
        if self._is_hf_repo_path(model_path):
            # Hugging Face 레포지토리 경로인 경우
            return self._download_from_hf_repo(model_path)
        else:
            # 일반 파일명인 경우
            full_path = os.path.join(self.models_dir, model_path)
            
            if os.path.exists(full_path):
                logger.info(f"LLM 모델 파일이 이미 존재합니다: {full_path}")
                return True
                
            logger.info(f"LLM 모델 파일이 없습니다. 다운로드를 시작합니다: {model_path}")
            
            # 기본 GGUF 모델 다운로드 URL - llama2-7b-chat
            if "llama-2-7b-chat.Q4_K_M.gguf" in model_path:
                model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
                return self._download_file(model_url, full_path)
            else:
                # 다른 모델은 Hugging Face에서 다운로드 시도
                try:
                    model_name = model_path.split('.')[0]  # 파일 확장자 제거
                    logger.info(f"Hugging Face에서 {model_name} 모델 다운로드 시도...")
                    
                    # TheBloke의 GGUF 모델 리포지토리에서 검색
                    repo_id = f"TheBloke/{model_name}-GGUF"
                    
                    # 모델 파일 다운로드
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=self.models_dir,
                        allow_patterns=[f"*{model_path}*"],
                    )
                    
                    logger.info(f"모델 다운로드 완료: {model_path}")
                    return True
                except Exception as e:
                    logger.error(f"모델 다운로드 중 오류 발생: {e}")
                    logger.warning(f"모델을 수동으로 다운로드하여 {full_path}에 저장해주세요.")
                    return False

    def _download_from_hf_repo(self, repo_id):
        """
        Hugging Face 레포지토리에서 GGUF 모델 파일을 다운로드합니다.
        적절한 GGUF 파일을 자동으로 선택합니다.
        """
        try:
            # 저장할 디렉토리 생성
            repo_name = repo_id.split('/')[-1]
            local_dir = os.path.join(self.models_dir, repo_name)
            os.makedirs(local_dir, exist_ok=True)
            
            logger.info(f"Hugging Face 레포지토리 {repo_id}에서 모델 파일 목록 가져오는 중...")
            
            # 레포지토리의 파일 목록 확인
            files = list_repo_files(repo_id)
            
            # GGUF 파일 찾기
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                logger.error(f"레포지토리 {repo_id}에서 GGUF 파일을 찾을 수 없습니다.")
                return False
            
            # 적절한 모델 파일 선택 (우선순위: Q4_K_M.gguf > Q5_K_M.gguf > 첫 번째 GGUF)
            selected_file = None
            
            # 중간 크기의 양자화 모델 찾기
            for pattern in ["Q4_K_M.gguf", "Q5_K_M.gguf"]:
                matches = [f for f in gguf_files if pattern in f]
                if matches:
                    selected_file = matches[0]
                    logger.info(f"선택된 양자화 모델: {selected_file}")
                    break
            
            # 없으면 첫 번째 GGUF 파일 사용
            if not selected_file:
                selected_file = gguf_files[0]
                logger.info(f"양자화된 모델을 찾을 수 없어 첫 번째 GGUF 파일 사용: {selected_file}")
            
            # 선택된 파일이 이미 다운로드되어 있는지 확인
            local_file_path = os.path.join(local_dir, os.path.basename(selected_file))
            if os.path.exists(local_file_path):
                logger.info(f"모델 파일이 이미 존재합니다: {local_file_path}")
                # 환경 변수에 실제 파일 경로 저장을 위해 반환
                return os.path.join(repo_name, os.path.basename(selected_file))
            
            # 파일 다운로드
            logger.info(f"모델 파일 다운로드 중: {selected_file}")
            hf_hub_download(
                repo_id=repo_id,
                filename=selected_file,
                local_dir=local_dir
            )
            
            logger.info(f"모델 파일 다운로드 완료: {local_file_path}")
            # 환경 변수에 실제 파일 경로 저장을 위해 반환
            return os.path.join(repo_name, os.path.basename(selected_file))
            
        except Exception as e:
            logger.error(f"Hugging Face 레포지토리에서 모델 다운로드 중 오류 발생: {e}")
            return False

    def _check_and_download_embedding(self, model_name):
        """
        임베딩 모델을 확인하고 없으면 다운로드
        """
        # Hugging Face 레포지토리 경로인지 확인
        if self._is_hf_repo_path(model_name):
            # 임베딩 모델은 레포지토리 전체를 다운로드
            repo_name = model_name.split('/')[-1]
            full_path = os.path.join(self.models_dir, repo_name)
            
            if os.path.exists(full_path):
                logger.info(f"임베딩 모델이 이미 존재합니다: {full_path}")
                return True
                
            logger.info(f"임베딩 모델이 없습니다. 다운로드를 시작합니다: {model_name}")
            
            try:
                # 모델 다운로드
                snapshot_download(
                    repo_id=model_name,
                    local_dir=full_path
                )
                
                logger.info(f"임베딩 모델 다운로드 완료: {model_name}")
                return True
            except Exception as e:
                logger.error(f"임베딩 모델 다운로드 중 오류 발생: {e}")
                logger.warning(f"임베딩 모델을 수동으로 다운로드하여 {full_path}에 저장해주세요.")
                return False
        else:
            # 일반 모델명인 경우 기존 로직 사용
            full_path = os.path.join(self.models_dir, model_name)
            
            if os.path.exists(full_path):
                logger.info(f"임베딩 모델이 이미 존재합니다: {full_path}")
                return True
                
            logger.info(f"임베딩 모델이 없습니다. 다운로드를 시작합니다: {model_name}")
            
            try:
                # 'sentence-transformers/' 접두사 포함 여부 확인 및 추가
                if not model_name.startswith("sentence-transformers/") and model_name == "all-MiniLM-L6-v2":
                    repo_id = f"sentence-transformers/{model_name}"
                else:
                    repo_id = model_name
                    
                logger.info(f"Hugging Face에서 {repo_id} 임베딩 모델 다운로드 중...")
                
                # 모델 다운로드
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=os.path.join(self.models_dir, model_name)
                )
                
                logger.info(f"임베딩 모델 다운로드 완료: {model_name}")
                return True
            except Exception as e:
                logger.error(f"임베딩 모델 다운로드 중 오류 발생: {e}")
                logger.warning(f"임베딩 모델을 수동으로 다운로드하여 {full_path}에 저장해주세요.")
                return False

    def _download_file(self, url, destination):
        """
        파일을 다운로드하고 진행률을 표시합니다.
        """
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # 파일이 이미 존재하는지 확인
            if os.path.exists(destination):
                existing_size = os.path.getsize(destination)
                if existing_size == total_size:
                    logger.info(f"파일이 이미 완전히 다운로드되어 있습니다: {destination}")
                    return True
                else:
                    logger.info(f"파일이 일부만 다운로드되어 있습니다. 새로 다운로드합니다.")
            
            # 경로가 없으면 생성
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # 다운로드 시작
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"파일 다운로드 완료: {destination}")
            return True
        except Exception as e:
            logger.error(f"파일 다운로드 중 오류 발생: {e}")
            return False 