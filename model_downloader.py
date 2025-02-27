import os
import logging
import requests
import re
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files, hf_hub_url
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
            
            # 분할된 GGUF 파일 패턴 확인
            split_patterns = [
                # 패턴 1: filename.gguf.part-1, filename.gguf.part-2 등
                r'(.+\.gguf)\.part-\d+$',
                # 패턴 2: filename-00001-of-00003.gguf 등
                r'(.+)-\d{5}-of-\d{5}\.gguf$'
            ]
            
            # 분할 파일 그룹 식별
            split_file_groups = self._identify_split_files(gguf_files, split_patterns)
            
            # 분할 파일이 있으면 처리
            if split_file_groups:
                logger.info(f"분할된 GGUF 파일 그룹 {len(split_file_groups)}개 발견")
                
                # 첫 번째 그룹 사용 (여러 모델이 있을 경우)
                base_name = list(split_file_groups.keys())[0]
                parts = sorted(split_file_groups[base_name])
                
                logger.info(f"분할된 GGUF 파일 병합 시작: {base_name} ({len(parts)}개 파트)")
                merged_file = self._download_and_merge_parts(repo_id, parts, base_name, local_dir)
                
                if merged_file:
                    return os.path.join(repo_name, os.path.basename(merged_file))
                    
                logger.error(f"분할 파일 병합 실패")
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

    def _identify_split_files(self, file_list, patterns):
        """
        파일 목록에서 분할된 파일 그룹을 식별합니다.
        
        Args:
            file_list: 파일 이름 목록
            patterns: 분할 파일 패턴 정규식 목록
            
        Returns:
            dict: {base_name: [part_files]} 형태의 딕셔너리
        """
        split_files = {}
        
        for pattern in patterns:
            for file in file_list:
                match = re.match(pattern, file)
                if match:
                    base_name = match.group(1)
                    if base_name not in split_files:
                        split_files[base_name] = []
                    split_files[base_name].append(file)
        
        # 실제로 분할 파일인지 확인 (2개 이상의 파일이 있는 경우만)
        return {k: v for k, v in split_files.items() if len(v) > 1}

    def _download_and_merge_parts(self, repo_id, part_files, base_name, local_dir):
        """
        분할된 파일들을 다운로드하고 병합합니다.
        
        Args:
            repo_id: Hugging Face 레포지토리 ID
            part_files: 분할 파일 목록
            base_name: 기본 파일 이름
            local_dir: 저장할 로컬 디렉토리
            
        Returns:
            str: 병합된 파일 경로 또는 None
        """
        try:
            # 병합 파일 경로
            merged_filename = f"{Path(base_name).stem}.gguf"
            merged_file_path = os.path.join(local_dir, merged_filename)
            
            # 이미 병합된 파일이 있는지 확인
            if os.path.exists(merged_file_path):
                logger.info(f"병합된 파일이 이미 존재합니다: {merged_file_path}")
                return merged_file_path
            
            # 임시 디렉토리 생성
            temp_dir = os.path.join(local_dir, "temp_parts")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 모든 파트 다운로드
            downloaded_parts = []
            for part in part_files:
                part_path = os.path.join(temp_dir, os.path.basename(part))
                if not os.path.exists(part_path):
                    logger.info(f"파트 다운로드 중: {part}")
                    url = hf_hub_url(repo_id, part)
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(part_path, 'wb') as f:
                        total_size = int(response.headers.get('content-length', 0))
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(part)) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                
                downloaded_parts.append(part_path)
            
            # 파일 병합
            logger.info(f"파일 병합 중: {merged_file_path}")
            with open(merged_file_path, 'wb') as outfile:
                for part_path in sorted(downloaded_parts):
                    logger.info(f"병합 중: {os.path.basename(part_path)}")
                    with open(part_path, 'rb') as infile:
                        while True:
                            chunk = infile.read(8192)
                            if not chunk:
                                break
                            outfile.write(chunk)
            
            logger.info(f"파일 병합 완료: {merged_file_path}")
            
            # 성공적으로 병합되었으면 임시 파일 삭제 (옵션)
            # import shutil
            # shutil.rmtree(temp_dir)
            
            return merged_file_path
            
        except Exception as e:
            logger.error(f"분할 파일 다운로드 및 병합 중 오류 발생: {e}")
            return None

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