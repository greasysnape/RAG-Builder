"""OpenAI Embeddings API 기반 임베딩 엔진"""
import os
from typing import List, Dict, Optional, Callable
from tqdm import tqdm
from .base_embedding import BaseEmbedding
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embeddings API 기반 임베딩 엔진"""

    # 모델별 차원
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self._load_model()

    def _load_model(self):
        """OpenAI 클라이언트 초기화"""
        try:
            if not self.api_key:
                logger.error("OPENAI_API_KEY가 설정되지 않았습니다")
                return

            # OpenAI 클라이언트 import 및 초기화
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI Embeddings 클라이언트 초기화 완료: {self.model_name}")
            except ImportError:
                logger.error("openai 패키지가 설치되지 않았습니다. 'pip install openai'를 실행하세요")
                return

            # 모델 차원 설정
            self.dimension = self.MODEL_DIMENSIONS.get(self.model_name, 1536)
            logger.info(f"모델 차원: {self.dimension}")

        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            self.client = None

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다")
            return [0.0] * self.dimension

        try:
            # OpenAI API 호출
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 실패: {e}")
            return [0.0] * self.dimension

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100,  # OpenAI는 최대 2048개까지 배치 가능
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> List[List[float]]:
        """여러 텍스트 배치 임베딩"""
        if not self.client:
            logger.error("OpenAI 클라이언트가 초기화되지 않았습니다")
            return [[0.0] * self.dimension] * len(texts)

        try:
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            # tqdm 진행바 (CLI 모드)
            batch_range = tqdm(
                range(0, len(texts), batch_size),
                total=total_batches,
                desc="🧠 OpenAI 임베딩 생성",
                unit="배치"
            ) if use_tqdm else range(0, len(texts), batch_size)

            for i in batch_range:
                batch = texts[i:i + batch_size]

                # OpenAI API 배치 호출
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )

                # 임베딩 추출
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # 진행상황 콜백 호출 (Streamlit용)
                if progress_callback:
                    progress_callback(min(i + batch_size, len(texts)), len(texts))

            logger.info(f"OpenAI 배치 임베딩 완료: {len(texts)} 텍스트")
            return all_embeddings

        except Exception as e:
            logger.error(f"OpenAI 배치 임베딩 실패: {e}")
            return [[0.0] * self.dimension] * len(texts)

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_loaded": self.client is not None,
            "type": "openai_embeddings",
            "provider": "OpenAI",
            "api_key_set": bool(self.api_key)
        }

    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.client is not None and bool(self.api_key)
