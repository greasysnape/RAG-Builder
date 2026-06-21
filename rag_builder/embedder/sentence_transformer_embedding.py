"""Sentence Transformers 기반 임베딩 엔진"""
import numpy as np
from typing import List, Dict, Optional, Callable
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .base_embedding import BaseEmbedding
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """sentence-transformers 기반 로컬 임베딩 엔진"""

    def __init__(self, model_name: str = None):
        super().__init__(model_name or settings.local_embedding_model)
        self.model = None
        self._load_model()

    def _load_model(self):
        """임베딩 모델 로드"""
        try:
            logger.info(f"Sentence Transformer 모델 로드 중: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # 모델 차원 확인
            test_embedding = self.model.encode(["test"])
            self.dimension = len(test_embedding[0])


            logger.info(f"모델 로드 완료 - 차원: {self.dimension}")

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            # 폴백 모델 시도
            try:
                logger.info("폴백 모델 시도: all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
                test_embedding = self.model.encode(["test"])
                self.dimension = len(test_embedding[0])
                logger.info(f"폴백 모델 로드 완료 - 차원: {self.dimension}")
            except Exception as e2:
                logger.error(f"폴백 모델 로드 실패: {e2}")
                self.model = None
                self.dimension = 384  # 기본값

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        if not self.model:
            logger.error("임베딩 모델이 로드되지 않았습니다")
            return [0.0] * self.dimension

        try:
            embedding = self.model.encode([text])
            return embedding[0].tolist()

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return [0.0] * self.dimension

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> List[List[float]]:
        """여러 텍스트 배치 임베딩"""
        if not self.model:
            logger.error("임베딩 모델이 로드되지 않았습니다")
            return [[0.0] * self.dimension] * len(texts)

        try:
            # 배치 처리
            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            # tqdm 진행바 (CLI 모드)
            batch_range = tqdm(
                range(0, len(texts), batch_size),
                total=total_batches,
                desc="🧠 임베딩 생성",
                unit="배치"
            ) if use_tqdm else range(0, len(texts), batch_size)

            for batch_idx, i in enumerate(batch_range):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.extend([emb.tolist() for emb in embeddings])

                # 진행상황 콜백 호출 (Streamlit용)
                if progress_callback:
                    progress_callback(min(i + batch_size, len(texts)), len(texts))

            logger.info(f"배치 임베딩 완료: {len(texts)} 텍스트")
            return all_embeddings

        except Exception as e:
            logger.error(f"배치 임베딩 실패: {e}")
            return [[0.0] * self.dimension] * len(texts)

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """코사인 유사도 계산"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_loaded": self.model is not None,
            "type": "sentence_transformer",
            "provider": "Local"
        }

    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.model is not None


class MultilingualEmbedding(SentenceTransformerEmbedding):
    """다국어 지원 임베딩 엔진"""

    def __init__(self):
        # 다국어 모델 사용
        super().__init__("paraphrase-multilingual-MiniLM-L12-v2")
