"""임베딩 엔진 팩토리"""
from typing import Optional
from .base_embedding import BaseEmbedding
from .sentence_transformer_embedding import SentenceTransformerEmbedding, MultilingualEmbedding
from .openai_embedding import OpenAIEmbedding
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """임베딩 엔진을 생성하는 팩토리 클래스"""

    # 지원하는 임베딩 타입
    EMBEDDING_TYPES = {
        "sentence_transformer": "Sentence Transformers (로컬)",
        "openai": "OpenAI Embeddings API",
    }

    # Sentence Transformer 사전 정의 모델
    SENTENCE_TRANSFORMER_MODELS = {
        "all-MiniLM-L6-v2": {"dimension": 384, "description": "빠르고 가벼운 모델 (영어)"},
        "all-mpnet-base-v2": {"dimension": 768, "description": "고품질 범용 모델 (영어)"},
        "paraphrase-multilingual-MiniLM-L12-v2": {"dimension": 384, "description": "다국어 지원 모델"},
    }

    # OpenAI 사전 정의 모델
    OPENAI_MODELS = {
        "text-embedding-3-small": {"dimension": 1536, "description": "빠르고 저렴한 모델"},
        "text-embedding-3-large": {"dimension": 3072, "description": "최고 품질 모델"},
        "text-embedding-ada-002": {"dimension": 1536, "description": "레거시 모델"},
    }

    @classmethod
    def create_embedding(
        cls,
        embedding_type: str = None,
        model_name: str = None,
        **kwargs
    ) -> BaseEmbedding:
        """
        임베딩 엔진 생성

        Args:
            embedding_type: 임베딩 타입 ("sentence_transformer" 또는 "openai")
            model_name: 모델명
            **kwargs: 추가 인자

        Returns:
            BaseEmbedding: 임베딩 엔진 인스턴스
        """
        # 기본값 설정
        if embedding_type is None:
            embedding_type = getattr(settings, 'embedding_type', 'sentence_transformer')

        if model_name is None:
            model_name = settings.local_embedding_model

        logger.info(f"임베딩 엔진 생성: type={embedding_type}, model={model_name}")

        try:
            if embedding_type == "sentence_transformer":
                return cls._create_sentence_transformer(model_name, **kwargs)
            elif embedding_type == "openai":
                return cls._create_openai(model_name, **kwargs)
            else:
                logger.error(f"지원하지 않는 임베딩 타입: {embedding_type}")
                logger.info("기본 Sentence Transformer로 폴백")
                return SentenceTransformerEmbedding(model_name)

        except Exception as e:
            logger.error(f"임베딩 엔진 생성 실패: {e}")
            logger.info("기본 Sentence Transformer로 폴백")
            return SentenceTransformerEmbedding("all-MiniLM-L6-v2")

    @classmethod
    def _create_sentence_transformer(cls, model_name: str, **kwargs) -> SentenceTransformerEmbedding:
        """Sentence Transformer 임베딩 엔진 생성"""
        # 특수 모델 처리
        if model_name == "paraphrase-multilingual-MiniLM-L12-v2":
            return MultilingualEmbedding()
        else:
            return SentenceTransformerEmbedding(model_name)

    @classmethod
    def _create_openai(cls, model_name: str, **kwargs) -> OpenAIEmbedding:
        """OpenAI 임베딩 엔진 생성"""
        api_key = kwargs.get('api_key')
        return OpenAIEmbedding(model_name, api_key)

    @classmethod
    def get_available_models(cls, embedding_type: str = "sentence_transformer") -> dict:
        """
        사용 가능한 모델 목록 반환

        Args:
            embedding_type: 임베딩 타입

        Returns:
            dict: 모델명과 정보를 담은 딕셔너리
        """
        if embedding_type == "sentence_transformer":
            return cls.SENTENCE_TRANSFORMER_MODELS
        elif embedding_type == "openai":
            return cls.OPENAI_MODELS
        else:
            return {}

    @classmethod
    def get_model_info(cls, embedding_type: str, model_name: str) -> Optional[dict]:
        """
        특정 모델의 정보 반환

        Args:
            embedding_type: 임베딩 타입
            model_name: 모델명

        Returns:
            dict: 모델 정보 (없으면 None)
        """
        models = cls.get_available_models(embedding_type)
        return models.get(model_name)
