"""벡터 스토어 팩토리"""
from typing import Optional
from .base_vector_store import BaseVectorStore
from .chroma_vector_store import ChromaVectorStore
from .faiss_vector_store import FAISSVectorStore
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """벡터 스토어를 생성하는 팩토리 클래스"""

    # 지원하는 벡터 DB 타입
    VECTOR_STORE_TYPES = {
        "chromadb": "ChromaDB (로컬, 메타데이터 풍부)",
        "faiss": "FAISS (로컬, 초고속)"
    }

    @classmethod
    def create_vector_store(
        cls,
        store_type: str = None,
        collection_name: str = "rag_documents",
        **kwargs
    ) -> BaseVectorStore:
        """
        벡터 스토어 생성

        Args:
            store_type: 벡터 스토어 타입 ("chromadb" 또는 "faiss")
            collection_name: 컬렉션명
            **kwargs: 추가 인자 (persist_directory, dimension 등)

        Returns:
            BaseVectorStore: 벡터 스토어 인스턴스
        """
        # 기본값 설정
        if store_type is None:
            store_type = getattr(settings, 'vector_db_type', 'chromadb')

        logger.info(f"벡터 스토어 생성: type={store_type}, collection={collection_name}")

        try:
            if store_type == "chromadb":
                return ChromaVectorStore(collection_name, **kwargs)
            elif store_type == "faiss":
                return FAISSVectorStore(collection_name, **kwargs)
            else:
                logger.error(f"지원하지 않는 벡터 스토어 타입: {store_type}")
                logger.info("기본 ChromaDB로 폴백")
                return ChromaVectorStore(collection_name, **kwargs)

        except Exception as e:
            logger.error(f"벡터 스토어 생성 실패: {e}")
            logger.info("기본 ChromaDB로 폴백")
            return ChromaVectorStore(collection_name, **kwargs)

    @classmethod
    def get_available_types(cls) -> dict:
        """
        사용 가능한 벡터 스토어 타입 목록 반환

        Returns:
            dict: 타입명과 설명을 담은 딕셔너리
        """
        return cls.VECTOR_STORE_TYPES

    @classmethod
    def get_type_info(cls, store_type: str) -> Optional[str]:
        """
        특정 벡터 스토어 타입의 정보 반환

        Args:
            store_type: 벡터 스토어 타입

        Returns:
            str: 타입 설명 (없으면 None)
        """
        return cls.VECTOR_STORE_TYPES.get(store_type)

    @classmethod
    def is_type_available(cls, store_type: str) -> bool:
        """
        특정 벡터 스토어 타입이 사용 가능한지 확인

        Args:
            store_type: 벡터 스토어 타입

        Returns:
            bool: 사용 가능 여부
        """
        if store_type == "chromadb":
            try:
                import chromadb
                return True
            except ImportError:
                return False
        elif store_type == "faiss":
            try:
                import faiss
                return True
            except ImportError:
                return False
        return False

    @classmethod
    def list_all_collections(cls, store_type: str = None) -> list:
        """
        모든 컬렉션 목록 가져오기

        Args:
            store_type: 벡터 스토어 타입 (기본값: settings.vector_db_type)

        Returns:
            list: 컬렉션명 목록
        """
        if store_type is None:
            store_type = getattr(settings, 'vector_db_type', 'chromadb')

        try:
            if store_type == "chromadb":
                return ChromaVectorStore.list_all_collections()
            elif store_type == "faiss":
                return FAISSVectorStore.list_all_collections()
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []

    @classmethod
    def get_all_collections_info(cls, store_type: str = None) -> list:
        """
        모든 컬렉션의 상세 정보 가져오기

        Args:
            store_type: 벡터 스토어 타입 (기본값: settings.vector_db_type)

        Returns:
            list: 컬렉션 정보 딕셔너리 목록
        """
        if store_type is None:
            store_type = getattr(settings, 'vector_db_type', 'chromadb')

        try:
            if store_type == "chromadb":
                return ChromaVectorStore.get_all_collections_info()
            elif store_type == "faiss":
                return FAISSVectorStore.get_all_collections_info()
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return []

    @classmethod
    def get_database_size(cls, store_type: str = None) -> dict:
        """
        데이터베이스 크기 정보 가져오기

        Args:
            store_type: 벡터 스토어 타입 (기본값: settings.vector_db_type)

        Returns:
            dict: 크기 정보
        """
        if store_type is None:
            store_type = getattr(settings, 'vector_db_type', 'chromadb')

        try:
            if store_type == "chromadb":
                return ChromaVectorStore.get_database_size()
            elif store_type == "faiss":
                return FAISSVectorStore.get_database_size()
        except Exception as e:
            logger.error(f"DB 크기 조회 실패: {e}")
            return {}
