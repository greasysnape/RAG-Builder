"""벡터 스토어 추상 기본 클래스"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseVectorStore(ABC):
    """벡터 스토어 추상 클래스"""

    def __init__(self, collection_name: str, **kwargs):
        self.collection_name = collection_name

    @abstractmethod
    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> bool:
        """문서를 벡터 스토어에 추가"""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """유사도 검색"""
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> bool:
        """ID로 문서 삭제"""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """컬렉션 전체 삭제"""
        pass

    @abstractmethod
    def check_duplicate_embeddings(self, file_paths: List[str]) -> Dict:
        """중복 임베딩 체크"""
        pass

    @abstractmethod
    def get_embedded_hashes(self) -> List[str]:
        """임베딩된 파일의 해시 목록"""
        pass

    @abstractmethod
    def delete_by_url_hash(self, url_hash: str) -> bool:
        """URL 해시로 문서 삭제"""
        pass

    # 정적 메서드들 (컬렉션 관리)
    @staticmethod
    @abstractmethod
    def list_all_collections() -> List[str]:
        """모든 컬렉션 목록"""
        pass

    @staticmethod
    @abstractmethod
    def get_all_collections_info() -> List[Dict]:
        """모든 컬렉션 상세 정보"""
        pass

    @staticmethod
    @abstractmethod
    def get_database_size() -> Dict:
        """데이터베이스 전체 크기"""
        pass
