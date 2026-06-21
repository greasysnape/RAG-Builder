"""
벡터 스토어 (하위 호환성 유지)

기존 코드 호환성을 위해 VectorStore를 ChromaVectorStore의 별칭으로 유지
새 코드에서는 vector_store_factory.py의 VectorStoreFactory 사용 권장
"""
from .chroma_vector_store import ChromaVectorStore

# 하위 호환성을 위한 별칭
VectorStore = ChromaVectorStore

__all__ = ['VectorStore', 'ChromaVectorStore']
