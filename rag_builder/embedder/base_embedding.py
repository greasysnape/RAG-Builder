"""임베딩 엔진 추상 기본 클래스"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable


class BaseEmbedding(ABC):
    """임베딩 엔진 추상 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension = None

    @abstractmethod
    def _load_model(self):
        """모델 로드 (서브클래스에서 구현)"""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        pass

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> List[List[float]]:
        """여러 텍스트 배치 임베딩"""
        pass

    def embed_chunks(
        self,
        text_chunks: List,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> List[Dict]:
        """텍스트 청크들을 임베딩하여 벡터 데이터 생성"""
        texts = [chunk.content for chunk in text_chunks]
        embeddings = self.embed_texts(texts, progress_callback=progress_callback, use_tqdm=use_tqdm)

        vector_data = []
        for chunk, embedding in zip(text_chunks, embeddings):
            vector_data.append({
                "id": f"{hash(chunk.source)}_{chunk.chunk_id}",
                "content": chunk.content,
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "embedding_model": self.model_name,
                    "embedding_dimension": len(embedding)
                },
                "source": chunk.source
            })

        return vector_data

    @abstractmethod
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        pass

    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.dimension
