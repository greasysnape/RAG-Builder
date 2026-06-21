from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TextChunk:
    content: str
    metadata: Dict
    source: str
    chunk_id: int

class TextSplitter:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitters = {
            "ko": RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", " ", ""]
            ),
            "en": RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }

    def split_document(self, processed_doc, chunk_strategy: str = "auto") -> List[TextChunk]:
        """단일 문서를 청크로 분할"""
        language = processed_doc.language
        splitter = self.splitters.get(language, self.splitters["en"])

        if chunk_strategy == "auto":
            strategy = self._determine_chunk_strategy(processed_doc)
        else:
            strategy = chunk_strategy

        chunks = self._split_by_strategy(processed_doc, splitter, strategy)
        return chunks

    def split_documents(self, processed_docs: List) -> List[TextChunk]:
        """여러 문서를 배치 분할"""
        all_chunks = []

        for doc in processed_docs:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"문서 분할 완료: {doc.original_url} -> {len(chunks)} 청크")

        return all_chunks

    def _determine_chunk_strategy(self, doc) -> str:
        """문서 특성에 따른 청크 전략 결정"""
        word_count = doc.word_count

        if word_count < 500:
            return "small"
        elif word_count > 3000:
            return "large"
        else:
            return "medium"

    def _split_by_strategy(self, doc, splitter, strategy: str) -> List[TextChunk]:
        """전략에 따른 문서 분할"""
        if strategy == "small":
            chunk_size = self.chunk_size // 2
        elif strategy == "large":
            chunk_size = self.chunk_size * 2
        else:
            chunk_size = self.chunk_size

        splitter.chunk_size = chunk_size
        text_chunks = splitter.split_text(doc.content)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(TextChunk(
                content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "chunk_strategy": strategy,
                    "title": doc.title,
                    "language": doc.language
                },
                source=doc.original_url,
                chunk_id=i
            ))

        return chunks

    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """간단한 텍스트 분할 메서드"""
        if metadata is None:
            metadata = {}

        # 기본 영어 분할기 사용
        splitter = self.splitters["en"]
        text_chunks = splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # URL 해시 기반 ID 생성 (중복 방지)
            url_hash = metadata.get('url_hash')
            if url_hash:
                # 형식: {url_hash}_{chunk_index}
                chunk_id = f"{url_hash}_{i}"
            else:
                # 폴백: 기존 방식
                chunk_id = f"{metadata.get('source', 'unknown')}_{i}"

            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "chunk_size": len(chunk_text)
                }
            })

        logger.info(f"텍스트 분할 완료: {len(text_chunks)}개 청크 생성")
        return chunks